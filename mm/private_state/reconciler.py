from __future__ import annotations

import asyncio
from typing import Literal

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.private_state import AbstractPrivateStateAdapter
from mm.core.models.execution import Fill
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.exchanges.shared.private_state import RestPollingPrivateStateAdapter
from mm.private_state.store import PrivateStateStore

FillEventSource = Literal["exchange_fill_history", "open_order_delta"]


class PrivateStateReconciler:
    """Keeps private exchange state in one view.

    For ``RestPollingPrivateStateAdapter`` (Nobitex/Wallex REST), a **single** poll
    loop applies one full ``PrivateStateSnapshot`` per interval (balances + open
    orders from the same ``snapshot()`` call). Other adapters use incremental
    balance and open-order streams.
    """

    def __init__(
        self,
        exchange: str,
        adapter: AbstractPrivateStateAdapter,
        bus: AbstractEventBus | None = None,
        store: PrivateStateStore | None = None,
        fill_event_source: FillEventSource = "exchange_fill_history",
    ):
        self.exchange = exchange.lower().strip()
        self.adapter = adapter
        self.bus = bus
        self.store = store or PrivateStateStore(
            self.exchange,
            trading_symbol=getattr(adapter, "symbol", None),
        )
        if fill_event_source not in ("exchange_fill_history", "open_order_delta"):
            raise ValueError(
                "fill_event_source must be 'exchange_fill_history' or 'open_order_delta'"
            )
        self.fill_event_source = fill_event_source
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        await self.adapter.connect()
        self._tasks.append(asyncio.create_task(self._publish_initial_snapshot()))
        if isinstance(self.adapter, RestPollingPrivateStateAdapter):
            # One authoritative snapshot per poll: balances + open orders from the same fetch.
            # Avoids incremental diff streams that inferred synthetic zeros when an asset vanished
            # from a partial or flaky REST response.
            self._tasks.extend([
                asyncio.create_task(self._listen_rest_polling_unified()),
                asyncio.create_task(self._listen_order_updates()),
            ])
            if self.fill_event_source == "exchange_fill_history":
                self._tasks.append(asyncio.create_task(self._listen_fills()))
        else:
            self._tasks.extend([
                asyncio.create_task(self._listen_balances()),
                asyncio.create_task(self._listen_open_orders()),
                asyncio.create_task(self._listen_order_updates()),
            ])
            if self.fill_event_source == "exchange_fill_history":
                self._tasks.append(asyncio.create_task(self._listen_fills()))
        if hasattr(self.adapter, "error_stream"):
            self._tasks.append(asyncio.create_task(self._listen_errors()))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.adapter.disconnect()

    def snapshot(self) -> PrivateStateSnapshot:
        return self.store.snapshot()

    async def _publish_initial_snapshot(self) -> None:
        try:
            snapshot = await self.adapter.snapshot()
            await self._publish_snapshot(self.store.apply_snapshot(snapshot))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._publish_adapter_error("initial_snapshot", exc)

    async def _listen_rest_polling_unified(self) -> None:
        poll_ms = max(500, int(getattr(self.adapter, "poll_interval_ms", 5000)))
        publish_errors = getattr(self.adapter, "_publish_error", None)
        try:
            while True:
                await asyncio.sleep(poll_ms / 1000.0)
                try:
                    snap_in = await self.adapter.snapshot()
                    fills = _fills_from_open_order_delta(
                        self.store.snapshot().open_orders,
                        snap_in.open_orders,
                    )
                    if self.fill_event_source == "open_order_delta":
                        for fill in fills:
                            await self._publish(f"private:state:{self.exchange}:fills", fill)
                    merged = self.store.apply_snapshot(snap_in)
                    await self._publish_snapshot(merged)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if callable(publish_errors):
                        await publish_errors("unified_rest_poll", exc)  # type: ignore[misc]
                    else:
                        await self._publish_adapter_error("unified_rest_poll", exc)
        except asyncio.CancelledError:
            raise

    async def _listen_balances(self) -> None:
        try:
            async for balance in self.adapter.balance_stream():
                snapshot = self.store.apply_balance(balance)
                await self._publish(f"private:state:{self.exchange}:balances", balance)
                await self._publish_snapshot(snapshot)
        except asyncio.CancelledError:
            raise

    async def _listen_open_orders(self) -> None:
        try:
            async for open_order in self.adapter.open_order_stream():
                snapshot = self.store.apply_open_order(open_order)
                await self._publish(f"private:state:{self.exchange}:open_orders", open_order)
                await self._publish_snapshot(snapshot)
        except asyncio.CancelledError:
            raise

    async def _listen_order_updates(self) -> None:
        try:
            async for update in self.adapter.order_update_stream():
                snapshot = self.store.apply_order_update(update)
                await self._publish(f"private:state:{self.exchange}:order_updates", update)
                await self._publish_snapshot(snapshot)
        except asyncio.CancelledError:
            raise

    async def _listen_fills(self) -> None:
        try:
            async for fill in self.adapter.fill_stream():
                snapshot = self.store.apply_fill(fill)
                await self._publish(f"private:state:{self.exchange}:fills", fill)
                await self._publish_snapshot(snapshot)
        except asyncio.CancelledError:
            raise

    async def _listen_errors(self) -> None:
        try:
            async for error in self.adapter.error_stream():  # type: ignore[attr-defined]
                await self._publish(f"private:state:{self.exchange}:errors", error)
        except asyncio.CancelledError:
            raise

    async def _publish_snapshot(self, snapshot: PrivateStateSnapshot) -> None:
        await self._publish(f"private:state:{self.exchange}:snapshot", snapshot)

    async def _publish_adapter_error(self, source: str, exc: Exception) -> None:
        publish_errors = getattr(self.adapter, "_publish_error", None)
        if callable(publish_errors):
            await publish_errors(source, exc)  # type: ignore[misc]
            return
        await self._publish(
            f"private:state:{self.exchange}:errors",
            {
                "source": source,
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )

    async def _publish(self, topic: str, event: Balance | OpenOrder | OrderUpdate | Fill | PrivateStateSnapshot | dict) -> None:
        if self.bus is not None:
            await self.bus.publish(topic, event)


def _fills_from_open_order_delta(
    previous: dict[str, OpenOrder],
    current: dict[str, OpenOrder],
) -> tuple[Fill, ...]:
    fills: list[Fill] = []
    for order_id, current_order in current.items():
        previous_order = previous.get(order_id)
        if previous_order is None:
            continue
        filled_delta = max(0.0, float(current_order.filled_qty) - float(previous_order.filled_qty))
        remaining_delta = max(0.0, float(previous_order.remaining_qty) - float(current_order.remaining_qty))
        qty = max(filled_delta, remaining_delta)
        if qty <= 1e-12:
            continue
        fills.append(
            Fill(
                exchange_order_id=order_id,
                actor_id=current_order.actor_id or previous_order.actor_id or "",
                exchange=current_order.order.exchange,
                symbol=current_order.order.symbol,
                side=current_order.order.side,
                price=float(current_order.order.price),
                qty=qty,
                fee=0.0,
                timestamp_ms=int(current_order.timestamp_ms),
            )
        )
    return tuple(fills)
