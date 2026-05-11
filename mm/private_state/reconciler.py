from __future__ import annotations

import asyncio

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.private_state import AbstractPrivateStateAdapter
from mm.core.models.execution import Fill
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.private_state.store import PrivateStateStore


class PrivateStateReconciler:
    """Keeps private REST snapshots and private event streams in one state view."""

    def __init__(
        self,
        exchange: str,
        adapter: AbstractPrivateStateAdapter,
        bus: AbstractEventBus | None = None,
        store: PrivateStateStore | None = None,
    ):
        self.exchange = exchange.lower().strip()
        self.adapter = adapter
        self.bus = bus
        self.store = store or PrivateStateStore(
            self.exchange,
            trading_symbol=getattr(adapter, "symbol", None),
        )
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        await self.adapter.connect()
        snapshot = await self.adapter.snapshot()
        await self._publish_snapshot(self.store.apply_snapshot(snapshot))
        self._tasks = [
            asyncio.create_task(self._listen_balances()),
            asyncio.create_task(self._listen_open_orders()),
            asyncio.create_task(self._listen_order_updates()),
            asyncio.create_task(self._listen_fills()),
        ]
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

    async def _publish(self, topic: str, event: Balance | OpenOrder | OrderUpdate | Fill | PrivateStateSnapshot | dict) -> None:
        if self.bus is not None:
            await self.bus.publish(topic, event)
