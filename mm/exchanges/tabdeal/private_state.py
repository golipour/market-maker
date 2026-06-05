from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import websockets

from mm.core.models.execution import Fill, Order
from mm.core.models.private import OpenOrder, OrderUpdate
from mm.exchanges.instrument_wire import build_instrument_wire
from mm.exchanges.shared.private_state import RestPollingPrivateStateAdapter
from mm.exchanges.shared.provider_adapter import now_ms
from mm.exchanges.tabdeal import private_rest


class TabdealRestPrivateStateAdapter(RestPollingPrivateStateAdapter):
    def __init__(
        self,
        symbol: str,
        *,
        api_token: Any,
        poll_interval_ms: int = 5000,
    ):
        super().__init__("tabdeal", symbol, api_token=api_token, poll_interval_ms=poll_interval_ms)

    async def fetch_balances(self) -> dict:
        return await asyncio.to_thread(private_rest.get_balances, self.api_token)

    async def fetch_open_orders(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_open_orders, self.symbol, self.api_token)

    async def fetch_fills(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_filled_orders, self.symbol, self.api_token)

    async def _publish_error(self, source: str, exc: Exception) -> None:
        if source == "fill_poll" and isinstance(exc, private_rest.TabdealTransientError):
            return
        await super()._publish_error(source, exc)


class TabdealWebSocketPrivateStateAdapter(TabdealRestPrivateStateAdapter):
    """Tabdeal listenKey websocket plus REST snapshots for balances/open-order baselines."""

    def __init__(self, symbol: str, *, api_token: private_rest.TabdealApiKeyAuth | None, poll_interval_ms: int = 5000):
        super().__init__(symbol, api_token=api_token, poll_interval_ms=poll_interval_ms)
        self._wire = build_instrument_wire("tabdeal", symbol)
        self._listen_key = ""
        self._ws_task: asyncio.Task | None = None
        self._order_update_queue: asyncio.Queue[OrderUpdate] = asyncio.Queue()
        self._fill_queue: asyncio.Queue[Fill] = asyncio.Queue()

    async def connect(self) -> None:
        await super().connect()
        auth = self.api_token if isinstance(self.api_token, private_rest.TabdealApiKeyAuth) else None
        self._listen_key = await asyncio.to_thread(private_rest.create_listen_key, auth.api_key if auth else None)
        if self._listen_key:
            self._ws_task = asyncio.create_task(self._run_user_stream())

    async def disconnect(self) -> None:
        await super().disconnect()
        if self._ws_task is not None:
            self._ws_task.cancel()
            await asyncio.gather(self._ws_task, return_exceptions=True)
            self._ws_task = None
        auth = self.api_token if isinstance(self.api_token, private_rest.TabdealApiKeyAuth) else None
        if self._listen_key and auth:
            await asyncio.to_thread(private_rest.close_listen_key, auth.api_key, self._listen_key)
        self._listen_key = ""

    async def order_update_stream(self) -> AsyncIterator[OrderUpdate]:
        while self._connected:
            yield await self._order_update_queue.get()

    async def fill_stream(self) -> AsyncIterator[Fill]:
        while self._connected:
            yield await self._fill_queue.get()

    async def _run_user_stream(self) -> None:
        auth = self.api_token if isinstance(self.api_token, private_rest.TabdealApiKeyAuth) else None
        keepalive_every_s = 30 * 60
        last_keepalive = now_ms()
        while self._connected and self._listen_key:
            try:
                async with websockets.connect(
                    f"wss://api1.tabdeal.org/stream/streams={self._listen_key}",
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    async for message in ws:
                        if auth and now_ms() - last_keepalive >= keepalive_every_s * 1000:
                            await asyncio.to_thread(private_rest.keepalive_listen_key, auth.api_key, self._listen_key)
                            last_keepalive = now_ms()
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            continue
                        await self._handle_user_message(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("user_stream", exc)
                await asyncio.sleep(2.0)

    async def _handle_user_message(self, payload: Any) -> None:
        if not isinstance(payload, dict) or payload.get("e") != "executionReport":
            return
        order_id = str(payload.get("i") or payload.get("c") or "")
        if not order_id:
            return
        side = "buy" if str(payload.get("S", "")).upper() == "BUY" else "sell"
        qty = _safe_float(payload.get("q"))
        filled = _safe_float(payload.get("z"))
        timestamp_ms = int(_safe_float(payload.get("E") or payload.get("O"), now_ms()))
        await self._order_update_queue.put(
            OrderUpdate(
                exchange="tabdeal",
                symbol=self.symbol,
                exchange_order_id=order_id,
                actor_id=str(payload.get("c") or "") or None,
                status=str(payload.get("X") or "").lower(),
                remaining_qty=max(0.0, qty - filled),
                filled_qty=filled,
                timestamp_ms=timestamp_ms,
            )
        )
        if str(payload.get("x") or "").upper() != "TRADE":
            return
        last_qty = _safe_float(payload.get("l"))
        if last_qty <= 0.0:
            return
        await self._fill_queue.put(
            Fill(
                exchange_order_id=order_id,
                actor_id=str(payload.get("c") or ""),
                exchange="tabdeal",
                symbol=self.symbol,
                side=side,
                price=_safe_float(payload.get("L") or payload.get("p")),
                qty=last_qty,
                fee=_safe_float(payload.get("n")),
                timestamp_ms=timestamp_ms,
            )
        )

    def normalize_open_order(self, row: dict, *, timestamp_ms: int) -> OpenOrder | None:
        o = super().normalize_open_order(row, timestamp_ms=timestamp_ms)
        if o is None:
            return None
        return OpenOrder(
            exchange_order_id=o.exchange_order_id,
            actor_id=o.actor_id,
            order=Order(
                exchange=o.order.exchange,
                symbol=o.order.symbol,
                side=o.order.side,
                price=o.order.price,
                qty=o.order.qty,
                order_type=o.order.order_type,
                client_order_id=o.order.client_order_id,
            ),
            remaining_qty=o.remaining_qty,
            filled_qty=o.filled_qty,
            status=o.status,
            timestamp_ms=o.timestamp_ms,
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float(default)
