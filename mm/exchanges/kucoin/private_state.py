from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import websockets

from mm.core.models.execution import Fill, Order
from mm.core.models.private import OpenOrder, OrderUpdate
from mm.exchanges.instrument_wire import build_instrument_wire
from mm.exchanges.kucoin import private_rest
from mm.exchanges.shared.private_state import RestPollingPrivateStateAdapter
from mm.exchanges.shared.provider_adapter import now_ms


class KucoinRestPrivateStateAdapter(RestPollingPrivateStateAdapter):
    def __init__(
        self,
        symbol: str,
        *,
        api_token: Any,
        poll_interval_ms: int = 5000,
    ):
        super().__init__("kucoin", symbol, api_token=api_token, poll_interval_ms=poll_interval_ms)
        self._seen_trade_ids: set[str] = set()

    async def fetch_balances(self) -> dict:
        return await asyncio.to_thread(private_rest.get_balances, self.api_token)

    async def fetch_open_orders(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_open_orders, self.symbol, self.api_token)

    async def fetch_fills(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_fills, self.symbol, self.api_token)

    async def _publish_error(self, source: str, exc: Exception) -> None:
        if source == "fill_poll" and isinstance(exc, private_rest.KucoinTransientError):
            return
        await super()._publish_error(source, exc)

    def normalize_open_order(self, row: dict, *, timestamp_ms: int) -> OpenOrder | None:
        if not isinstance(row, dict):
            return None
        order_id = str(row.get("id") or row.get("orderId") or row.get("clientOid") or "").strip()
        if not order_id:
            return None
        side = str(row.get("side", "")).lower()
        if side not in {"buy", "sell"}:
            return None
        price = _safe_float(row.get("price"))
        qty = _safe_float(row.get("size", row.get("originSize")))
        filled = _safe_float(row.get("dealSize", row.get("filledSize")))
        remaining = _safe_float(row.get("remainSize"))
        if remaining <= 0.0:
            remaining = max(0.0, qty - filled)
        if price <= 0.0 or remaining <= 0.0:
            return None
        status = str(row.get("status") or ("active" if row.get("active", True) else "done")).lower()
        client_order_id = str(row.get("clientOid") or "").strip() or None
        return OpenOrder(
            exchange_order_id=order_id,
            actor_id=client_order_id,
            order=Order(
                exchange=self.exchange,
                symbol=self.symbol,
                side=side,
                price=price,
                qty=remaining,
                order_type=str(row.get("type", "limit") or "limit"),
                client_order_id=client_order_id,
            ),
            remaining_qty=remaining,
            filled_qty=max(0.0, filled),
            status=status,
            timestamp_ms=_safe_int(row.get("createdAt"), timestamp_ms),
        )

    def normalize_fill(self, row: dict, *, timestamp_ms: int) -> Fill | None:
        if not isinstance(row, dict):
            return None
        trade_id = str(row.get("tradeId") or row.get("id") or "").strip()
        if trade_id and trade_id in self._seen_trade_ids:
            return None
        order_id = str(row.get("orderId") or row.get("order_id") or "").strip()
        if not order_id:
            return None
        side = str(row.get("side", "")).lower()
        if side not in {"buy", "sell"}:
            return None
        qty = _safe_float(row.get("size"))
        if qty <= 0.0:
            return None
        if trade_id:
            self._seen_trade_ids.add(trade_id)
        return Fill(
            exchange_order_id=order_id,
            actor_id=str(row.get("clientOid") or ""),
            exchange=self.exchange,
            symbol=self.symbol,
            side=side,
            price=_safe_float(row.get("price")),
            qty=qty,
            fee=_safe_float(row.get("fee"), 0.0),
            timestamp_ms=_safe_int(row.get("createdAt") or row.get("time"), timestamp_ms),
        )


class KucoinWebSocketPrivateStateAdapter(KucoinRestPrivateStateAdapter):
    """KuCoin bullet-private websocket plus REST snapshots."""

    def __init__(
        self,
        symbol: str,
        *,
        api_token: private_rest.KucoinApiKeyAuth | None,
        poll_interval_ms: int = 5000,
    ):
        super().__init__(symbol, api_token=api_token, poll_interval_ms=poll_interval_ms)
        self._wire = build_instrument_wire("kucoin", symbol)
        self._ws_task: asyncio.Task | None = None
        self._order_update_queue: asyncio.Queue[OrderUpdate] = asyncio.Queue()
        self._fill_queue: asyncio.Queue[Fill] = asyncio.Queue()

    async def connect(self) -> None:
        await super().connect()
        if isinstance(self.api_token, private_rest.KucoinApiKeyAuth):
            self._ws_task = asyncio.create_task(self._run_user_stream())

    async def disconnect(self) -> None:
        await super().disconnect()
        if self._ws_task is not None:
            self._ws_task.cancel()
            await asyncio.gather(self._ws_task, return_exceptions=True)
            self._ws_task = None

    async def order_update_stream(self) -> AsyncIterator[OrderUpdate]:
        while self._connected:
            yield await self._order_update_queue.get()

    async def fill_stream(self) -> AsyncIterator[Fill]:
        while self._connected:
            yield await self._fill_queue.get()

    async def _run_user_stream(self) -> None:
        while self._connected:
            url = await asyncio.to_thread(private_rest.bullet_private_websocket_url, self.api_token)
            if not url:
                await asyncio.sleep(2.0)
                continue
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    await ws.send(json.dumps({
                        "id": str(now_ms()),
                        "type": "subscribe",
                        "topic": "/spotMarket/tradeOrders",
                        "privateChannel": True,
                        "response": True,
                    }))
                    async for message in ws:
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            continue
                        if payload.get("type") == "ping":
                            await ws.send(json.dumps({"type": "pong", "id": payload.get("id")}))
                            continue
                        await self._handle_user_message(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("user_stream", exc)
                await asyncio.sleep(2.0)

    async def _handle_user_message(self, payload: Any) -> None:
        if not isinstance(payload, dict) or payload.get("type") != "message":
            return
        data = payload.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return
        if not isinstance(data, dict):
            return
        order_id = str(data.get("orderId") or data.get("id") or "").strip()
        if not order_id:
            return
        side = str(data.get("side", "")).lower()
        if side not in {"buy", "sell"}:
            return
        qty = _safe_float(data.get("size", data.get("originSize")))
        filled = _safe_float(data.get("filledSize", data.get("dealSize")))
        remaining = _safe_float(data.get("remainSize"))
        if remaining <= 0.0:
            remaining = max(0.0, qty - filled)
        timestamp_ms = _safe_int(data.get("ts") or data.get("orderTime"), now_ms())
        status = str(data.get("status") or data.get("type") or "open").lower()
        await self._order_update_queue.put(
            OrderUpdate(
                exchange="kucoin",
                symbol=self.symbol,
                exchange_order_id=order_id,
                actor_id=str(data.get("clientOid") or "") or None,
                status=status,
                remaining_qty=max(0.0, remaining),
                filled_qty=max(0.0, filled),
                timestamp_ms=timestamp_ms,
            )
        )
        match_size = _safe_float(data.get("matchSize"))
        if match_size <= 0.0 and status not in {"match", "filled"}:
            return
        if match_size <= 0.0:
            match_size = max(0.0, filled)
        if match_size <= 0.0:
            return
        trade_id = str(data.get("tradeId") or f"{order_id}:{timestamp_ms}:{match_size}")
        if trade_id in self._seen_trade_ids:
            return
        self._seen_trade_ids.add(trade_id)
        await self._fill_queue.put(
            Fill(
                exchange_order_id=order_id,
                actor_id=str(data.get("clientOid") or ""),
                exchange="kucoin",
                symbol=self.symbol,
                side=side,
                price=_safe_float(data.get("matchPrice") or data.get("price")),
                qty=match_size,
                fee=_safe_float(data.get("fee"), 0.0),
                timestamp_ms=timestamp_ms,
            )
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return int(default)
