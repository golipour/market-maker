from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets

from mm.core.models.execution import Fill, Order
from mm.core.models.private import OpenOrder, OrderUpdate
from mm.exchanges.bitpin import private_rest
from mm.exchanges.shared.private_state import RestPollingPrivateStateAdapter
from mm.exchanges.shared.provider_adapter import parse_timestamp_ms
from mm.exchanges.bitpin.ws import _centrifuge_payloads, _publication_data


class BitpinRestPrivateStateAdapter(RestPollingPrivateStateAdapter):
    def __init__(
        self,
        symbol: str,
        *,
        api_token: Any,
        poll_interval_ms: int = 5000,
    ):
        super().__init__("bitpin", symbol, api_token=api_token, poll_interval_ms=poll_interval_ms)

    async def fetch_balances(self) -> dict:
        return await asyncio.to_thread(private_rest.get_balances, self.api_token)

    async def fetch_open_orders(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_open_orders, self.symbol, self.api_token)

    async def fetch_fills(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_filled_orders, self.symbol, self.api_token)

    def normalize_open_order(self, row: dict, *, timestamp_ms: int) -> OpenOrder | None:
        normalized = _normalize_bitpin_order_row(row)
        return super().normalize_open_order(normalized, timestamp_ms=timestamp_ms)

    def normalize_fill(self, row: dict, *, timestamp_ms: int) -> Fill | None:
        normalized = _normalize_bitpin_order_row(row)
        return super().normalize_fill(normalized, timestamp_ms=timestamp_ms)


class BitpinWebSocketPrivateStateAdapter(BitpinRestPrivateStateAdapter):
    WS_URL = "wss://centrifugo.bitpin.ir/connection/websocket"

    def __init__(
        self,
        symbol: str,
        *,
        api_token: Any,
        poll_interval_ms: int = 5000,
    ):
        super().__init__(symbol, api_token=api_token, poll_interval_ms=poll_interval_ms)
        self._ws_task: asyncio.Task | None = None
        self._order_update_queue: asyncio.Queue[OrderUpdate] = asyncio.Queue()
        self._fill_queue: asyncio.Queue[Fill] = asyncio.Queue()

    async def connect(self) -> None:
        await super().connect()
        ws_info = await asyncio.to_thread(private_rest.get_websocket_info, self.api_token)
        if ws_info is not None:
            self._ws_task = asyncio.create_task(self._run_user_stream(ws_info))

    async def disconnect(self) -> None:
        await super().disconnect()
        if self._ws_task is not None:
            self._ws_task.cancel()
            await asyncio.gather(self._ws_task, return_exceptions=True)
            self._ws_task = None

    async def order_update_stream(self):
        while self._connected:
            yield await self._order_update_queue.get()

    async def fill_stream(self):
        while self._connected:
            yield await self._fill_queue.get()

    async def _run_user_stream(self, ws_info: private_rest.BitpinWebSocketInfo) -> None:
        reconnect_s = 1.0
        while self._connected:
            try:
                async with websockets.connect(self.WS_URL, ping_interval=20, ping_timeout=10) as ws:
                    reconnect_s = 1.0
                    await ws.send(json.dumps({"id": 1, "connect": {"token": ws_info.ws_token}}))
                    await ws.send(
                        json.dumps(
                            {
                                "id": 2,
                                "subscribe": {"channel": f"user:order_info#{ws_info.user_identifier}"},
                            }
                        )
                    )
                    async for raw in ws:
                        for payload in _centrifuge_payloads(raw):
                            await self._handle_user_message(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("user_stream", exc)
                if self._connected:
                    await asyncio.sleep(reconnect_s)
                    reconnect_s = min(30.0, reconnect_s * 2.0)

    async def _handle_user_message(self, payload: Any) -> None:
        data = _publication_data(payload)
        if not isinstance(data, dict):
            return
        event = str(data.get("event") or "")
        if event == "user_order_update":
            update = self._order_update_from_row(data)
            if update is not None:
                await self._order_update_queue.put(update)
        elif event == "user_match_update":
            fill = self._fill_from_row(data)
            if fill is not None:
                await self._fill_queue.put(fill)

    def _order_update_from_row(self, row: dict[str, Any]) -> OrderUpdate | None:
        order_id = str(row.get("id") or "").strip()
        if not order_id:
            return None
        qty = _safe_float(row.get("base_amount"))
        filled = _safe_float(row.get("dealed_base_amount"))
        return OrderUpdate(
            exchange="bitpin",
            symbol=self.symbol,
            exchange_order_id=order_id,
            actor_id=str(row.get("identifier") or "") or None,
            status=str(row.get("state") or "").lower(),
            remaining_qty=max(0.0, qty - filled),
            filled_qty=filled,
            timestamp_ms=parse_timestamp_ms(row.get("event_time") or row.get("created_at") or row.get("closed_at")),
        )

    def _fill_from_row(self, row: dict[str, Any]) -> Fill | None:
        order_id = str(row.get("order_id") or "").strip()
        if not order_id:
            return None
        qty = _safe_float(row.get("base_amount"))
        if qty <= 0.0:
            return None
        return Fill(
            exchange_order_id=order_id,
            actor_id=str(row.get("identifier") or ""),
            exchange="bitpin",
            symbol=self.symbol,
            side="buy" if str(row.get("side")).lower() == "buy" else "sell",
            price=_safe_float(row.get("price")),
            qty=qty,
            fee=_safe_float(row.get("commission")),
            timestamp_ms=parse_timestamp_ms(row.get("event_time") or row.get("created_at")),
        )


def _normalize_bitpin_order_row(row: dict) -> dict:
    if not isinstance(row, dict):
        return row
    out = dict(row)
    if "orderId" not in out and out.get("id") is not None:
        out["orderId"] = out.get("id")
    if "side" not in out and out.get("type") is not None:
        out["side"] = out.get("type")
    if "quantity" not in out:
        out["quantity"] = out.get("amount1") or out.get("amount") or out.get("resource")
    if "executedQty" not in out:
        out["executedQty"] = out.get("exchanged1") or out.get("matchedAmount")
    if "remaining" not in out:
        out["remaining"] = out.get("remain_amount")
    if "price" not in out:
        out["price"] = out.get("price_limit") or out.get("active_limit") or out.get("average_price")
    if "timestamp" not in out:
        out["timestamp"] = out.get("created_at") or out.get("activated_at") or out.get("closed_at")
    if "fee" not in out:
        out["fee"] = out.get("commission") or out.get("user_commission")
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float(default)
