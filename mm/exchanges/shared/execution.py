from __future__ import annotations

import asyncio
from typing import Any

from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.models.execution import Order


class TradeStatsRestOrderCommandAdapter(AbstractOrderCommandAdapter):
    """REST order-command adapter backed by the existing trade_stats exchange code.

    It accepts float-domain prices and quantities. Exchange precision and payload
    quirks stay below this boundary in trade_stats venue adapters.
    """

    def __init__(self, exchange: str, symbol: str, *, api_token: str | None):
        self.exchange = exchange.lower().strip()
        self.symbol = symbol
        self.api_token = api_token

    async def place_orders(self, orders: list[Order]) -> list[str]:
        order_ids: list[str] = []
        for order in orders:
            response = await asyncio.to_thread(self._place_one, order)
            status = str(response.get("status", response.get("state", "")) or "").lower()
            order_id = _order_id_from_response(response)
            if not order_id or status in {"failed", "error", "rejected"}:
                message = response.get("message") or response.get("msg") or response
                raise RuntimeError(f"place order failed on {self.exchange}: {message}")
            order_ids.append(order_id)
        return order_ids

    async def cancel_orders(self, order_ids: list[str]) -> None:
        for order_id in order_ids:
            response = await asyncio.to_thread(self._cancel_one, order_id)
            status = str(response.get("status", response.get("state", "")) or "").lower()
            if status in {"failed", "error", "rejected"}:
                message = response.get("message") or response.get("msg") or response
                raise RuntimeError(f"cancel order failed on {self.exchange}: {message}")

    def _place_one(self, order: Order) -> dict[str, Any]:
        import trade_stats

        adapter = trade_stats._adapter(self.exchange)  # type: ignore[attr-defined]
        if not hasattr(adapter, "place_limit_order"):
            return {"status": "failed", "message": "place_limit_order not supported"}
        return adapter.place_limit_order(
            symbol=order.symbol or self.symbol,
            side=order.side,
            amount=str(order.qty),
            price=str(order.price),
            api_token=self.api_token,
            client_order_id=order.client_order_id,
        )

    def _cancel_one(self, order_id: str) -> dict[str, Any]:
        import trade_stats

        adapter = trade_stats._adapter(self.exchange)  # type: ignore[attr-defined]
        if not hasattr(adapter, "cancel_order"):
            return {"status": "failed", "message": "cancel_order not supported"}
        try:
            native_id: int | str = int(order_id)
        except ValueError:
            native_id = order_id
        return adapter.cancel_order(native_id, api_token=self.api_token)


class DryRunOrderCommandAdapter(AbstractOrderCommandAdapter):
    """Non-mutating command adapter for local tests and print_events demos."""

    def __init__(self, exchange: str):
        self.exchange = exchange.lower().strip()
        self._next_id = 1
        self.placed: list[Order] = []
        self.cancelled: list[str] = []

    async def place_orders(self, orders: list[Order]) -> list[str]:
        ids = []
        for order in orders:
            self.placed.append(order)
            ids.append(f"dry-{self.exchange}-{self._next_id}")
            self._next_id += 1
        return ids

    async def cancel_orders(self, order_ids: list[str]) -> None:
        self.cancelled.extend(order_ids)


def _order_id_from_response(resp: dict[str, Any]) -> str:
    for key in ("orderId", "order_id", "id", "clientOrderId", "client_order_id"):
        value = resp.get(key)
        if value is not None and str(value).strip():
            return str(value)
    inner = resp.get("order") if isinstance(resp.get("order"), dict) else resp.get("data")
    if isinstance(inner, dict):
        return _order_id_from_response(inner)
    return ""
