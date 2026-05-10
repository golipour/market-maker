from __future__ import annotations

import asyncio
from typing import Any

from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.models.execution import Order


class RestOrderCommandAdapter(AbstractOrderCommandAdapter):
    """REST order commands via native venue clients (Nobitex / Wallex)."""

    def __init__(self, exchange: str, symbol: str, *, api_token: str | None):
        self.exchange = exchange.lower().strip()
        self.symbol = symbol
        self.api_token = api_token

    async def place_orders(self, orders: list[Order]) -> list[str]:
        order_ids: list[str] = []
        for order in orders:
            response = await asyncio.to_thread(self._place_one, order)
            order_id = _order_id_from_response(response)
            if self.exchange == "wallex":
                if not response.get("success", False) or not order_id:
                    message = response.get("message") or response.get("msg") or response
                    raise RuntimeError(f"place order failed on {self.exchange}: {message}")
            else:
                status = str(response.get("status", response.get("state", "")) or "").lower()
                if not order_id or status in {"failed", "error", "rejected"}:
                    message = response.get("message") or response.get("msg") or response
                    raise RuntimeError(f"place order failed on {self.exchange}: {message}")
            order_ids.append(order_id)
        return order_ids

    async def cancel_orders(self, order_ids: list[str]) -> None:
        for order_id in order_ids:
            response = await asyncio.to_thread(self._cancel_one, order_id)
            if self.exchange == "wallex":
                if not response.get("success", False):
                    message = response.get("message") or response.get("msg") or response
                    raise RuntimeError(f"cancel order failed on {self.exchange}: {message}")
            else:
                status = str(response.get("status", response.get("state", "")) or "").lower()
                if status in {"failed", "error", "rejected"}:
                    message = response.get("message") or response.get("msg") or response
                    raise RuntimeError(f"cancel order failed on {self.exchange}: {message}")

    def _place_one(self, order: Order) -> dict[str, Any]:
        sym = order.symbol or self.symbol
        if self.exchange == "nobitex":
            from mm.exchanges.nobitex import private_rest

            return private_rest.place_limit_order(
                sym,
                order.side,
                str(order.qty),
                str(order.price),
                self.api_token,
                client_order_id=order.client_order_id,
            )
        if self.exchange == "wallex":
            from mm.exchanges.wallex import private_rest

            return private_rest.place_limit_order(
                sym,
                order.side,
                str(order.qty),
                str(order.price),
                self.api_token,
                client_order_id=order.client_order_id,
            )
        return {"status": "failed", "message": f"unsupported exchange: {self.exchange}"}

    def _cancel_one(self, order_id: str) -> dict[str, Any]:
        if self.exchange == "nobitex":
            from mm.exchanges.nobitex import private_rest

            try:
                oid = int(order_id)
            except ValueError:
                return {"status": "failed", "message": "invalid order id"}
            return private_rest.cancel_order(oid, self.api_token)
        if self.exchange == "wallex":
            from mm.exchanges.wallex import private_rest

            return private_rest.cancel_order(order_id, self.api_token)
        return {"status": "failed", "message": f"unsupported exchange: {self.exchange}"}


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
        nested = _order_id_from_response(inner)
        if nested:
            return nested
    result = resp.get("result")
    if isinstance(result, dict):
        nested = _order_id_from_response(result)
        if nested:
            return nested
    return ""


TradeStatsRestOrderCommandAdapter = RestOrderCommandAdapter
