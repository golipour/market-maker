"""Nobitex-only REST order command adapter."""

from __future__ import annotations

from typing import Any

from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.models.execution import Order
from mm.exchanges.nobitex import private_rest
from mm.exchanges.shared.execution import map_in_thread_pool
from mm.exchanges.shared.order_response import order_id_from_response


class NobitexRestOrderCommandAdapter(AbstractOrderCommandAdapter):
    """Limit place/cancel via ``mm.exchanges.nobitex.private_rest``."""

    def __init__(
        self,
        symbol: str,
        *,
        api_token: str | None,
        max_concurrency: int = 8,
    ):
        self.symbol = symbol
        self.api_token = api_token
        self._max_concurrency = max(1, int(max_concurrency))

    async def place_orders(self, orders: list[Order]) -> list[str]:
        responses = await map_in_thread_pool(
            self._place_one,
            orders,
            max_concurrency=self._max_concurrency,
        )
        out_ids: list[str] = []
        for response in responses:
            oid = order_id_from_response(response)
            status = str(response.get("status", response.get("state", "")) or "").lower()
            if not oid or status in {"failed", "error", "rejected"}:
                message = response.get("message") or response.get("msg") or response
                raise RuntimeError(f"place order failed on nobitex: {message}")
            out_ids.append(oid)
        return out_ids

    async def cancel_orders(self, order_ids: list[str]) -> None:
        responses = await map_in_thread_pool(
            self._cancel_one,
            order_ids,
            max_concurrency=self._max_concurrency,
        )
        for response in responses:
            status = str(response.get("status", response.get("state", "")) or "").lower()
            if status in {"failed", "error", "rejected"}:
                message = response.get("message") or response.get("msg") or response
                raise RuntimeError(f"cancel order failed on nobitex: {message}")

    def _place_one(self, order: Order) -> dict[str, Any]:
        sym = order.symbol or self.symbol
        if str(order.order_type).lower().strip() == "market":
            return private_rest.place_market_order(
                sym,
                order.side,
                str(order.qty),
                self.api_token,
                client_order_id=order.client_order_id,
            )
        return private_rest.place_limit_order(
            sym,
            order.side,
            str(order.qty),
            str(order.price),
            self.api_token,
            client_order_id=order.client_order_id,
        )

    def _cancel_one(self, order_id: str) -> dict[str, Any]:
        try:
            oid = int(order_id)
        except ValueError:
            return {"status": "failed", "message": "invalid order id"}
        return private_rest.cancel_order(oid, self.api_token)
