from __future__ import annotations

from typing import Any

from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.models.execution import Order
from mm.exchanges.bitpin import private_rest
from mm.exchanges.shared.order_response import order_id_from_response


def _decimal_payload(value: float) -> str:
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text or "0"


class BitpinRestOrderCommandAdapter(AbstractOrderCommandAdapter):
    def __init__(
        self,
        symbol: str,
        *,
        api_token: Any,
        max_concurrency: int = 8,
    ):
        self.symbol = symbol
        self.api_token = api_token
        self._max_concurrency = max(1, int(max_concurrency))

    async def place_orders(self, orders: list[Order]) -> list[str]:
        responses: list[dict[str, Any]] = []
        for chunk in _chunks(orders, 10):
            response = self._place_bulk(chunk)
            batch_orders = response.get("orders")
            if not isinstance(batch_orders, list):
                message = response.get("message") or response.get("msg") or response
                raise RuntimeError(f"place bulk orders failed on bitpin: {message}")
            if len(batch_orders) != len(chunk):
                raise RuntimeError(
                    f"place bulk orders failed on bitpin: expected {len(chunk)} ids, got {len(batch_orders)}"
                )
            responses.extend(row for row in batch_orders if isinstance(row, dict))
        if len(responses) != len(orders):
            raise RuntimeError(f"place bulk orders failed on bitpin: malformed response")
        out_ids: list[str] = []
        for response in responses:
            oid = order_id_from_response(response)
            state = str(response.get("state", response.get("status", "")) or "").lower()
            if not oid or state in {"failed", "error", "rejected"}:
                message = response.get("message") or response.get("msg") or response
                raise RuntimeError(f"place order failed on bitpin: {message}")
            out_ids.append(oid)
        return out_ids

    async def cancel_orders(self, order_ids: list[str]) -> None:
        if not order_ids:
            return
        response = private_rest.cancel_bulk_orders(order_ids, self.api_token)
        state = str(response.get("state", response.get("status", "")) or "").lower()
        failed = response.get("not_canceled_orders")
        if state in {"failed", "error", "rejected"} or (isinstance(failed, list) and failed):
            message = response.get("message") or response.get("msg") or response
            raise RuntimeError(f"cancel bulk orders failed on bitpin: {message}")

    def _place_bulk(self, orders: list[Order]) -> dict[str, Any]:
        payloads = []
        for order in orders:
            sym = order.symbol or self.symbol
            payloads.append(
                {
                    "symbol": sym,
                    "base_amount": _decimal_payload(order.qty),
                    "price": _decimal_payload(order.price),
                    "side": order.side,
                    "order_type": str(order.order_type).lower().strip(),
                    "client_order_id": order.client_order_id,
                }
            )
        return private_rest.place_bulk_orders(payloads, self.api_token)


def _chunks(items: list[Order], size: int) -> list[list[Order]]:
    return [items[i : i + size] for i in range(0, len(items), max(1, int(size)))]
