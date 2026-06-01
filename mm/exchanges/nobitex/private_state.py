from __future__ import annotations

import asyncio
from typing import Any

from mm.core.models.execution import Fill, Order
from mm.core.models.private import OpenOrder
from mm.exchanges.instrument_wire import build_instrument_wire
from mm.exchanges.nobitex import private_rest
from mm.exchanges.shared.private_state import RestPollingPrivateStateAdapter


class NobitexRestPrivateStateAdapter(RestPollingPrivateStateAdapter):
    def __init__(
        self,
        symbol: str,
        *,
        api_token: Any,
        poll_interval_ms: int = 5000,
    ):
        super().__init__(
            "nobitex",
            symbol,
            api_token=api_token,
            poll_interval_ms=poll_interval_ms,
        )
        self._wire = build_instrument_wire("nobitex", symbol)

    async def fetch_balances(self) -> dict:
        return await asyncio.to_thread(private_rest.get_balances, self.api_token)

    async def fetch_open_orders(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_open_orders, self.symbol, self.api_token)

    async def fetch_fills(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_filled_orders, self.symbol, self.api_token)

    def normalize_open_order(self, row: dict, *, timestamp_ms: int) -> OpenOrder | None:
        o = super().normalize_open_order(row, timestamp_ms=timestamp_ms)
        if o is None or self._wire.quote_asset != "IRT":
            return o
        internal_price = self._wire.scale_public_price_from_api(float(o.order.price))
        order = Order(
            exchange=o.order.exchange,
            symbol=o.order.symbol,
            side=o.order.side,
            price=internal_price,
            qty=float(o.order.qty),
            order_type=o.order.order_type,
            client_order_id=o.order.client_order_id,
        )
        return OpenOrder(
            exchange_order_id=o.exchange_order_id,
            actor_id=o.actor_id,
            order=order,
            remaining_qty=o.remaining_qty,
            filled_qty=o.filled_qty,
            status=o.status,
            timestamp_ms=o.timestamp_ms,
        )

    def normalize_fill(self, row: dict, *, timestamp_ms: int) -> Fill | None:
        fill = super().normalize_fill(row, timestamp_ms=timestamp_ms)
        if fill is None or self._wire.quote_asset != "IRT":
            return fill
        return Fill(
            exchange_order_id=fill.exchange_order_id,
            actor_id=fill.actor_id,
            exchange=fill.exchange,
            symbol=fill.symbol,
            side=fill.side,
            price=self._wire.scale_public_price_from_api(float(fill.price)),
            qty=fill.qty,
            fee=self._wire.scale_public_price_from_api(float(fill.fee)),
            timestamp_ms=fill.timestamp_ms,
        )
