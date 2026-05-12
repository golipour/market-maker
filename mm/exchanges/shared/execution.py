from __future__ import annotations

import asyncio

from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.models.execution import Order


class DryRunOrderCommandAdapter(AbstractOrderCommandAdapter):
    """Non-mutating command adapter for local tests and ``mm.tools.print_events`` demos."""

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
