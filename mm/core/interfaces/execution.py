from __future__ import annotations

from abc import ABC, abstractmethod

from mm.core.models.execution import Order


class AbstractOrderCommandAdapter(ABC):
    """Private, authenticated write-side adapter for exchange order commands.

    Domain orders use float price/qty in base/quote units. Tick-size, lot-size,
    precision, min-notional checks, and exchange-native payloads belong inside
    implementations of this adapter or helpers they call.
    """

    @abstractmethod
    async def place_orders(self, orders: list[Order]) -> list[str]:
        """Place orders and return exchange order IDs in the same order."""
        ...

    @abstractmethod
    async def cancel_orders(self, order_ids: list[str]) -> None:
        ...

    async def amend_order(self, order_id: str, order: Order) -> str:
        await self.cancel_orders([order_id])
        placed = await self.place_orders([order])
        return placed[0]
