from __future__ import annotations

from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.adapters.joint import venue_target_book_to_action
from mm.core.models.execution import Action, Order
from mm.core.models.target_book import VenueTargetBook
from mm.execution.plan import ExecutionSyncReport, summarize_orders


class RecordingOrderManager(AbstractOrderManager):
    def __init__(self, exchange: str = "test") -> None:
        self.exchange = exchange.lower().strip()
        self.synced: list[list[Action]] = []

    @property
    def live_orders(self) -> dict[str, Order]:
        return {}

    async def sync(self, actions: list[Action]) -> None:
        self.synced.append(actions)

    async def sync_with_report(self, actions: list[Action]) -> ExecutionSyncReport:
        self.synced.append(actions)
        orders = tuple(order for action in actions for order in action.orders)
        summary = summarize_orders(orders)
        return ExecutionSyncReport(
            target=summary,
            kept=summary,
            limit_valid=summary,
            accepted=summary,
            placed=True,
        )

    async def sync_venue_target_book(self, target: VenueTargetBook) -> ExecutionSyncReport:
        action = venue_target_book_to_action(target)
        return await self.sync_with_report([action])
