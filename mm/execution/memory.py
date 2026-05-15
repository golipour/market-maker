from __future__ import annotations

from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Action


class RecordingOrderManager(AbstractOrderManager):
    def __init__(self):
        self.synced: list[list[Action]] = []

    @property
    def live_orders(self) -> dict:
        return {}

    async def sync(self, actions: list[Action]) -> None:
        self.synced.append(actions)

