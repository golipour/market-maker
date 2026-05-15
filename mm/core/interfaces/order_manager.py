from __future__ import annotations

from abc import ABC, abstractmethod

from mm.core.models.execution import Action


class AbstractOrderManager(ABC):
    @abstractmethod
    async def sync(self, actions: list[Action]) -> None:
        ...
