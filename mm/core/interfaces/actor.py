from __future__ import annotations

from abc import ABC, abstractmethod

from mm.core.models.execution import Action, ActorState, Fill
from mm.core.models.market import MarketEvent


class AbstractStrategy(ABC):
    @abstractmethod
    async def on_market_event(self, event: MarketEvent, state: ActorState) -> list[Action]:
        ...

    @abstractmethod
    async def on_fill(self, fill: Fill, state: ActorState) -> None:
        ...


class AbstractRiskEngine(ABC):
    @abstractmethod
    def validate(self, actions: list[Action], state: ActorState, event: MarketEvent) -> list[Action]:
        ...


class AbstractOrderManager(ABC):
    @abstractmethod
    async def sync(self, actions: list[Action]) -> None:
        ...

