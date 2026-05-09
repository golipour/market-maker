from __future__ import annotations

from abc import ABC, abstractmethod

from mm.core.models.execution import Action, ActorState, Fill
from mm.core.models.market import MarketEvent
from mm.core.models.strategy import StrategyInputFrame


class AbstractStrategy(ABC):
    @abstractmethod
    async def on_market_event(self, event: MarketEvent, state: ActorState) -> list[Action]:
        ...

    async def on_strategy_frame(self, frame: StrategyInputFrame, state: ActorState) -> list[Action]:
        """Handle an aligned primary-plus-optional-hedge frame.

        Existing single-venue strategies only need ``on_market_event``. Hedged
        strategies override this method and decide how to behave when
        ``frame.hedge`` is None.
        """
        return await self.on_market_event(frame.primary, state)

    @abstractmethod
    async def on_fill(self, fill: Fill, state: ActorState) -> None:
        ...


class AbstractRiskEngine(ABC):
    @abstractmethod
    def validate(self, actions: list[Action], state: ActorState, event: MarketEvent) -> list[Action]:
        ...

    def validate_frame(self, actions: list[Action], state: ActorState, frame: StrategyInputFrame) -> list[Action]:
        return self.validate(actions, state, frame.primary)


class AbstractOrderManager(ABC):
    @abstractmethod
    async def sync(self, actions: list[Action]) -> None:
        ...

