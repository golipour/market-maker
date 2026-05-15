from __future__ import annotations

from abc import ABC, abstractmethod

from mm.core.models.execution import Action, TradingState
from mm.core.models.market import MarketEvent
from mm.core.models.strategy import StrategyInputFrame


class AbstractRiskEngine(ABC):
    @abstractmethod
    def validate(self, actions: list[Action], state: TradingState, event: MarketEvent) -> list[Action]:
        ...

    def validate_frame(
        self,
        actions: list[Action],
        state: TradingState,
        frame: StrategyInputFrame,
    ) -> list[Action]:
        return self.validate(actions, state, frame.primary)
