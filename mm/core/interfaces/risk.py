from __future__ import annotations

from abc import ABC, abstractmethod

from mm.core.models.execution import Action, TradingState
from mm.core.models.market import MarketEvent
from mm.core.models.strategy import StrategyInputFrame
from mm.execution.plan import RiskResult


class AbstractRiskEngine(ABC):
    @abstractmethod
    def validate(self, actions: list[Action], state: TradingState, event: MarketEvent) -> list[Action]:
        ...

    def validate_with_result(
        self, actions: list[Action], state: TradingState, event: MarketEvent
    ) -> RiskResult:
        approved = self.validate(actions, state, event)
        proposed_orders = sum(len(a.orders_to_create) for a in actions)
        approved_orders = sum(len(a.orders_to_create) for a in approved)
        rejection_counts: dict[str, int] = {}
        if len(approved) < len(actions):
            rejection_counts["actions_filtered"] = len(actions) - len(approved)
        if approved_orders < proposed_orders:
            rejection_counts["orders_filtered"] = proposed_orders - approved_orders
        return RiskResult(approved=tuple(approved), rejection_counts=rejection_counts)

    def validate_frame(
        self,
        actions: list[Action],
        state: TradingState,
        frame: StrategyInputFrame,
    ) -> list[Action]:
        return self.validate_frame_with_result(actions, state, frame).approved_list

    def validate_frame_with_result(
        self,
        actions: list[Action],
        state: TradingState,
        frame: StrategyInputFrame,
    ) -> RiskResult:
        return self.validate_with_result(actions, state, frame.primary)
