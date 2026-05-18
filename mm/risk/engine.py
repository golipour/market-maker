from __future__ import annotations

import time
from dataclasses import dataclass, field

from mm.core.interfaces.risk import AbstractRiskEngine
from mm.core.models.execution import Action, TradingState
from mm.core.models.market import MarketEvent
from mm.core.models.strategy import StrategyInputFrame
from mm.execution.plan import RiskResult
from mm.observability import counter_inc, gauge


class PassThroughRiskEngine(AbstractRiskEngine):
    def validate(self, actions: list[Action], state: TradingState, event: MarketEvent) -> list[Action]:
        return self.validate_with_result(actions, state, event).approved_list

    def validate_with_result(
        self, actions: list[Action], state: TradingState, event: MarketEvent
    ) -> RiskResult:
        approved = list(actions)
        return RiskResult(
            approved=tuple(approved),
            rejection_counts={},
        )


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_inventory: float = float("inf")
    max_data_age_ms: int = 10_000
    live_execution_enabled: bool = False


class RuntimeRiskEngine(AbstractRiskEngine):
    """Deterministic risk gate for the async runtime."""

    def __init__(self, limits: RiskLimits | None = None, *, clock_ms=None):
        self.limits = limits or RiskLimits()
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.kill_switch = False

    def set_kill_switch(self, enabled: bool) -> None:
        self.kill_switch = bool(enabled)
        gauge("mm_kill_switch_state", 1.0 if self.kill_switch else 0.0)

    def validate(self, actions: list[Action], state: TradingState, event: MarketEvent) -> list[Action]:
        return self.validate_with_result(actions, state, event).approved_list

    def validate_frame(self, actions: list[Action], state: TradingState, frame: StrategyInputFrame) -> list[Action]:
        return self.validate_frame_with_result(actions, state, frame).approved_list

    def validate_frame_with_result(
        self,
        actions: list[Action],
        state: TradingState,
        frame: StrategyInputFrame,
    ) -> RiskResult:
        return self.validate_with_result(actions, state, frame.primary)

    def validate_with_result(
        self, actions: list[Action], state: TradingState, event: MarketEvent
    ) -> RiskResult:
        rejection_counts: dict[str, int] = {}
        if self.kill_switch:
            counter_inc("mm_risk_reject_total", labels={"reason": "kill_switch"})
            rejection_counts["kill_switch"] = sum(len(a.orders_to_create) for a in actions) or len(actions)
            return RiskResult(approved=(), rejection_counts=rejection_counts)
        if self.clock_ms() - event.t_end_ms > self.limits.max_data_age_ms:
            counter_inc("mm_risk_reject_total", labels={"reason": "stale_data"})
            rejection_counts["stale_data"] = sum(len(a.orders_to_create) for a in actions) or len(actions)
            return RiskResult(approved=(), rejection_counts=rejection_counts)

        filtered: list[Action] = []
        for action in actions:
            safe_orders = []
            for order in action.orders_to_create:
                signed_qty = order.qty if order.side.lower().strip() == "buy" else -order.qty
                if abs(state.inventory + signed_qty) <= self.limits.max_inventory:
                    safe_orders.append(order)
                else:
                    counter_inc("mm_risk_reject_total", labels={"reason": "max_inventory"})
                    rejection_counts["max_inventory"] = rejection_counts.get("max_inventory", 0) + 1
            if safe_orders or action.orders_to_cancel:
                filtered.append(
                    Action(
                        actor_id=action.actor_id,
                        orders_to_create=tuple(safe_orders),
                        orders_to_cancel=action.orders_to_cancel,
                    )
                )
            elif action.orders_to_create:
                rejection_counts["action_empty_after_filter"] = (
                    rejection_counts.get("action_empty_after_filter", 0) + 1
                )
        if len(filtered) < len(actions):
            rejection_counts["actions_filtered"] = len(actions) - len(filtered)
        return RiskResult(approved=tuple(filtered), rejection_counts=rejection_counts)
