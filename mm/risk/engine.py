from __future__ import annotations

import time
from dataclasses import dataclass

from mm.core.interfaces.risk import AbstractRiskEngine
from mm.core.models.execution import Action, TradingState
from mm.core.models.market import MarketEvent
from mm.observability import counter_inc, gauge


class PassThroughRiskEngine(AbstractRiskEngine):
    def validate(self, actions: list[Action], state: TradingState, event: MarketEvent) -> list[Action]:
        return actions


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
        if self.kill_switch:
            counter_inc("mm_risk_reject_total", labels={"reason": "kill_switch"})
            return []
        if self.clock_ms() - event.t_end_ms > self.limits.max_data_age_ms:
            counter_inc("mm_risk_reject_total", labels={"reason": "stale_data"})
            return []
        filtered: list[Action] = []
        for action in actions:
            safe_orders = []
            for order in action.orders_to_create:
                signed_qty = order.qty if order.side.lower().strip() == "buy" else -order.qty
                if abs(state.inventory + signed_qty) <= self.limits.max_inventory:
                    safe_orders.append(order)
                else:
                    counter_inc("mm_risk_reject_total", labels={"reason": "max_inventory"})
            if safe_orders or action.orders_to_cancel:
                filtered.append(
                    Action(
                        actor_id=action.actor_id,
                        orders_to_create=tuple(safe_orders),
                        orders_to_cancel=action.orders_to_cancel,
                    )
                )
        return filtered

