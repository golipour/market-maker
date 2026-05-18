from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from mm.core.interfaces.risk import AbstractRiskEngine
from mm.core.models.execution import Action, Order, TradingState
from mm.core.models.market import MarketEvent
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.execution.plan import RiskResult
from mm.observability import counter_inc, gauge
from mm.core.risk_limits import HedgeVenueRiskLimits, VenueRiskLimits


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
            exposure_before=_exposure_dict(state),
            exposure_after=_exposure_dict(state),
        )

    def validate_frame_with_result(
        self,
        actions: list[Action],
        state: TradingState,
        frame: StrategyInputFrame,
    ) -> RiskResult:
        return self.validate_with_result(actions, state, frame.primary)


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Resolved hard limits for ``RuntimeRiskEngine``."""

    primary: VenueRiskLimits = field(default_factory=VenueRiskLimits)
    hedge: HedgeVenueRiskLimits = field(default_factory=HedgeVenueRiskLimits)

    @property
    def max_inventory(self) -> float:
        return self.primary.max_inventory

    @property
    def max_data_age_ms(self) -> int:
        return self.primary.max_market_data_age_ms


class RuntimeRiskEngine(AbstractRiskEngine):
    """Deterministic hard risk gate for the async runtime."""

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
        rejection_counts: dict[str, int] = {}
        exposure_before = _exposure_dict(state)
        now_ms = int(self.clock_ms())

        if self.kill_switch:
            counter_inc("mm_risk_reject_total", labels={"reason": "kill_switch"})
            rejection_counts["kill_switch"] = _proposed_order_count(actions) or len(actions)
            return RiskResult(
                approved=(),
                rejection_counts=rejection_counts,
                exposure_before=exposure_before,
                exposure_after=exposure_before,
            )

        gate_reason = self._frame_gate_reason(frame, now_ms=now_ms)
        if gate_reason is not None:
            counter_inc("mm_risk_reject_total", labels={"reason": gate_reason})
            rejection_counts[gate_reason] = _proposed_order_count(actions) or len(actions)
            return RiskResult(
                approved=(),
                rejection_counts=rejection_counts,
                exposure_before=exposure_before,
                exposure_after=exposure_before,
            )

        filtered, order_rejections, exposure_after = self._filter_actions(
            actions,
            state,
            exchange=frame.primary.exchange,
            symbol=frame.primary.symbol,
            limits=self.limits.primary,
        )
        rejection_counts.update(order_rejections)
        if len(filtered) < len(actions):
            rejection_counts["actions_filtered"] = len(actions) - len(filtered)
        return RiskResult(
            approved=tuple(filtered),
            rejection_counts=rejection_counts,
            exposure_before=exposure_before,
            exposure_after=exposure_after,
        )

    def validate_with_result(
        self, actions: list[Action], state: TradingState, event: MarketEvent
    ) -> RiskResult:
        frame = StrategyInputFrame(primary=event)
        return self.validate_frame_with_result(actions, state, frame)

    def _frame_gate_reason(self, frame: StrategyInputFrame, *, now_ms: int) -> str | None:
        primary = self.limits.primary
        market_age_ms = now_ms - int(frame.primary.t_end_ms)
        if market_age_ms > primary.max_market_data_age_ms:
            return "stale_market_data"

        priv_reason = _private_freshness_reason(
            frame.primary_private,
            now_ms=now_ms,
            max_age_ms=primary.max_private_state_age_ms,
            missing_reason="missing_primary_private",
            stale_reason="stale_primary_private",
        )
        if priv_reason is not None:
            return priv_reason

        if frame.has_hedge:
            hedge = self.limits.hedge
            if frame.hedge is not None:
                hedge_market_age = now_ms - int(frame.hedge.t_end_ms)
                if hedge_market_age > hedge.max_market_data_age_ms:
                    return "stale_hedge_market_data"
            hedge_priv_reason = _private_freshness_reason(
                frame.hedge_private,
                now_ms=now_ms,
                max_age_ms=hedge.max_private_state_age_ms,
                missing_reason="missing_hedge_private",
                stale_reason="stale_hedge_private",
            )
            if hedge_priv_reason is not None:
                return hedge_priv_reason
        return None

    def _filter_actions(
        self,
        actions: list[Action],
        state: TradingState,
        *,
        exchange: str,
        symbol: str,
        limits: VenueRiskLimits,
    ) -> tuple[list[Action], dict[str, int], dict[str, float]]:
        rejection_counts: dict[str, int] = {}
        inventory = float(state.inventory)
        cash = float(state.cash)
        open_notional = _open_notional(state, exchange=exchange, symbol=symbol)

        filtered: list[Action] = []
        for action in actions:
            safe_orders: list[Order] = []
            for order in action.orders_to_create:
                ok, reason = _check_order(
                    order,
                    inventory=inventory,
                    cash=cash,
                    open_notional=open_notional,
                    limits=limits,
                    exchange=exchange,
                    symbol=symbol,
                )
                if ok:
                    safe_orders.append(order)
                    inventory, cash, open_notional = _apply_order(
                        order,
                        inventory=inventory,
                        cash=cash,
                        open_notional=open_notional,
                    )
                elif reason is not None:
                    counter_inc("mm_risk_reject_total", labels={"reason": reason})
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
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

        exposure_after = {
            "inventory": inventory,
            "cash": cash,
            "open_notional": open_notional,
        }
        return filtered, rejection_counts, exposure_after


def _proposed_order_count(actions: list[Action]) -> int:
    return sum(len(a.orders_to_create) for a in actions)


def _exposure_dict(state: TradingState) -> dict[str, float]:
    return {
        "inventory": float(state.inventory),
        "cash": float(state.cash),
    }


def _private_age_ms(snapshot: PrivateStateSnapshot | None, *, now_ms: int) -> int | None:
    if snapshot is None:
        return None
    return max(0, now_ms - int(snapshot.timestamp_ms))


def _private_freshness_reason(
    snapshot: PrivateStateSnapshot | None,
    *,
    now_ms: int,
    max_age_ms: int,
    missing_reason: str,
    stale_reason: str,
) -> str | None:
    if not math.isfinite(max_age_ms) or max_age_ms == float("inf"):
        return None
    age = _private_age_ms(snapshot, now_ms=now_ms)
    if age is None:
        return missing_reason
    if age > max_age_ms:
        return stale_reason
    return None


def _open_notional(state: TradingState, *, exchange: str, symbol: str) -> float:
    ex = exchange.lower().strip()
    sym = symbol.strip()
    total = 0.0
    for order in state.open_orders.values():
        if order.exchange.lower().strip() == ex and order.symbol.strip() == sym:
            total += float(order.price) * float(order.qty)
    return total


def _check_order(
    order: Order,
    *,
    inventory: float,
    cash: float,
    open_notional: float,
    limits: VenueRiskLimits,
    exchange: str,
    symbol: str,
) -> tuple[bool, str | None]:
    if order.exchange.lower().strip() != exchange.lower().strip():
        return True, None
    if order.symbol.strip() != symbol.strip():
        return True, None

    qty = float(order.qty)
    if qty <= 0.0:
        return False, "invalid_qty"

    notional = float(order.price) * qty
    if notional > limits.max_order_notional:
        return False, "max_order_notional"
    if open_notional + notional > limits.max_open_notional:
        return False, "max_open_notional"

    side = order.side.lower().strip()
    if side == "buy":
        if cash - notional < limits.min_cash:
            return False, "min_cash"
        if inventory + qty > limits.max_inventory:
            return False, "max_inventory"
    elif side == "sell":
        if qty > inventory:
            return False, "insufficient_inventory"
        if inventory - qty < limits.min_inventory:
            return False, "min_inventory"
    else:
        return False, "invalid_side"
    return True, None


def _apply_order(
    order: Order,
    *,
    inventory: float,
    cash: float,
    open_notional: float,
) -> tuple[float, float, float]:
    notional = float(order.price) * float(order.qty)
    side = order.side.lower().strip()
    if side == "buy":
        cash -= notional
        inventory += float(order.qty)
    elif side == "sell":
        cash += notional
        inventory -= float(order.qty)
    open_notional += notional
    return inventory, cash, open_notional
