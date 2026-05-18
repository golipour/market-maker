"""Accumulate per-frame facts for ``ControllerDiagnostics``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from mm.core.models.execution import Action, TradingState
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.diagnostics.models import (
    ActionPipelineDiagnostics,
    ControllerDiagnostics,
    ExecutionGateDiagnostics,
    FrameDiagnostics,
    ModelScore,
    OptimizerDiagnostics,
    RiskDiagnostics,
    StateDiagnostics,
)
from mm.runtime.config import ControllerRuntimeConfig, RiskRuntimeConfig


def _private_age_ms(snapshot: PrivateStateSnapshot | None, *, now_ms: int) -> int | None:
    if snapshot is None:
        return None
    return max(0, now_ms - int(snapshot.timestamp_ms))


def _order_stage_summary(orders: tuple[Any, ...]) -> dict[str, Any]:
    buy_qty = sell_qty = 0.0
    buy_notional = sell_notional = 0.0
    for order in orders:
        qty = float(getattr(order, "qty", 0.0))
        price = float(getattr(order, "price", 0.0))
        side = str(getattr(order, "side", "")).lower().strip()
        if side == "buy":
            buy_qty += qty
            buy_notional += price * qty
        elif side == "sell":
            sell_qty += qty
            sell_notional += price * qty
    return {
        "order_count": len(orders),
        "buy_qty": buy_qty,
        "sell_qty": sell_qty,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
    }


@dataclass(slots=True)
class DecisionContext:
    """Facts gathered across one ``StrategyRunner.on_frame`` invocation."""

    config: ControllerRuntimeConfig
    frame: StrategyInputFrame
    clean_frame: StrategyInputFrame
    trading_state: TradingState
    frame_step: int
    run_id: str = ""
    now_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    latency_ms: float = 0.0

    decision: StrategyDecision | None = None
    controller_error: str | None = None
    proposed_actions: list[Action] = field(default_factory=list)
    approved_actions: list[Action] = field(default_factory=list)
    executed: bool = False
    blocked_reasons: list[str] = field(default_factory=list)
    risk_rejection_counts: dict[str, int] = field(default_factory=dict)

    def record_controller_error(self, exc: BaseException) -> None:
        self.controller_error = f"{type(exc).__name__}: {exc}"
        self.decision = None

    def record_decision(self, decision: StrategyDecision) -> None:
        self.decision = decision

    def record_risk_result(self, proposed: list[Action], approved: list[Action]) -> None:
        self.proposed_actions = list(proposed)
        self.approved_actions = list(approved)
        if len(approved) < len(proposed):
            self.risk_rejection_counts["actions_filtered"] = len(proposed) - len(approved)
            proposed_orders = sum(len(a.orders_to_create) for a in proposed)
            approved_orders = sum(len(a.orders_to_create) for a in approved)
            if approved_orders < proposed_orders:
                self.risk_rejection_counts["orders_filtered"] = proposed_orders - approved_orders

    def record_execution_gates(self, *, executed: bool, blocked_reasons: list[str]) -> None:
        self.executed = executed
        self.blocked_reasons = list(blocked_reasons)

    def to_diagnostics(self) -> ControllerDiagnostics:
        decision = self.decision
        warnings: list[str] = []
        if self.controller_error:
            warnings.append(self.controller_error)

        frame_diag = FrameDiagnostics(
            hedge_status=self.frame.hedge_status,
            hedge_lag_ms=self.frame.hedge_lag_ms,
            frame_age_ms=(
                max(0, self.now_ms - self.frame.created_at_ms)
                if self.frame.created_at_ms > 0
                else None
            ),
            primary_private_age_ms=_private_age_ms(self.frame.primary_private, now_ms=self.now_ms),
            hedge_private_age_ms=_private_age_ms(self.frame.hedge_private, now_ms=self.now_ms),
        )

        state = self._state_diagnostics(decision)
        models = self._model_scores(decision)
        optimizer = self._optimizer_diagnostics(decision)
        action_pipeline = self._action_pipeline(decision)
        risk = self._risk_diagnostics()
        execution_gates = ExecutionGateDiagnostics(
            executed=self.executed,
            blocked_reasons=tuple(self.blocked_reasons),
        )

        return ControllerDiagnostics(
            actor_id=self.config.actor_id,
            exchange=self.frame.primary.exchange,
            symbol=self.frame.primary.symbol,
            t_end_ms=self.frame.t_end_ms,
            frame_step=self.frame_step,
            created_at_ms=self.now_ms,
            run_id=self.run_id,
            frame=frame_diag,
            state=state,
            models=models,
            optimizer=optimizer,
            action_pipeline=action_pipeline,
            risk=risk,
            execution_gates=execution_gates,
            warnings=tuple(warnings),
        )

    def _state_diagnostics(self, decision: StrategyDecision | None) -> StateDiagnostics:
        if decision is not None:
            return StateDiagnostics(
                mid=decision.mid,
                primary_inventory=decision.inventory,
                primary_cash=decision.cash,
                hedge_inventory=0.0,
                hedge_cash=0.0,
                primary_buy_vwap=decision.primary_buy_vwap,
                primary_sell_vwap=decision.primary_sell_vwap,
                hedge_buy_vwap=decision.hedge_buy_vwap,
                hedge_sell_vwap=decision.hedge_sell_vwap,
            )
        return StateDiagnostics(
            primary_inventory=self.trading_state.inventory,
            primary_cash=self.trading_state.cash,
        )

    def _model_scores(self, decision: StrategyDecision | None) -> dict[str, ModelScore]:
        if decision is None:
            return {}
        if decision.model_scores:
            return dict(decision.model_scores)
        return {}

    def _optimizer_diagnostics(self, decision: StrategyDecision | None) -> OptimizerDiagnostics:
        if decision is None:
            return OptimizerDiagnostics(
                config={"kind": (self.config.optimizer or {}).get("kind", "monte_carlo_mpc_tensor_batched")},
            )
        opt = dict(decision.diagnostics.optimizer)
        return OptimizerDiagnostics(
            config={"kind": opt.pop("kind", (self.config.optimizer or {}).get("kind", "monte_carlo_mpc_tensor_batched"))},
            objective={
                "legacy": {
                    "last_expected_delta_wealth": opt.get("last_expected_delta_wealth"),
                    "last_components": opt.get("last_components"),
                }
            },
            action={"continuous": dict(decision.diagnostics.continuous_action_depth)},
            rollout={"log": opt.get("log", "")},
        )

    def _action_pipeline(self, decision: StrategyDecision | None) -> ActionPipelineDiagnostics:
        if decision is None:
            return ActionPipelineDiagnostics()
        action = decision.action
        ladder_orders = tuple(decision.ladder_orders)
        venue_orders = tuple(decision.orders)
        approved_orders = tuple(order for a in self.approved_actions for order in a.orders_to_create)
        return ActionPipelineDiagnostics(
            continuous=dict(decision.diagnostics.continuous_action_depth),
            ladder=_order_stage_summary_from_dicts(ladder_orders),
            venue_quantized=_order_stage_summary_from_dicts(venue_orders),
            capped=_order_stage_summary_from_dicts(venue_orders),
            risk_approved=_order_stage_summary(approved_orders),
            venue_accepted=_order_stage_summary(approved_orders) if self.executed else {},
        )

    def _risk_diagnostics(self) -> RiskDiagnostics:
        limits = self.config.risk
        return RiskDiagnostics(
            approved=len(self.approved_actions) >= len(self.proposed_actions) and not self.risk_rejection_counts,
            rejection_counts=dict(self.risk_rejection_counts),
            exposure_before={
                "inventory": self.trading_state.inventory,
                "cash": self.trading_state.cash,
            },
            exposure_after={
                "inventory": self.trading_state.inventory,
                "cash": self.trading_state.cash,
            },
            limits=_risk_limits_payload(limits),
        )


def _order_stage_summary_from_dicts(orders: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    buy_qty = sell_qty = 0.0
    buy_notional = sell_notional = 0.0
    for order in orders:
        qty = float(order.get("qty", 0.0))
        price = float(order.get("price", 0.0))
        side = str(order.get("side", "")).lower().strip()
        if side == "buy":
            buy_qty += qty
            buy_notional += float(order.get("notional", price * qty))
        elif side == "sell":
            sell_qty += qty
            sell_notional += float(order.get("notional", price * qty))
    return {
        "order_count": len(orders),
        "buy_qty": buy_qty,
        "sell_qty": sell_qty,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
    }


def _risk_limits_payload(limits: RiskRuntimeConfig) -> dict[str, Any]:
    return {
        "enabled": limits.enabled,
        "max_inventory": limits.max_inventory,
        "max_data_age_ms": limits.max_data_age_ms,
    }


def execution_blocked_reasons(
    *,
    config: ControllerRuntimeConfig,
    frame_step: int,
    approved: list[Action],
    observe_only_control: bool,
    primary_manager_present: bool,
) -> list[str]:
    """Derive why live execution did not run (mirrors ``StrategyRunner._should_execute_live``)."""

    reasons: list[str] = []
    if not approved:
        reasons.append("no_approved_actions")
    if config.observe_only:
        reasons.append("observe_only_config")
    if observe_only_control:
        reasons.append("observe_only_control")
    if frame_step <= config.action_start_after_steps:
        reasons.append("warmup")
    if not primary_manager_present:
        reasons.append("no_manager")
    if approved and len(approved) < 1:
        reasons.append("risk_rejected")
    return reasons
