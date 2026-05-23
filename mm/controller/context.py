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
    orderbook_snapshot_payload,
)
from mm.execution.plan import ActionPipelineReport, ActionStageReport, ExecutionSyncReport
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
    risk_exposure_before: dict[str, float] = field(default_factory=dict)
    risk_exposure_after: dict[str, float] = field(default_factory=dict)
    action_pipeline_report: ActionPipelineReport | None = None

    def record_controller_error(self, exc: BaseException) -> None:
        self.controller_error = f"{type(exc).__name__}: {exc}"
        self.decision = None

    def record_decision(self, decision: StrategyDecision) -> None:
        self.decision = decision
        if decision.action_pipeline_report is not None:
            self.action_pipeline_report = decision.action_pipeline_report

    def record_risk_result(
        self,
        proposed: list[Action],
        approved: list[Action],
        *,
        rejection_counts: dict[str, int] | None = None,
        exposure_before: dict[str, float] | None = None,
        exposure_after: dict[str, float] | None = None,
    ) -> None:
        self.proposed_actions = list(proposed)
        self.approved_actions = list(approved)
        if exposure_before is not None:
            self.risk_exposure_before = dict(exposure_before)
        if exposure_after is not None:
            self.risk_exposure_after = dict(exposure_after)
        if rejection_counts:
            self.risk_rejection_counts = dict(rejection_counts)
        elif len(approved) < len(proposed):
            self.risk_rejection_counts["actions_filtered"] = len(proposed) - len(approved)
            proposed_orders = sum(len(a.orders) for a in proposed)
            approved_orders = sum(len(a.orders) for a in approved)
            if approved_orders < proposed_orders:
                self.risk_rejection_counts["orders_filtered"] = proposed_orders - approved_orders

    def apply_risk_to_pipeline(self, risk_report: ActionStageReport) -> None:
        if self.action_pipeline_report is None:
            return
        self.action_pipeline_report.risk_approved = risk_report

    def apply_execution_to_pipeline(self, execution_report: ExecutionSyncReport, *, executed: bool) -> None:
        if self.action_pipeline_report is None:
            return
        self.action_pipeline_report.target = execution_report.target
        self.action_pipeline_report.kept = execution_report.kept
        self.action_pipeline_report.canceled = execution_report.canceled
        self.action_pipeline_report.placed = ActionStageReport(
            order_count=execution_report.limit_valid.order_count,
            buy_qty=execution_report.limit_valid.buy_qty,
            sell_qty=execution_report.limit_valid.sell_qty,
            buy_notional=execution_report.limit_valid.buy_notional,
            sell_notional=execution_report.limit_valid.sell_notional,
            rejection_counts=dict(execution_report.venue_rejection_counts),
        )
        self.action_pipeline_report.venue_accepted = (
            execution_report.accepted if executed else self.action_pipeline_report.placed
        )

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
            cleared_orderbook=orderbook_snapshot_payload(self.clean_frame.primary.orderbook),
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
        if "objective" in opt and isinstance(opt.get("objective"), dict):
            return OptimizerDiagnostics(
                config=dict(opt.get("config", {})),
                timing=dict(opt.get("timing", {})),
                objective=dict(opt.get("objective", {})),
                action=dict(opt.get("action", {})),
                gradients=dict(opt.get("gradients", {})),
                rollout=dict(opt.get("rollout", {})),
            )
        return OptimizerDiagnostics(
            config={"kind": opt.get("kind", (self.config.optimizer or {}).get("kind", "monte_carlo_mpc_tensor_batched"))},
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
        if self.action_pipeline_report is not None:
            report = self.action_pipeline_report
            return ActionPipelineDiagnostics(
                continuous=dict(report.continuous),
                ladder=report.ladder.as_dict(),
                venue_quantized=report.venue_quantized.as_dict(),
                capped=report.capped.as_dict(),
                risk_approved=report.risk_approved.as_dict(),
                target=report.target.as_dict(),
                kept=report.kept.as_dict(),
                canceled=report.canceled.as_dict(),
                placed=report.placed.as_dict(),
                venue_accepted=report.venue_accepted.as_dict(),
            )
        if decision is None:
            return ActionPipelineDiagnostics()
        approved_orders = tuple(order for a in self.approved_actions for order in a.orders)
        return ActionPipelineDiagnostics(
            continuous=dict(decision.diagnostics.continuous_action_depth),
            ladder=_order_stage_summary_from_dicts(tuple(decision.ladder_orders)),
            venue_quantized=_order_stage_summary_from_dicts(tuple(decision.orders)),
            capped=_order_stage_summary_from_dicts(tuple(decision.orders)),
            risk_approved=_order_stage_summary(approved_orders),
            venue_accepted=_order_stage_summary(approved_orders) if self.executed else {},
        )

    def _risk_diagnostics(self) -> RiskDiagnostics:
        limits = self.config.risk
        orders_ok = all(
            len(a.orders) == len(p.orders)
            for a, p in zip(self.approved_actions, self.proposed_actions)
        ) if self.proposed_actions else True
        exposure_before = self.risk_exposure_before or {
            "inventory": self.trading_state.inventory,
            "cash": self.trading_state.cash,
        }
        exposure_after = self.risk_exposure_after or exposure_before
        return RiskDiagnostics(
            approved=orders_ok and not self.risk_rejection_counts,
            rejection_counts=dict(self.risk_rejection_counts),
            exposure_before=exposure_before,
            exposure_after=exposure_after,
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
        "primary": limits.primary.as_dict(),
        "hedge": limits.hedge.as_dict(),
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
