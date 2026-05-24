"""Typed controller diagnostics contract (version 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm.core.models.market import RawOrderbook

CONTROLLER_DIAGNOSTICS_TYPE = "ControllerDiagnostics"
CONTROLLER_DIAGNOSTICS_VERSION = 1


def orderbook_snapshot_payload(book: RawOrderbook | None) -> dict[str, Any]:
    """JSON-friendly primary orderbook levels for diagnostics and Grafana."""

    if book is None:
        return {"bids": [], "asks": []}
    return {
        "exchange": book.exchange,
        "symbol": book.symbol,
        "timestamp_ms": book.timestamp_ms,
        "bids": [{"price": float(level.price), "qty": float(level.qty)} for level in book.bids],
        "asks": [{"price": float(level.price), "qty": float(level.qty)} for level in book.asks],
    }


@dataclass(frozen=True, slots=True)
class ModelScore:
    """Per-model score snapshot; NLL fields filled in Phase 2."""

    model: str
    kind: str = ""
    updated: bool = False
    skip_reason: str | None = None
    observation: dict[str, Any] = field(default_factory=dict)
    params_before: dict[str, Any] = field(default_factory=dict)
    params_after: dict[str, Any] = field(default_factory=dict)
    log_likelihood: float | None = None
    nll: float | None = None
    standardized_residual: float | None = None


@dataclass(frozen=True, slots=True)
class ObjectiveSnapshot:
    value: float | None = None
    loss: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    contribution_fractions: dict[str, float] = field(default_factory=dict)
    expected_total_reward: float | None = None
    expected_total_return: float | None = None
    expected_next_reward: float | None = None
    main_reward_mean: float | None = None
    hedge_reward_mean: float | None = None


@dataclass(frozen=True, slots=True)
class ActionSnapshot:
    effective_depth: dict[str, Any] = field(default_factory=dict)
    raw_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OptimizerDiagnostics:
    config: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    objective: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    gradients: dict[str, Any] = field(default_factory=dict)
    rollout: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionPipelineDiagnostics:
    continuous: dict[str, Any] = field(default_factory=dict)
    ladder: dict[str, Any] = field(default_factory=dict)
    venue_quantized: dict[str, Any] = field(default_factory=dict)
    capped: dict[str, Any] = field(default_factory=dict)
    target_book: dict[str, Any] = field(default_factory=dict)
    risk_approved_target: dict[str, Any] = field(default_factory=dict)
    planned_cancel: dict[str, Any] = field(default_factory=dict)
    planned_place: dict[str, Any] = field(default_factory=dict)
    rules_valid: dict[str, Any] = field(default_factory=dict)
    router_accepted: dict[str, Any] = field(default_factory=dict)
    kept: dict[str, Any] = field(default_factory=dict)
    # Deprecated aliases retained for dashboard migration.
    risk_approved: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    canceled: dict[str, Any] = field(default_factory=dict)
    placed: dict[str, Any] = field(default_factory=dict)
    venue_accepted: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FrameDiagnostics:
    global_status: str = "disabled"
    global_lag_ms: int | None = None
    frame_age_ms: int | None = None
    local_private_age_ms: int | None = None
    global_private_age_ms: int | None = None
    sync_status: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StateDiagnostics:
    mid: float = 0.0
    local_inventory: float = 0.0
    primary_cash: float = 0.0
    global_inventory: float = 0.0
    hedge_cash: float = 0.0
    local_buy_vwap: float = 0.0
    local_sell_vwap: float = 0.0
    global_buy_vwap: float | None = None
    global_sell_vwap: float | None = None
    primary_base_wallet: dict[str, Any] = field(default_factory=dict)
    primary_quote_wallet: dict[str, Any] = field(default_factory=dict)
    hedge_base_wallet: dict[str, Any] = field(default_factory=dict)
    hedge_quote_wallet: dict[str, Any] = field(default_factory=dict)
    primary_wealth: float = 0.0
    hedge_wealth: float | None = None


@dataclass(frozen=True, slots=True)
class RiskDiagnostics:
    approved: bool = True
    rejection_counts: dict[str, int] = field(default_factory=dict)
    exposure_before: dict[str, float] = field(default_factory=dict)
    exposure_after: dict[str, float] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionGateDiagnostics:
    executed: bool = False
    blocked_reasons: tuple[str, ...] = ()
    venue_gates: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ControllerDiagnostics:
    """Separate event from ``StrategyDecision`` for stable Grafana contracts."""

    actor_id: str
    exchange: str
    symbol: str
    t_end_ms: int
    frame_step: int
    created_at_ms: int
    run_id: str = ""
    frame: FrameDiagnostics = field(default_factory=FrameDiagnostics)
    state: StateDiagnostics = field(default_factory=StateDiagnostics)
    models: dict[str, ModelScore] = field(default_factory=dict)
    optimizer: OptimizerDiagnostics = field(default_factory=OptimizerDiagnostics)
    action_pipeline: ActionPipelineDiagnostics = field(default_factory=ActionPipelineDiagnostics)
    risk: RiskDiagnostics = field(default_factory=RiskDiagnostics)
    execution_gates: ExecutionGateDiagnostics = field(default_factory=ExecutionGateDiagnostics)
    warnings: tuple[str, ...] = ()
    cleared_orderbook: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "type": CONTROLLER_DIAGNOSTICS_TYPE,
            "version": CONTROLLER_DIAGNOSTICS_VERSION,
            "actor_id": self.actor_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "t_end_ms": self.t_end_ms,
            "frame_step": self.frame_step,
            "created_at_ms": self.created_at_ms,
            "run_id": self.run_id,
            "frame": _frame_payload(self.frame),
            "state": _state_payload(self.state),
            "models": {name: _model_score_payload(score) for name, score in self.models.items()},
            "optimizer": _optimizer_payload(self.optimizer),
            "action_pipeline": _action_pipeline_payload(self.action_pipeline),
            "risk": _risk_payload(self.risk),
            "execution_gates": {
                "executed": self.execution_gates.executed,
                "blocked_reasons": list(self.execution_gates.blocked_reasons),
                "venue_gates": dict(self.execution_gates.venue_gates),
            },
            "warnings": list(self.warnings),
            "cleared_orderbook": dict(self.cleared_orderbook),
        }


def _frame_payload(frame: FrameDiagnostics) -> dict[str, Any]:
    return {
        "global_status": frame.global_status,
        "global_lag_ms": frame.global_lag_ms,
        "frame_age_ms": frame.frame_age_ms,
        "local_private_age_ms": frame.local_private_age_ms,
        "global_private_age_ms": frame.global_private_age_ms,
        "sync_status": dict(frame.sync_status),
    }


def _state_payload(state: StateDiagnostics) -> dict[str, Any]:
    return {
        "mid": state.mid,
        "local_inventory": state.local_inventory,
        "primary_cash": state.primary_cash,
        "global_inventory": state.global_inventory,
        "hedge_cash": state.hedge_cash,
        "local_buy_vwap": state.local_buy_vwap,
        "local_sell_vwap": state.local_sell_vwap,
        "global_buy_vwap": state.global_buy_vwap,
        "global_sell_vwap": state.global_sell_vwap,
        "primary_base_wallet": dict(state.primary_base_wallet),
        "primary_quote_wallet": dict(state.primary_quote_wallet),
        "hedge_base_wallet": dict(state.hedge_base_wallet),
        "hedge_quote_wallet": dict(state.hedge_quote_wallet),
        "primary_wealth": state.primary_wealth,
        "hedge_wealth": state.hedge_wealth,
    }


def _model_score_payload(score: ModelScore) -> dict[str, Any]:
    return {
        "model": score.model,
        "kind": score.kind,
        "updated": score.updated,
        "skip_reason": score.skip_reason,
        "observation": dict(score.observation),
        "params_before": dict(score.params_before),
        "params_after": dict(score.params_after),
        "log_likelihood": score.log_likelihood,
        "nll": score.nll,
        "standardized_residual": score.standardized_residual,
    }


def _optimizer_payload(optimizer: OptimizerDiagnostics) -> dict[str, Any]:
    return {
        "config": dict(optimizer.config),
        "timing": dict(optimizer.timing),
        "objective": dict(optimizer.objective),
        "action": dict(optimizer.action),
        "gradients": dict(optimizer.gradients),
        "rollout": dict(optimizer.rollout),
    }


def _action_pipeline_payload(pipeline: ActionPipelineDiagnostics) -> dict[str, Any]:
    return {
        "continuous": dict(pipeline.continuous),
        "ladder": dict(pipeline.ladder),
        "venue_quantized": dict(pipeline.venue_quantized),
        "capped": dict(pipeline.capped),
        "target_book": dict(pipeline.target_book),
        "risk_approved_target": dict(pipeline.risk_approved_target),
        "planned_cancel": dict(pipeline.planned_cancel),
        "planned_place": dict(pipeline.planned_place),
        "rules_valid": dict(pipeline.rules_valid),
        "router_accepted": dict(pipeline.router_accepted),
        "kept": dict(pipeline.kept),
        "risk_approved": dict(pipeline.risk_approved),
        "target": dict(pipeline.target),
        "canceled": dict(pipeline.canceled),
        "placed": dict(pipeline.placed),
        "venue_accepted": dict(pipeline.venue_accepted),
        "_deprecated_aliases": {
            "risk_approved": "use risk_approved_target",
            "target": "use target_book for strategy intent",
            "canceled": "use planned_cancel",
            "placed": "use rules_valid (was limit-valid, not exchange placed)",
            "venue_accepted": "use router_accepted",
        },
    }


def _risk_payload(risk: RiskDiagnostics) -> dict[str, Any]:
    return {
        "approved": risk.approved,
        "rejection_counts": dict(risk.rejection_counts),
        "exposure_before": dict(risk.exposure_before),
        "exposure_after": dict(risk.exposure_after),
        "limits": dict(risk.limits),
    }
