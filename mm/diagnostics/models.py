"""Typed controller diagnostics contract (version 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CONTROLLER_DIAGNOSTICS_TYPE = "ControllerDiagnostics"
CONTROLLER_DIAGNOSTICS_VERSION = 1


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
    expected_total_reward: float | None = None
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
    risk_approved: dict[str, Any] = field(default_factory=dict)
    execution_deduped: dict[str, Any] = field(default_factory=dict)
    venue_limit_valid: dict[str, Any] = field(default_factory=dict)
    venue_accepted: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FrameDiagnostics:
    hedge_status: str = "disabled"
    hedge_lag_ms: int | None = None
    frame_age_ms: int | None = None
    primary_private_age_ms: int | None = None
    hedge_private_age_ms: int | None = None


@dataclass(frozen=True, slots=True)
class StateDiagnostics:
    mid: float = 0.0
    primary_inventory: float = 0.0
    primary_cash: float = 0.0
    hedge_inventory: float = 0.0
    hedge_cash: float = 0.0
    primary_buy_vwap: float = 0.0
    primary_sell_vwap: float = 0.0
    hedge_buy_vwap: float | None = None
    hedge_sell_vwap: float | None = None


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
            },
            "warnings": list(self.warnings),
        }


def _frame_payload(frame: FrameDiagnostics) -> dict[str, Any]:
    return {
        "hedge_status": frame.hedge_status,
        "hedge_lag_ms": frame.hedge_lag_ms,
        "frame_age_ms": frame.frame_age_ms,
        "primary_private_age_ms": frame.primary_private_age_ms,
        "hedge_private_age_ms": frame.hedge_private_age_ms,
    }


def _state_payload(state: StateDiagnostics) -> dict[str, Any]:
    return {
        "mid": state.mid,
        "primary_inventory": state.primary_inventory,
        "primary_cash": state.primary_cash,
        "hedge_inventory": state.hedge_inventory,
        "hedge_cash": state.hedge_cash,
        "primary_buy_vwap": state.primary_buy_vwap,
        "primary_sell_vwap": state.primary_sell_vwap,
        "hedge_buy_vwap": state.hedge_buy_vwap,
        "hedge_sell_vwap": state.hedge_sell_vwap,
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
        "risk_approved": dict(pipeline.risk_approved),
        "execution_deduped": dict(pipeline.execution_deduped),
        "venue_limit_valid": dict(pipeline.venue_limit_valid),
        "venue_accepted": dict(pipeline.venue_accepted),
    }


def _risk_payload(risk: RiskDiagnostics) -> dict[str, Any]:
    return {
        "approved": risk.approved,
        "rejection_counts": dict(risk.rejection_counts),
        "exposure_before": dict(risk.exposure_before),
        "exposure_after": dict(risk.exposure_after),
        "limits": dict(risk.limits),
    }
