"""Typed controller diagnostics contract (version 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm.core.models.market import RawOrderbook

from mm.core.payload_schema import (
    CONTROLLER_DIAGNOSTICS_TYPE,
    CONTROLLER_DIAGNOSTICS_VERSION,
    controller_diagnostics_event,
)


def orderbook_snapshot_payload(book: RawOrderbook | None) -> dict[str, Any]:
    """JSON-friendly local orderbook levels for diagnostics and Grafana."""

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
    global_reward_mean: float | None = None


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
    capped: dict[str, Any] = field(default_factory=dict)
    target_book: dict[str, Any] = field(default_factory=dict)
    risk_approved_target: dict[str, Any] = field(default_factory=dict)
    planned_cancel: dict[str, Any] = field(default_factory=dict)
    planned_place: dict[str, Any] = field(default_factory=dict)
    rules_valid: dict[str, Any] = field(default_factory=dict)
    router_accepted: dict[str, Any] = field(default_factory=dict)
    kept: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FrameDiagnostics:
    global_status: str = "disabled"
    global_lag_ms: int | None = None
    frame_age_ms: int | None = None
    frame_created_at_ms: int | None = None
    local_orderbook_timestamp_ms: int | None = None
    local_private_age_ms: int | None = None
    global_private_age_ms: int | None = None
    sync_status: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StateDiagnostics:
    mid: float | None = None
    local_inventory: float = 0.0
    local_cash: float = 0.0
    global_inventory: float = 0.0
    global_cash: float = 0.0
    local_buy_vwap: float | None = None
    local_sell_vwap: float | None = None
    global_buy_vwap: float | None = None
    global_sell_vwap: float | None = None
    local_base_wallet: dict[str, Any] = field(default_factory=dict)
    local_quote_wallet: dict[str, Any] = field(default_factory=dict)
    global_base_wallet: dict[str, Any] = field(default_factory=dict)
    global_quote_wallet: dict[str, Any] = field(default_factory=dict)
    local_wealth: float = 0.0
    global_wealth: float | None = None


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
    frame_id: str = ""
    decision_id: str = ""
    target_book_id: str = ""
    frame: FrameDiagnostics = field(default_factory=FrameDiagnostics)
    state: StateDiagnostics = field(default_factory=StateDiagnostics)
    models: dict[str, ModelScore] = field(default_factory=dict)
    optimizer: OptimizerDiagnostics = field(default_factory=OptimizerDiagnostics)
    action_pipeline: ActionPipelineDiagnostics = field(default_factory=ActionPipelineDiagnostics)
    risk: RiskDiagnostics = field(default_factory=RiskDiagnostics)
    execution_gates: ExecutionGateDiagnostics = field(default_factory=ExecutionGateDiagnostics)
    warnings: tuple[str, ...] = ()
    cleared_orderbook: dict[str, Any] = field(default_factory=dict)
    global_cleared_orderbook: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return controller_diagnostics_event(self)
