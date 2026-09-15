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


# Envelope safety cap after the spread window. Generous enough for Grafana depth
# charts, small enough to stay under the Redis 64KB diagnostics limit.
DIAGNOSTICS_ORDERBOOK_LEVELS_CAP = 80
DIAGNOSTICS_ORDERBOOK_LEVELS_CAP_MAX = 100


def diagnostics_orderbook_level_cap(*, grid_levels: int | None = None) -> int:
    """Count backstop for diagnostics books, independent of venue tick density."""

    if grid_levels is None:
        return DIAGNOSTICS_ORDERBOOK_LEVELS_CAP
    return min(
        DIAGNOSTICS_ORDERBOOK_LEVELS_CAP_MAX,
        max(DIAGNOSTICS_ORDERBOOK_LEVELS_CAP, 2 * int(grid_levels)),
    )


def orderbook_snapshot_payload(
    book: RawOrderbook | None,
    *,
    mid: float | None = None,
    spread_max: float | None = None,
    max_levels: int = DIAGNOSTICS_ORDERBOOK_LEVELS_CAP,
) -> dict[str, Any]:
    """Return the near-market ladder the controller actually consumes.

    Primary bound is ``|price - mid| <= spread_max`` (same window as MPC
    ``ob_rel``). A count cap is applied only after that window, so a dense
    tick book cannot blow the diagnostics envelope.
    """

    if book is None:
        return {"bids": [], "asks": []}
    if max_levels <= 0:
        raise ValueError("max_levels must be positive")
    raw_bids = list(book.bids)
    raw_asks = list(book.asks)
    window_mid = float(mid) if mid is not None and mid > 0 else None
    window_spread = float(spread_max) if spread_max is not None and spread_max > 0 else None
    if window_mid is not None and window_spread is not None:
        window_bids = [
            level
            for level in raw_bids
            if level.qty > 0 and 0.0 <= (window_mid - float(level.price)) <= window_spread
        ]
        window_asks = [
            level
            for level in raw_asks
            if level.qty > 0 and 0.0 <= (float(level.price) - window_mid) <= window_spread
        ]
    else:
        window_bids = [level for level in raw_bids if level.qty > 0]
        window_asks = [level for level in raw_asks if level.qty > 0]
    kept_bids = window_bids[:max_levels]
    kept_asks = window_asks[:max_levels]
    payload: dict[str, Any] = {
        "exchange": book.exchange,
        "symbol": book.symbol,
        "timestamp_ms": book.timestamp_ms,
        "bids": [{"price": float(level.price), "qty": float(level.qty)} for level in kept_bids],
        "asks": [{"price": float(level.price), "qty": float(level.qty)} for level in kept_asks],
    }
    if window_mid is not None:
        payload["mid"] = window_mid
    if window_spread is not None:
        payload["spread_max"] = window_spread
    dropped = (
        len(kept_bids) < len(raw_bids)
        or len(kept_asks) < len(raw_asks)
        or len(window_bids) > max_levels
        or len(window_asks) > max_levels
    )
    if dropped:
        payload.update(
            {
                "bids_count": len(raw_bids),
                "asks_count": len(raw_asks),
                "bids_in_window": len(window_bids),
                "asks_in_window": len(window_asks),
                "levels_limit": max_levels,
                "truncated": True,
            }
        )
    return payload


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
    ledger_source: str = "physical"


@dataclass(frozen=True, slots=True)
class PhysicalStateDiagnostics:
    local_inventory: float = 0.0
    local_cash: float = 0.0
    global_inventory: float = 0.0
    global_cash: float = 0.0
    consolidated_base_inventory: float = 0.0
    consolidated_quote_wealth: float = 0.0


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
    ledger_source: str = "physical"
    physical_state: PhysicalStateDiagnostics | None = None

    def as_payload(self) -> dict[str, Any]:
        return controller_diagnostics_event(self)
