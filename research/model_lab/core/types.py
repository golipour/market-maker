"""Shared types for walk-forward model evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class IntervalStep:
    """One aggregated market interval (same cadence as runtime ``MarketEvent``)."""

    ts: datetime
    t_end_ms: int
    exchange: str
    symbol: str
    interval_ms: int
    buy_volume: float
    sell_volume: float
    best_bid: float
    best_ask: float
    microprice: float
    spread: float
    mid: float
    orderbook: Any  # object with .bids, .asks, .mid (price/qty levels)

    @property
    def total_volume(self) -> float:
        return float(self.buy_volume) + float(self.sell_volume)


@dataclass(frozen=True)
class StepScore:
    """One-step prequential score (predict current step before update)."""

    step_index: int
    ts: datetime
    model_name: str
    model_kind: str
    family: str
    log_likelihood: float | None = None
    nll: float | None = None
    skip_reason: str | None = None
    standardized_residual: float | None = None
    observation: dict[str, Any] = field(default_factory=dict)
    params_before: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WalkForwardSummary:
    """Aggregated metrics for one model run."""

    model_name: str
    model_kind: str
    family: str
    n_steps: int
    n_scored: int
    n_skipped: int
    mean_nll: float | None
    median_nll: float | None
    total_nll: float | None
    mean_log_likelihood: float | None
    skip_reasons: dict[str, int] = field(default_factory=dict)


def orderbook_from_levels(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    *,
    mid: float | None = None,
) -> SimpleNamespace:
    """Minimal orderbook object compatible with production price models."""

    bid_levels = [SimpleNamespace(price=float(p), qty=float(q)) for p, q in bids if q > 0]
    ask_levels = [SimpleNamespace(price=float(p), qty=float(q)) for p, q in asks if q > 0]
    if mid is None:
        if bid_levels and ask_levels:
            mid = 0.5 * (float(bid_levels[0].price) + float(ask_levels[0].price))
        else:
            mid = 0.0
    return SimpleNamespace(bids=bid_levels, asks=ask_levels, mid=float(mid))


def mpc_step_from_interval(step: IntervalStep) -> tuple[Any, ...]:
    """Tuple shape expected by production torch flow/price ``update`` methods."""

    return (
        int(step.t_end_ms),
        step.orderbook,
        float(step.buy_volume),
        float(step.sell_volume),
        0.0,
        0.0,
        0.0,
        (),
        (),
        None,
        None,
        None,
        None,
        0.0,
        0.0,
    )
