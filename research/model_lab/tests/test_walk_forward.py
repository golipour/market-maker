"""Synthetic walk-forward smoke tests (no database required)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research.model_lab.core.types import IntervalStep, orderbook_from_levels
from research.model_lab.core.walk_forward import run_walk_forward
from research.model_lab.models.baselines import ConstantZeroFlowModel, GaussianMidRandomWalkModel
from research.model_lab.models.registry import build_model


def _synthetic_steps(n: int = 50) -> list[IntervalStep]:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    steps: list[IntervalStep] = []
    mid = 160_000.0
    for i in range(n):
        mid += (i % 5 - 2) * 2.0
        bid = mid - 50.0
        ask = mid + 50.0
        ob = orderbook_from_levels([(bid, 100.0), (bid - 10, 200.0)], [(ask, 100.0), (ask + 10, 200.0)], mid=mid)
        steps.append(
            IntervalStep(
                ts=base + timedelta(seconds=30 * i),
                t_end_ms=int((base + timedelta(seconds=30 * i)).timestamp() * 1000),
                exchange="nobitex",
                symbol="USDT_IRT",
                interval_ms=30000,
                buy_volume=float((i * 17) % 400),
                sell_volume=float((i * 11) % 300),
                best_bid=bid,
                best_ask=ask,
                microprice=mid,
                spread=ask - bid,
                mid=mid,
                orderbook=ob,
            )
        )
    return steps


def test_walk_forward_zero_flow_baseline() -> None:
    result = run_walk_forward(ConstantZeroFlowModel(), _synthetic_steps())
    assert result.summary is not None
    assert result.summary.n_scored > 0
    assert result.summary.mean_nll is not None


def test_walk_forward_mid_random_walk() -> None:
    result = run_walk_forward(GaussianMidRandomWalkModel(), _synthetic_steps())
    assert result.summary is not None
    assert result.summary.n_scored > 0


def test_build_production_flow_model() -> None:
    model = build_model("flow:hurdle_lognormal")
    result = run_walk_forward(model, _synthetic_steps(30))
    assert result.summary is not None
    assert result.summary.n_scored > 0


def test_build_production_residual_price_model() -> None:
    model = build_model("price:residual")
    result = run_walk_forward(model, _synthetic_steps(30))
    assert result.summary is not None
    assert result.summary.n_scored > 0
