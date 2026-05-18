from __future__ import annotations

import math

import pytest
import torch

from mm.mpc.mpc_state import OrderbookRelative, State
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched
from mm.mpc.optimizer_diagnostics import EpochObjectiveMetrics
from mm.mpc.reward_tensor import terminal_objective


class _FixedFlowModel:
    def sample(self, sample_shape, device=None, dtype=None):
        mod = torch
        return (
            mod.full(sample_shape, 2.0, dtype=dtype or mod.float64, device=device),
            mod.full(sample_shape, 1.0, dtype=dtype or mod.float64, device=device),
        )


class _FixedPriceModel:
    def sample_residual(self, sample_shape, device=None, dtype=None):
        mod = torch
        z = mod.zeros(sample_shape, dtype=dtype or mod.float64, device=device)
        return z, z


class _FixedExecutionModel:
    maker_fee = 0.0
    hedge_fee = 0.0


def _minimal_state() -> State:
    return State(
        mid=100.0,
        ob_rel=OrderbookRelative(
            bids=((1.0, 50.0), (2.0, 30.0)),
            asks=((1.0, 40.0), (2.0, 25.0)),
        ),
        buy_vol=2.0,
        sell_vol=1.0,
        inventory=0.0,
        cash=10_000.0,
        buy_vwap=99.5,
        sell_vwap=100.5,
    )


def test_objective_components_sum_equals_value() -> None:
    delta_w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    inv = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64)
    objective, components = terminal_objective(
        delta_w,
        inv,
        expected_pnl_weight=1.0,
        inventory_target_penalty_weight=0.1,
        target_inventory=0.0,
        pnl_volatility_penalty_weight=0.05,
    )
    metrics = EpochObjectiveMetrics.from_tensors(
        objective_value=float(objective.item()),
        components={k: float(v.item()) for k, v in components.items()},
        expected_total_reward=2.0,
        expected_next_reward=1.0,
        main_reward_mean=2.0,
        hedge_reward_mean=0.0,
    )
    assert sum(metrics.components.values()) == pytest.approx(metrics.objective_value, rel=1e-9)


def test_optimizer_diagnostics_best_not_worse_than_initial() -> None:
    mpc = MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        _FixedFlowModel(),
        _FixedExecutionModel(),
        n_components=1,
        n_samples=4,
        horizon=1,
        spread_max=20.0,
        spread_step=5.0,
        adam_epochs=8,
        adam_lr=0.1,
        fixed_mc_samples=True,
        warm_start=False,
        device="cpu",
        inventory_target_penalty_weight=0.0,
        pnl_volatility_penalty_weight=0.0,
        quote_shape_uniformity_penalty_weight=0.0,
        quote_size_target_penalty_weight=0.0,
        quote_spread_penalty_weight=0.0,
    )
    action, diagnostics = mpc.optimize_with_diagnostics(_minimal_state())
    payload = diagnostics.to_payload()
    initial = payload["objective"]["initial"]
    best = payload["objective"]["best"]
    assert initial["value"] is not None
    assert best["value"] is not None
    assert float(best["value"]) >= float(initial["value"]) - 1e-9
    comp_sum = sum(float(v) for v in best["components"].values())
    assert comp_sum == pytest.approx(float(best["value"]), rel=1e-5)
    assert payload["action"]["best"]["effective_depth"]["bid"]
    assert payload["action"]["initial"]["effective_depth"]["bid"]
    assert diagnostics.best_epoch is not None
    assert action.n_components == 1


def test_objective_snapshot_payload_improvement_fields() -> None:
    mpc = MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        _FixedFlowModel(),
        _FixedExecutionModel(),
        n_components=1,
        n_samples=2,
        horizon=1,
        spread_max=10.0,
        spread_step=5.0,
        adam_epochs=3,
        fixed_mc_samples=True,
        device="cpu",
    )
    _, diagnostics = mpc.optimize_with_diagnostics(_minimal_state())
    improvement = diagnostics.to_payload()["objective"]["improvement"]
    assert "best_minus_initial" in improvement
    assert "final_minus_initial" in improvement
    if improvement["best_minus_initial"] is not None:
        assert math.isfinite(float(improvement["best_minus_initial"]))
