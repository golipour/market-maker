from __future__ import annotations

import math

import pytest
import torch

from mm.mpc.mpc_state import OrderbookRelative, State
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched
from mm.mpc.optimizer_diagnostics import EpochObjectiveMetrics, compute_contribution_fractions
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


def test_contribution_fractions_sum_to_one() -> None:
    components = {
        "risk_adjusted_return": 10.0,
        "local_risk_adjusted_return": 6.0,
        "global_risk_adjusted_return": 4.0,
        "book_quality": 2.0,
        "book_presence": 0.8,
    }
    fracs = compute_contribution_fractions(components)
    assert set(fracs) == {"risk_adjusted_return", "book_quality"}
    assert fracs["risk_adjusted_return"] == pytest.approx(10.0 / 12.0, rel=1e-9)
    assert fracs["book_quality"] == pytest.approx(2.0 / 12.0, rel=1e-9)
    assert sum(fracs.values()) == pytest.approx(1.0, rel=1e-9)


def test_contribution_fractions_include_soft_objective_terms() -> None:
    components = {
        "risk_adjusted_return": 8.0,
        "book_quality": 1.0,
        "viability_penalty": -0.5,
        "global_book_quality": 0.25,
        "global_viability_penalty": -0.25,
        "book_presence": 0.9,
        "book_tightness": 0.7,
    }
    fracs = compute_contribution_fractions(components)
    assert set(fracs) == {
        "risk_adjusted_return",
        "book_quality",
        "viability_penalty",
        "global_book_quality",
        "global_viability_penalty",
    }
    total_abs = 8.0 + 1.0 + 0.5 + 0.25 + 0.25
    assert fracs["risk_adjusted_return"] == pytest.approx(8.0 / total_abs, rel=1e-9)
    assert sum(fracs.values()) == pytest.approx(1.0, rel=1e-9)


def test_objective_snapshot_includes_contribution_fractions() -> None:
    metrics = EpochObjectiveMetrics.from_tensors(
        objective_value=5.0,
        components={"risk_adjusted_return": 5.0},
        expected_total_reward=0.0,
        expected_total_return=0.0,
        expected_next_reward=0.0,
        main_reward_mean=0.0,
        global_reward_mean=0.0,
    )
    snap = metrics.to_snapshot()
    assert snap.contribution_fractions["risk_adjusted_return"] == pytest.approx(1.0)


def test_objective_components_sum_equals_value() -> None:
    returns = torch.tensor([0.01, 0.02, -0.01], dtype=torch.float64)
    objective, components = terminal_objective(
        returns,
        return_risk_aversion_beta=5.0,
    )
    metrics = EpochObjectiveMetrics.from_tensors(
        objective_value=float(objective.item()),
        components={k: float(v.item()) for k, v in components.items()},
        expected_total_reward=2.0,
        expected_total_return=0.01,
        expected_next_reward=1.0,
        main_reward_mean=2.0,
        global_reward_mean=0.0,
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
        cumulative_qty_floor_at_spread_max=1.0,
        cumulative_qty_cap_at_spread_max=10.0,
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
    assert set(best["components"]) == {"risk_adjusted_return"}
    assert best["expected_total_return"] is not None
    fracs = best.get("contribution_fractions") or {}
    assert fracs
    assert sum(float(v) for v in fracs.values()) == pytest.approx(1.0, rel=1e-5)
    assert payload["action"]["best"]["effective_depth"]["bid"]
    assert payload["action"]["initial"]["effective_depth"]["bid"]


def test_optimizer_ignores_nonfinite_warm_start_action() -> None:
    mpc = MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        _FixedFlowModel(),
        _FixedExecutionModel(),
        n_components=1,
        n_samples=4,
        horizon=1,
        spread_max=20.0,
        spread_step=5.0,
        adam_epochs=2,
        adam_lr=0.1,
        fixed_mc_samples=True,
        warm_start=True,
        device="cpu",
        cumulative_qty_floor_at_spread_max=1.0,
        cumulative_qty_cap_at_spread_max=10.0,
    )
    mpc._last_raw_flat = torch.full((6,), float("nan"), dtype=torch.float64).numpy()

    action, diagnostics = mpc.optimize_with_diagnostics(_minimal_state())
    best_flat = diagnostics.to_payload()["action"]["best"]["raw_params"]["flat"]

    assert all(math.isfinite(float(x)) for x in action.to_flat())
    assert all(math.isfinite(float(x)) for x in best_flat)
    assert diagnostics.best_epoch is not None
    assert action.n_components == 1


def test_payload_contribution_fractions_match_final_only() -> None:
    mpc = MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        _FixedFlowModel(),
        _FixedExecutionModel(),
        n_components=1,
        n_samples=2,
        horizon=1,
        spread_max=10.0,
        spread_step=5.0,
        adam_epochs=2,
        fixed_mc_samples=True,
        device="cpu",
        cumulative_qty_floor_at_spread_max=1.0,
        cumulative_qty_cap_at_spread_max=10.0,
    )
    _, diagnostics = mpc.optimize_with_diagnostics(_minimal_state())
    payload = diagnostics.to_payload()
    top = payload["objective"]["contribution_fractions"]
    final = payload["objective"]["final"]["contribution_fractions"]
    assert top == final
    assert sum(float(v) for v in top.values()) == pytest.approx(1.0, rel=1e-5)
    assert top.keys() == final.keys()


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
        cumulative_qty_floor_at_spread_max=1.0,
        cumulative_qty_cap_at_spread_max=10.0,
    )
    _, diagnostics = mpc.optimize_with_diagnostics(_minimal_state())
    improvement = diagnostics.to_payload()["objective"]["improvement"]
    assert "best_minus_initial" in improvement
    assert "final_minus_initial" in improvement
    if improvement["best_minus_initial"] is not None:
        assert math.isfinite(float(improvement["best_minus_initial"]))
