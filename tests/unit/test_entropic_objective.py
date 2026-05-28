from __future__ import annotations

import pytest
import torch

from mm.mpc.reward_tensor import (
    objective_entropic_return,
    objective_threshold_shortfall_penalty,
    terminal_objective,
)


def test_entropic_return_prefers_better_tail() -> None:
    good = torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.float64)
    bad = torch.tensor([0.01, 0.02, -0.20, 0.04], dtype=torch.float64)
    score_good = float(objective_entropic_return(good, beta=20.0).item())
    score_bad = float(objective_entropic_return(bad, beta=20.0).item())
    assert score_good > score_bad


def test_threshold_shortfall_penalty_is_one_sided_and_normalized() -> None:
    cash = torch.tensor([[80.0, 120.0], [50.0, 100.0]], dtype=torch.float64)
    inv = torch.tensor([[0.0, 2.0], [4.0, 0.0]], dtype=torch.float64)
    penalty = float(
        objective_threshold_shortfall_penalty(
            cash,
            inv,
            cash_threshold=100.0,
            inventory_threshold=2.0,
        ).item()
    )
    assert penalty == pytest.approx(((0.2 + 1.0) + 0.0 + 0.5 + 1.0) / 4.0)


def test_terminal_objective_has_two_components() -> None:
    returns = torch.tensor([0.01, 0.02, -0.05, 0.03], dtype=torch.float64)
    cash_path = torch.tensor([[100.0], [80.0], [120.0], [100.0]], dtype=torch.float64)
    inv_path = torch.tensor([[1.0], [2.0], [0.0], [3.0]], dtype=torch.float64)
    objective, components = terminal_objective(
        returns,
        cash_path,
        inv_path,
        return_risk_aversion_beta=10.0,
        inventory_threshold_penalty_weight=0.5,
        cash_threshold=100.0,
        inventory_threshold=2.0,
    )
    assert set(components) == {"risk_adjusted_return", "threshold_shortfall_penalty"}
    assert components["threshold_shortfall_penalty"] <= 0.0
    assert torch.isclose(objective, sum(components.values()))


def test_entropic_return_gradients_flow() -> None:
    returns = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
    score = objective_entropic_return(returns, beta=5.0)
    score.backward()
    assert returns.grad is not None
    assert torch.isfinite(returns.grad).all()


def test_entropic_return_stable_for_extreme_negative_tail() -> None:
    returns = torch.tensor([0.01, 0.02, -500.0, 0.03], dtype=torch.float64)
    score = objective_entropic_return(returns, beta=50.0)
    assert torch.isfinite(score)
    assert abs(float(score.item())) < 1e6


def test_entropic_return_matches_unshifted_formula_for_moderate_returns() -> None:
    returns = torch.tensor([0.01, 0.02, -0.05, 0.03], dtype=torch.float64)
    beta = 10.0
    shifted = float(objective_entropic_return(returns, beta=beta).item())
    naive_log_mean = torch.logsumexp(-beta * returns, dim=0) - torch.log(
        torch.tensor(float(returns.numel()), dtype=returns.dtype)
    )
    naive = float((-naive_log_mean / beta).item())
    assert shifted == pytest.approx(naive, rel=1e-9, abs=1e-12)
