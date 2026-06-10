from __future__ import annotations

import pytest
import torch

from mm.mpc.reward_tensor import (
    objective_entropic_return,
    terminal_objective,
)


def test_entropic_return_prefers_better_tail() -> None:
    good = torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.float64)
    bad = torch.tensor([0.01, 0.02, -0.20, 0.04], dtype=torch.float64)
    score_good = float(objective_entropic_return(good, beta=20.0).item())
    score_bad = float(objective_entropic_return(bad, beta=20.0).item())
    assert score_good > score_bad


def test_terminal_objective_is_pure_risk_adjusted_return() -> None:
    returns = torch.tensor([0.01, 0.02, -0.05, 0.03], dtype=torch.float64)
    objective, components = terminal_objective(
        returns,
        return_risk_aversion_beta=10.0,
    )
    assert set(components) == {"risk_adjusted_return"}
    assert torch.isclose(objective, components["risk_adjusted_return"])


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
