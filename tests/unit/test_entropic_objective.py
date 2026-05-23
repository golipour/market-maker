from __future__ import annotations

import torch

from mm.mpc.reward_tensor import (
    objective_entropic_return,
    objective_inventory_capacity_penalty,
    terminal_objective,
)


def test_entropic_return_prefers_better_tail() -> None:
    good = torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.float64)
    bad = torch.tensor([0.01, 0.02, -0.20, 0.04], dtype=torch.float64)
    score_good = float(objective_entropic_return(good, beta=20.0).item())
    score_bad = float(objective_entropic_return(bad, beta=20.0).item())
    assert score_good > score_bad


def test_inventory_capacity_penalty_is_quadratic_and_normalized() -> None:
    inv = torch.tensor([[0.0, 2.0], [4.0, 0.0]], dtype=torch.float64)
    penalty = float(
        objective_inventory_capacity_penalty(
            inv,
            target_inventory=2.0,
            inventory_scale=2.0,
        ).item()
    )
    assert penalty == 0.75


def test_terminal_objective_has_two_components() -> None:
    returns = torch.tensor([0.01, 0.02, -0.05, 0.03], dtype=torch.float64)
    inv_path = torch.tensor([[1.0], [2.0], [0.0], [3.0]], dtype=torch.float64)
    objective, components = terminal_objective(
        returns,
        inv_path,
        return_risk_aversion_beta=10.0,
        inventory_target_penalty_weight=0.5,
        target_inventory=2.0,
        inventory_scale=2.0,
    )
    assert set(components) == {"risk_adjusted_return", "inventory_capacity_penalty"}
    assert components["inventory_capacity_penalty"] <= 0.0
    assert torch.isclose(objective, sum(components.values()))


def test_entropic_return_gradients_flow() -> None:
    returns = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
    score = objective_entropic_return(returns, beta=5.0)
    score.backward()
    assert returns.grad is not None
    assert torch.isfinite(returns.grad).all()
