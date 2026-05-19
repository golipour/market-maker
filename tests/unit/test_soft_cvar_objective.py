from __future__ import annotations

import torch

from mm.mpc.reward_tensor import (
    objective_soft_cvar,
    objective_soft_tail_mean,
    terminal_objective,
)


def test_soft_tail_mean_weights_low_rewards() -> None:
    rewards = torch.tensor([10.0, 10.0, -20.0, 10.0], dtype=torch.float64)
    tail = float(objective_soft_tail_mean(rewards, temperature=50.0).item())
    mean = float(rewards.mean().item())
    assert tail < mean


def test_soft_cvar_positive_when_tail_is_bad() -> None:
    rewards = torch.tensor([5.0, 5.0, -15.0, 5.0], dtype=torch.float64)
    cvar = float(objective_soft_cvar(rewards, temperature=30.0).item())
    assert cvar > 0.0


def test_terminal_objective_includes_cvar_component() -> None:
    rewards = torch.tensor([1.0, 2.0, -8.0, 3.0], dtype=torch.float64)
    inv = torch.zeros_like(rewards)
    objective, components = terminal_objective(
        rewards,
        inv,
        expected_pnl_weight=1.0,
        pnl_volatility_penalty_weight=0.0,
        pnl_cvar_penalty_weight=0.5,
        cvar_softmax_temperature=20.0,
    )
    assert "pnl_cvar_penalty" in components
    assert components["pnl_cvar_penalty"] <= 0.0
    assert torch.isclose(objective, sum(components.values()))


def test_soft_cvar_gradients_flow() -> None:
    rewards = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
    cvar = objective_soft_cvar(rewards, temperature=10.0)
    cvar.backward()
    assert rewards.grad is not None
    assert torch.isfinite(rewards.grad).all()
