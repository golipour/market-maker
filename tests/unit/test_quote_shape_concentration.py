from __future__ import annotations

import math

import torch

from mm.mpc.mpc_tensor_batched import _action_concentration_penalty_from_marginal_depth


def test_concentration_in_valid_range() -> None:
    """Herfindahl is in [1/n_levels, 1] per side; average of bid/ask in that range."""
    spread_ticks = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    l_raw = math.log(math.exp(50.0) - 1.0)
    k_raw = math.log(math.exp(20.0) - 1.0)
    params = torch.tensor([l_raw, k_raw, 2.0, l_raw, k_raw, 2.0], dtype=torch.float64)
    h = float(_action_concentration_penalty_from_marginal_depth(params, spread_ticks, 1))
    n_levels = spread_ticks.numel()
    assert 1.0 / n_levels <= h <= 1.0


def test_concentration_lower_when_spread_across_levels() -> None:
    spread_ticks = torch.linspace(0.0, 10.0, 11, dtype=torch.float64)
    n = 4
    # Uniform logits + spread x0 centers -> more distributed marginal depth.
    params = torch.zeros(3 * n, dtype=torch.float64)
    for i in range(n):
        params[3 * i + 2] = 2.0 + i * 2.0
    h_spread = float(
        _action_concentration_penalty_from_marginal_depth(params, spread_ticks, n)
    )
    # Concentrated: one big L at one x0.
    l_raw = math.log(math.exp(100.0) - 1.0)
    k_raw = math.log(math.exp(5.0) - 1.0)
    params_one = torch.tensor([l_raw, k_raw, 5.0], dtype=torch.float64)
    h_one = float(
        _action_concentration_penalty_from_marginal_depth(
            params_one, spread_ticks, 1
        )
    )
    assert h_spread < h_one
