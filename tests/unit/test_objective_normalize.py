from __future__ import annotations

import math

import torch

from mm.mpc.mpc_tensor_batched import _action_size_penalty_qrange, _action_spread_at_depth
from mm.mpc.objective_normalize import inventory_scale, wealth_scale
from mm.mpc.reward_tensor import terminal_objective


def test_wealth_and_inventory_scales() -> None:
    assert wealth_scale(1_000_000.0, 10.0, 50_000.0) == 1_000_000.0 + 500_000.0
    assert inventory_scale(10.0, 3.0) == 10.0
    assert inventory_scale(0.0, None) == 1.0


def test_size_penalty_is_dimensionless_when_normalized() -> None:
    l_raw = math.log(math.exp(100.0) - 1.0)
    k_raw = math.log(math.exp(1.0) - 1.0)
    params = torch.tensor([l_raw, k_raw, 50.0, l_raw, k_raw, 50.0], dtype=torch.float64)
    target = 10.0
    raw = float(
        _action_size_penalty_qrange(
            params, 1, 100.0, target, normalize=False
        ).item()
    )
    norm = float(
        _action_size_penalty_qrange(
            params, 1, 100.0, target, normalize=True
        ).item()
    )
    assert raw > 0.0
    assert norm >= 0.0
    assert norm <= raw / (target * target) + 1e-6 or norm < raw


def test_terminal_objective_pnl_scales_with_wealth() -> None:
    delta_w = torch.tensor([1000.0, 2000.0], dtype=torch.float64)
    inv = torch.tensor([0.0, 0.0], dtype=torch.float64)
    _, comp_raw = terminal_objective(
        delta_w,
        inv,
        expected_pnl_weight=1.0,
        wealth_scale=1.0,
        normalize_components=False,
    )
    _, comp_norm = terminal_objective(
        delta_w,
        inv,
        expected_pnl_weight=1.0,
        wealth_scale=1000.0,
        normalize_components=True,
    )
    assert comp_norm["expected_pnl"] == comp_raw["expected_pnl"] / 1000.0


def test_spread_penalty_zero_at_target_spread() -> None:
    """Symmetric relative spread penalty is ~0 when soft-argmin spread equals target."""
    l_raw = math.log(math.exp(50.0) - 1.0)
    k_raw = math.log(math.exp(2.0) - 1.0)
    x0 = 100.0
    params = torch.tensor([l_raw, k_raw, x0, l_raw, k_raw, x0], dtype=torch.float64)
    target_spread = 100.0
    penalty = float(
        _action_spread_at_depth(
            params,
            1,
            spread_max=200.0,
            target_cumulative_qty=10.0,
            quote_spread_target=target_spread,
            normalize=True,
            grid_points=400,
        ).item()
    )
    assert penalty >= 0.0
    assert math.isfinite(penalty)
