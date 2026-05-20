from __future__ import annotations

import math

import pytest
import torch

from mm.mpc.mpc_tensor_batched import (
    _action_size_penalty_qrange,
    _action_spread_at_depth,
    _quoted_spread_bps_from_action,
)
from mm.mpc.objective_normalize import (
    inventory_scale,
    relative_spread_penalty_excess_bps,
    resolve_quote_spread_target_bps,
    wealth_scale,
)
from mm.mpc.reward_tensor import terminal_objective


def test_wealth_and_inventory_scales() -> None:
    assert wealth_scale(1_000_000.0, 10.0, 50_000.0) == 1_000_000.0 + 500_000.0
    assert inventory_scale(10.0, 3.0) == 10.0
    assert inventory_scale(0.0, None) == 1.0


def test_resolve_quote_spread_target_bps_defaults() -> None:
    assert resolve_quote_spread_target_bps(None) == 50.0
    assert resolve_quote_spread_target_bps(30.0) == 30.0


def test_spread_penalty_asymmetric_in_bps() -> None:
    tight = torch.tensor(40.0, dtype=torch.float64)
    wide = torch.tensor(75.0, dtype=torch.float64)
    tgt = 50.0
    assert float(relative_spread_penalty_excess_bps(tight, tgt).item()) == 0.0
    assert float(relative_spread_penalty_excess_bps(wide, tgt).item()) == pytest.approx(0.5, rel=1e-9)


def test_spread_penalty_normalized_uses_bps() -> None:
    params = torch.zeros(6, dtype=torch.float64)
    mid = 100_000.0
    spread_bps = float(
        _action_spread_at_depth(
            params,
            1,
            100.0,
            1.0,
            mid,
            quote_spread_target_bps=50.0,
            normalize=False,
            grid_points=8,
        ).item()
    )
    penalty = float(
        _action_spread_at_depth(
            params,
            1,
            100.0,
            1.0,
            mid,
            quote_spread_target_bps=50.0,
            normalize=True,
            grid_points=8,
        ).item()
    )
    assert spread_bps >= 0.0
    if spread_bps > 50.0:
        assert penalty == pytest.approx((spread_bps - 50.0) / 50.0, rel=1e-6)
    else:
        assert penalty == 0.0


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


def test_quoted_spread_bps_positive_with_mid() -> None:
    l_raw = math.log(math.exp(10.0) - 1.0)
    k_raw = math.log(math.exp(0.5) - 1.0)
    params = torch.tensor([l_raw, k_raw, 5.0, l_raw, k_raw, 5.0], dtype=torch.float64)
    _, _, spread_bps = _quoted_spread_bps_from_action(params, 1, 50.0, 3.0, 50_000.0)
    assert float(spread_bps.item()) >= 0.0
    assert math.isfinite(float(spread_bps.item()))
