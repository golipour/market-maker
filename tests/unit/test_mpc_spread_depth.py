from __future__ import annotations

import math

import torch

from mm.mpc.mpc_tensor_batched import (
    MonteCarloMPCTensorBatched,
    _action_spread_at_depth,
    _desired_quote_size_per_side,
    soft_fill_batched,
)
from mm.mpc.mpc_state import SigmoidMixtureAction
from mm.mpc.reward_tensor import terminal_objective
from mm.mpc.state_tensor import depth_bid_tensor


def test_quote_size_projection_applies_target_then_range() -> None:
    assert _desired_quote_size_per_side(
        5.0,
        quote_size_target_per_side=3.0,
        hard_project_quote_size_target=True,
        quote_size_min_per_side=0.1,
        quote_size_max_per_side=10.0,
        hard_project_quote_size_range=True,
    ) == 3.0

    assert _desired_quote_size_per_side(
        5.0,
        quote_size_target_per_side=15.0,
        hard_project_quote_size_target=True,
        quote_size_min_per_side=0.1,
        quote_size_max_per_side=10.0,
        hard_project_quote_size_range=True,
    ) == 10.0

    assert _desired_quote_size_per_side(
        15.0,
        quote_size_target_per_side=None,
        hard_project_quote_size_target=False,
        quote_size_min_per_side=0.1,
        quote_size_max_per_side=10.0,
        hard_project_quote_size_range=True,
    ) == 10.0

    assert _desired_quote_size_per_side(
        5.0,
        quote_size_target_per_side=3.0,
        hard_project_quote_size_target=False,
        quote_size_min_per_side=0.1,
        quote_size_max_per_side=10.0,
        hard_project_quote_size_range=False,
    ) is None


def test_spread_depth_stored() -> None:
    opt = object()
    flow = object()
    exe = object()
    mpc = MonteCarloMPCTensorBatched(opt, flow, exe, spread_depth=12.5)
    assert mpc.spread_depth == 12.5


def test_clear_optimizer_names_map_to_legacy_attributes() -> None:
    opt = object()
    flow = object()
    exe = object()
    mpc = MonteCarloMPCTensorBatched(
        opt,
        flow,
        exe,
        expected_pnl_weight=2.0,
        pnl_volatility_penalty_weight=3.0,
        inventory_target_penalty_weight=4.0,
        target_inventory=5.0,
        quote_shape_uniformity_penalty_weight=6.0,
        quote_size_target_penalty_weight=7.0,
        quote_spread_penalty_weight=8.0,
        quote_depth_target_for_spread=9.0,
        quote_size_target_per_side=10.0,
        hard_project_quote_size_target=True,
        hard_project_quote_size_range=True,
        quote_size_min_per_side=1.0,
        quote_size_max_per_side=20.0,
        hedge_rollout_enabled=True,
    )
    assert mpc.expected_pnl_weight == 2.0
    assert mpc.reward_coef == 2.0
    assert mpc.pnl_volatility_penalty_weight == 3.0
    assert mpc.risk_coef == 3.0
    assert mpc.inventory_target_penalty_weight == 4.0
    assert mpc.inv_coef == 4.0
    assert mpc.target_inventory == 5.0
    assert mpc.target_inv == 5.0
    assert mpc.quote_shape_uniformity_penalty_weight == 6.0
    assert mpc.uniformity_coef == 6.0
    assert mpc.quote_size_target_penalty_weight == 7.0
    assert mpc.size_coef == 7.0
    assert mpc.quote_spread_penalty_weight == 8.0
    assert mpc.spread_coef == 8.0
    assert mpc.quote_depth_target_for_spread == 9.0
    assert mpc.spread_depth == 9.0
    assert mpc.quote_size_target_per_side == 10.0
    assert mpc.hard_project_quote_size_target is True
    assert mpc.hard_project_quote_size_range is True
    assert mpc.quote_size_min_per_side == 1.0
    assert mpc.quote_size_max_per_side == 20.0
    assert mpc.hedge_enabled is True


def test_terminal_objective_uses_clear_component_names() -> None:
    delta_w = torch.tensor([1.0, 3.0], dtype=torch.float64)
    inv_terminal = torch.tensor([2.0, 4.0], dtype=torch.float64)

    objective, components = terminal_objective(
        delta_w,
        inv_terminal,
        expected_pnl_weight=2.0,
        inventory_target_penalty_weight=0.5,
        target_inventory=2.0,
        pnl_volatility_penalty_weight=0.25,
        uniformity=torch.tensor(4.0, dtype=torch.float64),
        quote_shape_uniformity_penalty_weight=0.1,
        size=torch.tensor(5.0, dtype=torch.float64),
        quote_size_target_penalty_weight=0.2,
        spread=torch.tensor(6.0, dtype=torch.float64),
        quote_spread_penalty_weight=0.3,
    )

    assert set(components) == {
        "expected_pnl",
        "inventory_target_penalty",
        "pnl_volatility_penalty",
        "quote_shape_uniformity_penalty",
        "quote_size_target_penalty",
        "quote_spread_penalty",
    }
    assert torch.isclose(objective, sum(components.values()))


def test_hedge_enabled_explicit_overrides_hedge_coef() -> None:
    opt = object()
    flow = object()
    exe = object()
    on = MonteCarloMPCTensorBatched(opt, flow, exe, hedge_enabled=True, hedge_coef=0.0)
    assert on.hedge_enabled is True
    off = MonteCarloMPCTensorBatched(opt, flow, exe, hedge_enabled=False, hedge_coef=1.0)
    assert off.hedge_enabled is False


def test_hedge_enabled_defaults_from_hedge_coef() -> None:
    opt = object()
    flow = object()
    exe = object()
    assert MonteCarloMPCTensorBatched(opt, flow, exe, hedge_coef=0.0).hedge_enabled is False
    assert MonteCarloMPCTensorBatched(opt, flow, exe, hedge_coef=0.01).hedge_enabled is True


def test_effective_sigmoid_action_exposes_executable_depth() -> None:
    raw = SigmoidMixtureAction(
        bid_L=(100.0,),
        bid_k=(1.0,),
        bid_x0=(10.0,),
        ask_L=(100.0,),
        ask_k=(1.0,),
        ask_x0=(10.0,),
    )
    effective = raw.as_effective()

    assert effective.to_flat() == raw.to_flat()
    assert effective.depth_bid(0.0) == 0.0
    assert effective.depth_ask(0.0) == 0.0
    assert math.isclose(effective.depth_bid(12.0), raw.depth_bid(12.0) - raw.depth_bid(0.0))
    assert math.isclose(effective.raw_depth_bid(12.0), raw.depth_bid(12.0))


def test_spread_at_depth_uses_depth_minus_depth_at_zero() -> None:
    # One bid and one ask component with effective L=100, k=1, x0=10.
    # Raw D(0) is already positive, so using raw D(s) would place target depth
    # too close to zero. The correct target is raw D(s) - raw D(0).
    l_raw = math.log(math.exp(100.0) - 1.0)
    k_raw = math.log(math.exp(1.0) - 1.0)
    params = torch.tensor([l_raw, k_raw, 10.0, l_raw, k_raw, 10.0], dtype=torch.float64)

    target = 10.0
    spread = _action_spread_at_depth(
        params,
        n_components=1,
        spread_max=20.0,
        target_cumulative_qty=target,
        grid_points=400,
        softmin_temp=5.0,
    )

    raw0 = float(depth_bid_tensor(params, torch.tensor([0.0], dtype=torch.float64), 1)[0])
    expected = 10.0 + math.log((target + raw0) / (100.0 - target - raw0))
    # The function returns bid spread + ask spread; both sides are symmetric.
    assert math.isclose(float(spread), 2.0 * expected, rel_tol=0.02)


def test_soft_fill_uses_baseline_shifted_marginal_quote_depth() -> None:
    l_raw = math.log(math.exp(100.0) - 1.0)
    k_raw = math.log(math.exp(1.0) - 1.0)
    params = torch.tensor([l_raw, k_raw, 10.0, l_raw, k_raw, 10.0], dtype=torch.float64)
    spread_grid = torch.tensor([1.0, 2.0], dtype=torch.float64)
    empty_book = torch.zeros((1, 2), dtype=torch.float64)
    sell_vwap = torch.tensor([[100.0]], dtype=torch.float64)
    buy_vwap = torch.tensor([[100.0]], dtype=torch.float64)
    sell_vol = torch.tensor([[1_000.0]], dtype=torch.float64)
    buy_vol = torch.tensor([[1_000.0]], dtype=torch.float64)

    _, _, _, inv_delta = soft_fill_batched(
        params,
        n_components=1,
        sell_vwap=sell_vwap,
        buy_vwap=buy_vwap,
        book_cum_bid=empty_book,
        book_cum_ask=empty_book,
        sell_vol=sell_vol,
        buy_vol=buy_vol,
        spread_grid=spread_grid,
        maker_fee=0.0,
        soft_temp=0.01,
    )

    raw = depth_bid_tensor(params, torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64), 1)
    expected_bid_quote = float(raw[2] - raw[0])
    raw_ask = raw
    expected_ask_quote = float(raw_ask[2] - raw_ask[0])
    # Buy and sell quote depths are symmetric, so fully consuming both sides should
    # leave near-zero inventory. If D(0) were included, both side fills would be
    # inflated by raw[0].
    assert math.isclose(expected_bid_quote, expected_ask_quote, rel_tol=1e-12)
    assert abs(float(inv_delta[0])) < 1e-6
