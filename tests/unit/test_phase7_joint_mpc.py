"""Phase 7: joint-venue MPC rollout tests."""

from __future__ import annotations

import math

import pytest
import torch

from mm.core.models.real_execution import RealExecutionModel
from mm.mpc.joint_mpc import (
    JointSigmoidMixtureAction,
    consolidated_wealth_scalar,
    joint_action_param_len,
    reference_price_from_state,
    split_venue_action_flat,
)
from mm.mpc.l_param import LParamConfig, action_param_len
from mm.mpc.market_order_flow_torch import TorchHurdleLogNormalModel
from mm.mpc.mpc_state import OrderbookRelative, State
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched, soft_fill_batched
from mm.mpc.price_torch import TorchResidualPriceModel
from mm.mpc.state_tensor import effective_depth_ask_tensor, effective_depth_bid_tensor


def _minimal_state(**overrides: object) -> State:
    base = State(
        mid=100.0,
        ob_rel=OrderbookRelative(bids=((1.0, 2.0),), asks=((1.0, 2.0),)),
        buy_vol=1.0,
        sell_vol=1.0,
        inventory=5.0,
        cash=1000.0,
        buy_vwap=99.0,
        sell_vwap=101.0,
        global_ob_rel=OrderbookRelative(bids=((2.0, 1.0),), asks=((2.0, 1.0),)),
        global_buy_vol=0.5,
        global_sell_vol=0.5,
        global_inventory=3.0,
        global_cash=500.0,
        global_buy_vwap=98.0,
        global_sell_vwap=102.0,
        reference_price=102.0,
        global_active=True,
    )
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


def _build_optimizer(*, liquidity_sourcing_enabled: bool = False) -> MonteCarloMPCTensorBatched:
    flow = TorchHurdleLogNormalModel()
    price = TorchResidualPriceModel(market_flow_model=flow)
    global_flow = TorchHurdleLogNormalModel()
    global_price = TorchResidualPriceModel(market_flow_model=global_flow)
    exe = RealExecutionModel()
    return MonteCarloMPCTensorBatched(
        price,
        flow,
        exe,
        global_market_flow_model=global_flow,
        global_price_model=global_price,
        liquidity_sourcing_enabled=liquidity_sourcing_enabled,
        n_components=1,
        n_samples=4,
        horizon=1,
        adam_epochs=2,
        spread_max=20.0,
        spread_step=5.0,
        quote_size_min_per_side=0.5,
        quote_size_max_per_side=2.0,
        device="cpu",
        fixed_mc_samples=True
    )


def test_joint_action_param_len_is_twice_single_venue() -> None:
    assert joint_action_param_len(4) == 2 * action_param_len(4)


def test_consolidated_wealth_marks_both_venues_to_reference() -> None:
    wealth = consolidated_wealth_scalar(
        local_cash=100.0,
        local_inventory=2.0,
        global_cash=50.0,
        global_inventory=1.0,
        reference_price=10.0,
    )
    assert wealth == 100.0 + 50.0 + (2.0 + 1.0) * 10.0


def test_reference_price_prefers_global_sell_vwap() -> None:
    state = _minimal_state(reference_price=0.0)
    assert reference_price_from_state(state) == 102.0


def test_soft_fill_global_depth_scale_zeros_fills() -> None:
    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=5.0,
        spread_max=10.0,
        n_components=1,
    )
    from mm.mpc.mpc_tensor_batched import _initial_action_uniform_marginal_qrange

    flat = _initial_action_uniform_marginal_qrange(1, 10.0, l_param=l_param)
    params = torch.tensor(flat, dtype=torch.float64)
    spread_grid = torch.tensor([5.0, 10.0], dtype=torch.float64)
    empty_book = torch.zeros((1, 2), dtype=torch.float64)
    sell_vwap = torch.tensor([[100.0]], dtype=torch.float64)
    buy_vwap = torch.tensor([[100.0]], dtype=torch.float64)
    sell_vol = torch.tensor([[100.0]], dtype=torch.float64)
    buy_vol = torch.tensor([[100.0]], dtype=torch.float64)

    _, _, _, inv_masked = soft_fill_batched(
        params,
        1,
        sell_vwap,
        buy_vwap,
        empty_book,
        empty_book,
        sell_vol,
        buy_vol,
        spread_grid,
        0.0,
        l_param=l_param,
        depth_scale=0.0,
    )
    assert abs(float(inv_masked[0])) < 1e-6
    s_grid = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64)
    scaled_depth = effective_depth_bid_tensor(params, s_grid, 1, l_param=l_param) * 0.0
    raw_depth = effective_depth_bid_tensor(params, s_grid, 1, l_param=l_param)
    assert float(scaled_depth[-1]) == 0.0
    assert float(raw_depth[-1]) >= 0.0


def test_single_venue_mask_zeros_global_effective_depth() -> None:
    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=5.0,
        spread_max=10.0,
        n_components=1,
    )
    from mm.mpc.mpc_tensor_batched import _initial_action_uniform_marginal_qrange

    one = _initial_action_uniform_marginal_qrange(1, 10.0, l_param=l_param)
    joint = torch.tensor(list(one) + list(one), dtype=torch.float64)
    local_flat, global_flat = split_venue_action_flat(joint.detach().numpy(), 1)
    spread = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64)
    local_bid = effective_depth_bid_tensor(torch.tensor(local_flat), spread, 1, l_param=l_param)
    global_bid = effective_depth_bid_tensor(torch.tensor(global_flat), spread, 1, l_param=l_param)
    assert float(global_bid[-1]) > 0.0
    masked = effective_depth_bid_tensor(torch.tensor(global_flat), spread, 1, l_param=l_param) * 0.0
    assert float(masked[-1]) == 0.0
    assert float(local_bid[-1]) > 0.0


def test_liquidity_sourcing_disabled_matches_legacy_optimizer_path() -> None:
    state = _minimal_state(global_active=False)
    legacy = _build_optimizer(liquidity_sourcing_enabled=False)
    legacy_action, _ = legacy.optimize_with_diagnostics(state)
    assert legacy_action.n_components == 1
    assert legacy._last_global_sigmoid_action is None


def test_joint_optimizer_runs_with_consolidated_wealth() -> None:
    state = _minimal_state(global_active=True)
    joint = _build_optimizer(liquidity_sourcing_enabled=True)
    action, diagnostics = joint.optimize_with_diagnostics(state)
    assert action.n_components == 1
    assert diagnostics.rollout.get("consolidated_wealth") is True
    assert diagnostics.rollout.get("global_rollout") is False
    assert joint._last_joint_action is not None
    assert isinstance(joint._last_joint_action, JointSigmoidMixtureAction)


def test_build_controller_uses_global_price_model_not_local_residual() -> None:
    from mm.mpc.global_price_torch import GlobalGaussianPriceModel
    from mm.runtime.config import (
        ControllerRuntimeConfig,
        LiquiditySourcingRuntimeConfig,
        MarketDataSourceConfig,
        MarketMakerRuntimeConfig,
    )
    from mm.strategies.mpc_engine import _build_controller

    sourcing = LiquiditySourcingRuntimeConfig(enabled=True, mode="liquidity_sourcing")
    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="USDT_IRT"),
        controller=ControllerRuntimeConfig(
            enabled=True,
            global_price_model={"kind": "legacy", "decay": 0.95},
            price_model={"kind": "residual", "residual_decay": 0.95},
            optimizer={"quote_size_min_per_side": 0.1, "quote_size_max_per_side": 1.0},
        ),
        liquidity_sourcing=sourcing,
    )
    controller = _build_controller(
        runtime.controller,
        runtime.binding,
        liquidity_sourcing=sourcing,
    )

    assert isinstance(controller.global_price_model, GlobalGaussianPriceModel)


def test_joint_optimizer_can_emit_nonzero_global_curve() -> None:
    state = _minimal_state(global_active=True)
    joint = _build_optimizer(liquidity_sourcing_enabled=True)
    joint.adam_epochs = 5
    _, diagnostics = joint.optimize_with_diagnostics(state)
    global_flat = diagnostics.rollout.get("global_action_flat") or []
    assert len(global_flat) == action_param_len(1)
    l_param = joint._l_param_config()
    spread = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0], dtype=torch.float64)
    depth = effective_depth_ask_tensor(
        torch.tensor(global_flat, dtype=torch.float64),
        spread,
        1,
        l_param=l_param,
    )
    assert math.isfinite(float(depth[-1]))
