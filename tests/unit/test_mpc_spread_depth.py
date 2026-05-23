from __future__ import annotations

import math

import pytest
import torch

from mm.mpc.mpc_tensor_batched import (
    MonteCarloMPCTensorBatched,
    _soft_fill_fraction,
    soft_fill_batched,
)
from mm.mpc.mpc_state import SigmoidMixtureAction
from mm.mpc.reward_tensor import terminal_objective
from mm.mpc.state_tensor import depth_bid_tensor


def test_optimizer_stores_objective_params() -> None:
    opt = object()
    flow = object()
    exe = object()
    mpc = MonteCarloMPCTensorBatched(
        opt,
        flow,
        exe,
        return_risk_aversion_beta=2.0,
        inventory_target_penalty_weight=4.0,
        target_inventory=5.0,
        target_inventory_multiplier=1.5,
        quote_size_target_per_side=10.0,
        hedge_rollout_enabled=True,
    )
    assert mpc.return_risk_aversion_beta == 2.0
    assert mpc.inventory_target_penalty_weight == 4.0
    assert mpc.target_inventory == 5.0
    assert mpc.target_inventory_multiplier == 1.5
    assert mpc.quote_size_target_per_side == 10.0
    assert mpc.inv_coef == 4.0
    assert mpc.target_inv == 5.0
    assert mpc.hedge_enabled is True


def test_resolve_inventory_target_from_quote_size() -> None:
    mpc = MonteCarloMPCTensorBatched(
        object(),
        object(),
        object(),
        quote_size_target_per_side=100.0,
        target_inventory_multiplier=1.5,
    )
    assert mpc._resolve_inventory_target() == 150.0
    assert mpc._resolve_inventory_scale() == 150.0


def test_terminal_objective_uses_clear_component_names() -> None:
    returns = torch.tensor([0.01, 0.03], dtype=torch.float64)
    inv_path = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

    objective, components = terminal_objective(
        returns,
        inv_path,
        return_risk_aversion_beta=5.0,
        inventory_target_penalty_weight=0.5,
        target_inventory=2.0,
        inventory_scale=2.0,
    )

    assert set(components) == {"risk_adjusted_return", "inventory_capacity_penalty"}
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
    assert math.isclose(expected_bid_quote, expected_ask_quote, rel_tol=1e-12)
    assert abs(float(inv_delta[0])) < 1e-6


def test_soft_fill_fraction_hardsigmoid_truncates_negative_reach() -> None:
    reach = torch.tensor([[-4.0, -1.0, 0.0, 1.0, 4.0]], dtype=torch.float64)
    hard = _soft_fill_fraction(reach, soft_temp=1.0, activation="hardsigmoid")
    soft = _soft_fill_fraction(reach, soft_temp=1.0, activation="sigmoid")

    assert hard[0, 0].item() == 0.0
    assert soft[0, 0].item() > 0.0
    assert hard[0, -1].item() == 1.0


def test_soft_fill_hardsigmoid_reduces_unreachable_level_fill_vs_sigmoid() -> None:
    reach = torch.tensor([[-4.0, -3.0, 4.0]], dtype=torch.float64)
    soft_temp = 1.0
    hard = _soft_fill_fraction(reach, soft_temp, activation="hardsigmoid")
    soft = _soft_fill_fraction(reach, soft_temp, activation="sigmoid")

    assert hard[0, 0].item() == 0.0
    assert soft[0, 0].item() > 0.0
    assert hard[0, 1].item() == 0.0
    assert soft[0, 1].item() > 0.0
    assert hard[0, 2].item() == 1.0


def test_mpc_optimizer_accepts_soft_fill_activation_config() -> None:
    from mm.core.models.real_execution import RealExecutionModel
    from mm.mpc.market_order_flow_torch import TorchHurdleLogNormalModel
    from mm.mpc.price_torch import TorchResidualPriceModel

    flow = TorchHurdleLogNormalModel()
    price = TorchResidualPriceModel(market_flow_model=flow)
    exe = RealExecutionModel()
    mpc = MonteCarloMPCTensorBatched(
        price,
        flow,
        exe,
        soft_fill_activation="hardsigmoid",
    )
    assert mpc.soft_fill_activation == "hardsigmoid"

    with pytest.raises(ValueError, match="unsupported soft_fill_activation"):
        MonteCarloMPCTensorBatched(price, flow, exe, soft_fill_activation="relu")
