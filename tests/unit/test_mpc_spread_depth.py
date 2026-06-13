from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mm.mpc.mpc_tensor_batched import (
    MonteCarloMPCTensorBatched,
    VenueOptimizerParams,
    _quote_capacity_bounds_batched,
    _soft_fill_fraction,
    soft_fill_batched,
)
from mm.mpc.mpc_state import OrderbookRelative, SigmoidMixtureAction, State
from mm.mpc.reward_tensor import terminal_objective
from mm.mpc.state_tensor import depth_bid_tensor


class _FixedFlowModel:
    def sample(self, sample_shape, device=None, dtype=None):
        mod = torch
        return (
            mod.full(sample_shape, 1.0, dtype=dtype or mod.float64, device=device),
            mod.full(sample_shape, 1.0, dtype=dtype or mod.float64, device=device),
        )


class _FixedPriceModel:
    def sample_residual(self, sample_shape, device=None, dtype=None):
        mod = torch
        z = mod.zeros(sample_shape, dtype=dtype or mod.float64, device=device)
        return z, z


def test_optimizer_stores_objective_params() -> None:
    opt = object()
    flow = object()
    exe = object()
    mpc = MonteCarloMPCTensorBatched(
        opt,
        flow,
        exe,
        return_risk_aversion_beta=2.0,
        quote_size_min_per_side=1.0,
        quote_size_max_per_side=10.0,
        quote_imbalance_skew_strength=0.4,
    )
    assert mpc.return_risk_aversion_beta == 2.0
    assert mpc.quote_size_min_per_side == 1.0
    assert mpc.quote_size_max_per_side == 10.0
    assert mpc.quote_imbalance_skew_strength == 0.4
    assert mpc.soft_fill_queue_model == "back_of_queue"


def test_optimizer_passes_risk_aversion_beta_to_objective(monkeypatch) -> None:
    seen: list[float] = []

    def fake_terminal_objective(returns, *, return_risk_aversion_beta):
        seen.append(float(return_risk_aversion_beta))
        objective = returns.mean()
        return objective, {"risk_adjusted_return": objective}

    monkeypatch.setattr("mm.mpc.mpc_tensor_batched.terminal_objective", fake_terminal_objective)
    mpc = MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        _FixedFlowModel(),
        object(),
        n_components=1,
        n_samples=1,
        horizon=1,
        spread_max=2.0,
        spread_step=1.0,
        adam_epochs=1,
        fixed_mc_samples=True,
        return_risk_aversion_beta=7.5,
        quote_size_min_per_side=1.0,
    )

    mpc.optimize(
        State(
            mid=100.0,
            ob_rel=OrderbookRelative(bids=((1.0, 2.0),), asks=((1.0, 2.0),)),
            buy_vol=1.0,
            sell_vol=1.0,
            inventory=1.0,
            cash=1000.0,
            buy_vwap=99.0,
            sell_vwap=101.0,
        )
    )

    assert seen
    assert all(value == pytest.approx(7.5) for value in seen)


def test_terminal_objective_uses_clear_component_names() -> None:
    returns = torch.tensor([0.01, 0.03], dtype=torch.float64)

    objective, components = terminal_objective(
        returns,
        return_risk_aversion_beta=5.0,
    )

    assert set(components) == {"risk_adjusted_return"}
    assert torch.isclose(objective, components["risk_adjusted_return"])


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
    from mm.mpc.l_param import LParamConfig
    from mm.mpc.mpc_tensor_batched import _initial_action_uniform_marginal_qrange

    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=100.0,
        spread_max=100.0,
        n_components=1,
    )
    flat = _initial_action_uniform_marginal_qrange(1, 100.0, l_param=l_param)
    flat[-2] = 30.0
    flat[-1] = 30.0
    params = torch.tensor(flat, dtype=torch.float64)
    spread_grid = torch.tensor([1.0, 2.0], dtype=torch.float64)
    empty_book = torch.zeros((1, 2), dtype=torch.float64)
    mid_price = torch.tensor([[100.0]], dtype=torch.float64)
    sell_vol = torch.tensor([[1_000.0]], dtype=torch.float64)
    buy_vol = torch.tensor([[1_000.0]], dtype=torch.float64)

    _, _, _, inv_delta = soft_fill_batched(
        params,
        n_components=1,
        mid_price=mid_price,
        book_cum_bid=empty_book,
        book_cum_ask=empty_book,
        sell_vol=sell_vol,
        buy_vol=buy_vol,
        spread_grid=spread_grid,
        maker_fee=0.0,
        soft_temp=0.01,
        l_param=l_param,
    )

    from mm.mpc.state_tensor import effective_depth_ask_tensor, effective_depth_bid_tensor

    s_grid = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    bid_eff = effective_depth_bid_tensor(params, s_grid, 1, l_param=l_param)
    ask_eff = effective_depth_ask_tensor(params, s_grid, 1, l_param=l_param)
    expected_bid_quote = float(bid_eff[2] - bid_eff[0])
    expected_ask_quote = float(ask_eff[2] - ask_eff[0])
    assert math.isclose(expected_bid_quote, expected_ask_quote, rel_tol=1e-12)
    assert abs(float(inv_delta[0])) < 1e-6


def test_spread_structured_decoder_keeps_collapsed_center_depth_bounded() -> None:
    from mm.mpc.l_param import (
        LParamConfig,
        action_param_len,
        effective_depth_at_center_from_flat_np,
        effective_qrange_from_flat_np,
    )
    from mm.mpc.state_tensor import effective_depth_bid_tensor

    n_components = 4
    l_param = LParamConfig(
        quote_size_min=0.3,
        quote_size_max=10.0,
        spread_max=200.0,
        n_components=n_components,
        quote_spread_max=100.0,
        quote_spread_min_depth=0.3,
        quote_spread_k_epsilon=0.01,
    )
    flat = np.zeros(action_param_len(n_components), dtype=np.float64)
    flat[0] = -30.0
    flat[2] = 30.0
    flat[3 * n_components + 2] = -30.0
    flat[6 * n_components] = 30.0
    flat[6 * n_components + 1] = 30.0

    np_qrange = effective_qrange_from_flat_np(
        flat,
        n_components,
        side_off=0,
        spread_max=200.0,
        l_param=l_param,
    )
    np_center_depth = effective_depth_at_center_from_flat_np(
        flat,
        n_components,
        side_off=0,
        l_param=l_param,
    )
    torch_qrange = effective_depth_bid_tensor(
        torch.tensor(flat, dtype=torch.float64),
        torch.tensor([200.0], dtype=torch.float64),
        n_components,
        l_param=l_param,
    )[0]

    assert np_qrange == pytest.approx(10.0, rel=1e-9)
    assert np_center_depth == pytest.approx(0.3, rel=1e-9)
    assert float(torch_qrange) == pytest.approx(10.0, rel=1e-9)


def test_spread_grid_clamps_final_edge_to_spread_max() -> None:
    from mm.mpc.mpc_tensor_batched import _spread_grid_tensor

    grid = _spread_grid_tensor(
        torch,
        grid_levels=4,
        spread_step=30.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
        spread_max=100.0,
    )

    assert grid.tolist() == pytest.approx([30.0, 60.0, 90.0, 100.0])


def test_soft_fill_cash_prices_use_clean_mid_spreads_not_vwap_anchor() -> None:
    from mm.mpc.l_param import LParamConfig

    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=1.0,
        spread_max=1.0,
        n_components=1,
    )
    params = torch.tensor([0.0, 20.0, 0.0, 0.0, 20.0, 0.0, 30.0, 30.0], dtype=torch.float64)
    spread_grid = torch.tensor([1.0], dtype=torch.float64)
    empty_book = torch.zeros((1, 1), dtype=torch.float64)
    mid_price = torch.tensor([[100.0]], dtype=torch.float64)
    sell_vol = torch.tensor([[10.0]], dtype=torch.float64)
    buy_vol = torch.tensor([[0.0]], dtype=torch.float64)

    _, _, cash_delta, inv_delta = soft_fill_batched(
        params,
        n_components=1,
        mid_price=mid_price,
        book_cum_bid=empty_book,
        book_cum_ask=empty_book,
        sell_vol=sell_vol,
        buy_vol=buy_vol,
        spread_grid=spread_grid,
        maker_fee=0.0,
        soft_temp=0.01,
        soft_fill_activation="hardsigmoid",
        l_param=l_param,
    )

    assert float(inv_delta[0]) == pytest.approx(1.0, rel=1e-6)
    assert float(cash_delta[0]) == pytest.approx(-99.0, rel=1e-6)


def test_soft_fill_queue_model_back_of_queue_reduces_own_fill_share_and_keeps_gradients() -> None:
    from mm.mpc.l_param import LParamConfig

    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=1.0,
        spread_max=2.0,
        n_components=1,
    )
    params = torch.tensor(
        [0.0, 20.0, 0.0, 0.0, 20.0, 0.0, 30.0, 30.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    spread_grid = torch.tensor([1.0, 2.0], dtype=torch.float64)
    book_cum_bid = torch.tensor([[1.0, 10.0]], dtype=torch.float64)
    book_cum_ask = torch.zeros((1, 2), dtype=torch.float64)
    mid_price = torch.tensor([[100.0]], dtype=torch.float64)
    sell_vol = torch.tensor([[2.0]], dtype=torch.float64)
    buy_vol = torch.tensor([[0.0]], dtype=torch.float64)

    _, _, _, inv_pro_rata = soft_fill_batched(
        params,
        n_components=1,
        mid_price=mid_price,
        book_cum_bid=book_cum_bid,
        book_cum_ask=book_cum_ask,
        sell_vol=sell_vol,
        buy_vol=buy_vol,
        spread_grid=spread_grid,
        maker_fee=0.0,
        soft_temp=0.01,
        soft_fill_activation="hardsigmoid",
        soft_fill_queue_model="pro_rata",
        l_param=l_param,
    )
    _, _, _, inv_queue_back = soft_fill_batched(
        params,
        n_components=1,
        mid_price=mid_price,
        book_cum_bid=book_cum_bid,
        book_cum_ask=book_cum_ask,
        sell_vol=sell_vol,
        buy_vol=buy_vol,
        spread_grid=spread_grid,
        maker_fee=0.0,
        soft_temp=0.01,
        soft_fill_activation="hardsigmoid",
        soft_fill_queue_model="back_of_queue",
        l_param=l_param,
    )

    assert float(inv_queue_back.detach()[0]) < float(inv_pro_rata.detach()[0])
    inv_queue_back.sum().backward()
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


def test_soft_fill_dynamic_decoder_max_limits_simulated_fill_qty() -> None:
    from mm.mpc.l_param import LParamConfig

    l_param = LParamConfig(
        quote_size_min=0.1,
        quote_size_max=10.0,
        spread_max=2.0,
        n_components=1,
    )
    params = torch.tensor([0.0, 20.0, 0.0, 0.0, 20.0, 0.0, 30.0, 30.0], dtype=torch.float64)
    spread_grid = torch.tensor([1.0, 2.0], dtype=torch.float64)
    empty_book = torch.zeros((1, 2), dtype=torch.float64)
    mid_price = torch.tensor([[100.0]], dtype=torch.float64)

    _, _, _, inv_delta = soft_fill_batched(
        params,
        n_components=1,
        mid_price=mid_price,
        book_cum_bid=empty_book,
        book_cum_ask=empty_book,
        sell_vol=torch.tensor([[100.0]], dtype=torch.float64),
        buy_vol=torch.tensor([[100.0]], dtype=torch.float64),
        spread_grid=spread_grid,
        maker_fee=0.0,
        soft_temp=0.01,
        soft_fill_activation="hardsigmoid",
        bid_quote_size_max=torch.tensor([[0.4]], dtype=torch.float64),
        ask_quote_size_max=torch.tensor([[0.1]], dtype=torch.float64),
        l_param=l_param,
    )

    assert float(inv_delta[0]) == pytest.approx(0.3, rel=1e-6)


def test_quote_capacity_bounds_shrink_to_capacity_and_skew_correcting_side() -> None:
    optimizer = VenueOptimizerParams(
        quote_size_min_per_side=0.3,
        quote_size_max_per_side=None,
        quote_imbalance_skew_strength=1.0,
    )
    cash = torch.tensor([[20.0], [100.0], [20.0]], dtype=torch.float64)
    inventory = torch.tensor([[0.2], [0.5], [2.0]], dtype=torch.float64)
    mid = torch.full_like(cash, 100.0)

    bid_min, bid_max, ask_min, ask_max = _quote_capacity_bounds_batched(
        cash=cash,
        inventory=inventory,
        mid_price=mid,
        optimizer=optimizer,
    )

    assert bid_max[:, 0].tolist() == pytest.approx([0.2, 1.0, 0.2])
    assert ask_max[:, 0].tolist() == pytest.approx([0.2, 0.5, 2.0])
    assert bid_min[:, 0].tolist() == pytest.approx([0.2, 0.4, 0.2])
    assert ask_min[:, 0].tolist() == pytest.approx([0.2, 0.3, 0.5454545455])


def test_quote_capacity_bounds_use_wallet_totals_not_free_only() -> None:
    """Bid/ask caps use full wallet totals passed into the optimizer state."""
    from mm.core.models.private import Balance, PrivateStateSnapshot
    from mm.core.wallet import inventory_cash_from_snapshot

    snapshot = PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=1,
        balances={
            "USDT": Balance(
                exchange="nobitex",
                asset="USDT",
                free=20.0,
                locked=80.0,
                total=100.0,
                timestamp_ms=1,
            ),
            "BTC": Balance(
                exchange="nobitex",
                asset="BTC",
                free=1.0,
                locked=4.0,
                total=5.0,
                timestamp_ms=1,
            ),
        },
    )
    inventory, cash = inventory_cash_from_snapshot(snapshot, "BTC", "USDT")
    assert inventory == pytest.approx(5.0)
    assert cash == pytest.approx(100.0)

    optimizer = VenueOptimizerParams(
        quote_size_min_per_side=0.0,
        quote_size_max_per_side=None,
    )
    bid_min, bid_max, ask_min, ask_max = _quote_capacity_bounds_batched(
        cash=torch.tensor([[cash]], dtype=torch.float64),
        inventory=torch.tensor([[inventory]], dtype=torch.float64),
        mid_price=torch.tensor([[100.0]], dtype=torch.float64),
        optimizer=optimizer,
    )
    assert float(bid_max[0, 0]) == pytest.approx(1.0)
    assert float(ask_max[0, 0]) == pytest.approx(5.0)
    assert float(bid_min[0, 0]) == pytest.approx(0.0)
    assert float(ask_min[0, 0]) == pytest.approx(0.0)


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
        soft_fill_queue_model="pro_rata",
        quote_size_min_per_side=1.0,
        quote_size_max_per_side=10.0,
    )
    assert mpc.soft_fill_activation == "hardsigmoid"
    assert mpc.soft_fill_queue_model == "pro_rata"

    with pytest.raises(ValueError, match="unsupported soft_fill_activation"):
        MonteCarloMPCTensorBatched(
            price, flow, exe, soft_fill_activation="relu",
            quote_size_min_per_side=1.0, quote_size_max_per_side=10.0,
        )
    with pytest.raises(ValueError, match="unsupported soft_fill_queue_model"):
        MonteCarloMPCTensorBatched(
            price, flow, exe, soft_fill_queue_model="middle_of_queue",
            quote_size_min_per_side=1.0, quote_size_max_per_side=10.0,
        )
