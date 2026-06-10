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
        vwap_depth=5.0,
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


def _build_optimizer(
    *,
    liquidity_sourcing_enabled: bool = False,
    reference_price_source: str = "global",
    local_optimizer: dict | None = None,
    global_optimizer: dict | None = None,
) -> MonteCarloMPCTensorBatched:
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
        reference_price_source=reference_price_source,
        n_components=1,
        n_samples=4,
        horizon=1,
        adam_epochs=2,
        spread_max=20.0,
        spread_step=5.0,
        quote_size_min_per_side=0.5,
        quote_size_max_per_side=2.0,
        local_optimizer=local_optimizer,
        global_optimizer=global_optimizer,
        device="cpu",
        fixed_mc_samples=True
    )


def test_joint_action_param_len_is_twice_single_venue() -> None:
    assert joint_action_param_len(4) == 2 * action_param_len(4)


def test_joint_action_param_len_allows_asymmetric_venue_complexity() -> None:
    assert joint_action_param_len(4, 2) == action_param_len(4) + action_param_len(2)
    local_flat, global_flat = split_venue_action_flat(
        list(range(joint_action_param_len(4, 2))),
        4,
        2,
    )
    assert len(local_flat) == action_param_len(4)
    assert len(global_flat) == action_param_len(2)


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


def test_reference_price_can_select_local_sell_vwap() -> None:
    state = _minimal_state(reference_price=102.0)
    assert reference_price_from_state(state, "local") == 101.0
    assert reference_price_from_state(state, "global") == 102.0


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
    mid_price = torch.tensor([[100.0]], dtype=torch.float64)
    sell_vol = torch.tensor([[100.0]], dtype=torch.float64)
    buy_vol = torch.tensor([[100.0]], dtype=torch.float64)

    _, _, _, inv_masked = soft_fill_batched(
        params,
        1,
        mid_price,
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


def test_liquidity_sourcing_disabled_uses_single_venue_optimizer_path() -> None:
    state = _minimal_state(global_active=False)
    optimizer = _build_optimizer(
        liquidity_sourcing_enabled=False,
        reference_price_source="local",
    )
    action, _ = optimizer.optimize_with_diagnostics(state)
    assert action.n_components == 1
    assert optimizer._last_global_sigmoid_action is None


def test_global_reference_runs_joint_without_liquidity_sourcing() -> None:
    state = _minimal_state(global_active=False, global_inventory=0.0, global_cash=0.0)
    optimizer = _build_optimizer(
        liquidity_sourcing_enabled=False,
        reference_price_source="global",
    )
    _, diagnostics = optimizer.optimize_with_diagnostics(state)
    assert diagnostics.config.get("joint_venue") is True
    assert diagnostics.rollout.get("liquidity_sourcing") is False
    assert diagnostics.rollout.get("global_reference") is True
    assert optimizer._last_global_sigmoid_action is None


def test_inventory_vwap_depth_tracks_marked_inventory() -> None:
    from mm.strategies.mpc_engine import _inventory_vwap_depth

    assert _inventory_vwap_depth(
        5.0,
        7.0,
        liquidity_sourcing_enabled=False,
    ) == pytest.approx(5.0)
    assert _inventory_vwap_depth(
        5.0,
        7.0,
        liquidity_sourcing_enabled=True,
    ) == pytest.approx(12.0)
    assert _inventory_vwap_depth(
        -5.0,
        2.0,
        liquidity_sourcing_enabled=True,
    ) == pytest.approx(3.0)


def test_joint_optimizer_runs_with_consolidated_wealth() -> None:
    state = _minimal_state(global_active=True)
    joint = _build_optimizer(liquidity_sourcing_enabled=True)
    action, diagnostics = joint.optimize_with_diagnostics(state)
    assert action.n_components == 1
    assert diagnostics.rollout.get("consolidated_wealth") is True
    assert diagnostics.rollout.get("global_rollout") is False
    assert joint._last_joint_action is not None
    assert isinstance(joint._last_joint_action, JointSigmoidMixtureAction)


def test_joint_optimizer_uses_separate_local_and_global_maker_fees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mm.mpc.joint_optimizer as joint_optimizer

    state = _minimal_state(global_active=True)
    joint = _build_optimizer(liquidity_sourcing_enabled=True)
    joint.execution_model = RealExecutionModel(
        maker_fee=0.0,
        local_maker_fee=0.0002,
        global_maker_fee=0.0007,
    )
    fees: list[float] = []
    original_soft_fill = joint_optimizer.soft_fill_batched

    def capture_soft_fill(*args: object, **kwargs: object) -> object:
        fees.append(float(args[8]))
        return original_soft_fill(*args, **kwargs)

    monkeypatch.setattr(joint_optimizer, "soft_fill_batched", capture_soft_fill)

    joint.optimize_with_diagnostics(state)

    assert fees
    assert fees[0::2]
    assert fees[1::2]
    assert all(fee == pytest.approx(0.0002) for fee in fees[0::2])
    assert all(fee == pytest.approx(0.0007) for fee in fees[1::2])


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
        global_=MarketDataSourceConfig(role="global", exchange="wallex", symbol="USDT_IRT"),
        controller=ControllerRuntimeConfig(
            enabled=True,
            global_price_model={"kind": "ewma_gaussian", "decay": 0.95},
            local_price_model={"kind": "residual", "residual_decay": 0.95},
            optimizer={
                "local": {"quote_size_min_per_side": 0.1, "quote_size_max_per_side": 1.0},
                "global": {"quote_size_min_per_side": 0.1, "quote_size_max_per_side": 1.0},
            },
        ),
        liquidity_sourcing=sourcing,
    )
    controller = _build_controller(
        runtime.controller,
        runtime.binding,
        liquidity_sourcing=sourcing,
    )

    assert isinstance(controller.global_price_model, GlobalGaussianPriceModel)


def test_build_controller_uses_first_class_global_market_flow_model() -> None:
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
        global_=MarketDataSourceConfig(role="global", exchange="wallex", symbol="USDT_IRT"),
        controller=ControllerRuntimeConfig(
            enabled=True,
            local_market_flow_model={"kind": "hurdle_lognormal", "init_p0": 0.1},
            global_market_flow_model={"kind": "hurdle_lognormal", "init_p0": 0.9},
            optimizer={
                "local": {"quote_size_min_per_side": 0.1, "quote_size_max_per_side": 1.0},
                "global": {"quote_size_min_per_side": 0.1, "quote_size_max_per_side": 1.0},
            },
        ),
        liquidity_sourcing=sourcing,
    )

    controller = _build_controller(
        runtime.controller,
        runtime.binding,
        liquidity_sourcing=sourcing,
    )

    assert controller.local_market_flow_model.get_params()["p0_buy"] == pytest.approx(0.1)
    assert controller.global_market_flow_model.get_params()["p0_buy"] == pytest.approx(0.9)


def test_build_controller_keeps_global_models_for_global_reference_without_sourcing() -> None:
    from mm.mpc.global_price_torch import GlobalGaussianPriceModel
    from mm.runtime.config import (
        ControllerRuntimeConfig,
        LiquiditySourcingRuntimeConfig,
        MarketDataSourceConfig,
        MarketMakerRuntimeConfig,
    )
    from mm.strategies.mpc_engine import _build_controller

    sourcing = LiquiditySourcingRuntimeConfig(enabled=False)
    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="USDT_IRT"),
        global_=MarketDataSourceConfig(role="global", exchange="wallex", symbol="USDT_IRT"),
        controller=ControllerRuntimeConfig(
            enabled=True,
            reference_price_source="global",
            global_price_model={"kind": "ewma_gaussian", "decay": 0.95},
            optimizer={
                "local": {"quote_size_min_per_side": 0.1, "quote_size_max_per_side": 1.0},
                "global": {"quote_size_min_per_side": 0.1, "quote_size_max_per_side": 1.0},
            },
        ),
        liquidity_sourcing=sourcing,
    )

    controller = _build_controller(
        runtime.controller,
        runtime.binding,
        liquidity_sourcing=sourcing,
    )

    assert controller.liquidity_sourcing_enabled is False
    assert controller.global_market_flow_model is not None
    assert isinstance(controller.global_price_model, GlobalGaussianPriceModel)


def test_build_controller_uses_separate_local_global_optimizer_params() -> None:
    from mm.runtime.config import (
        ControllerRuntimeConfig,
        ExecutionRuntimeConfig,
        LiquiditySourcingRuntimeConfig,
        MarketDataSourceConfig,
        MarketMakerRuntimeConfig,
        VenueExecutionConfig,
    )
    from mm.strategies.mpc_engine import _build_controller

    sourcing = LiquiditySourcingRuntimeConfig(enabled=True, mode="liquidity_sourcing")
    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="USDT_IRT"),
        global_=MarketDataSourceConfig(role="global", exchange="wallex", symbol="USDT_IRT"),
        controller=ControllerRuntimeConfig(
            enabled=True,
            reference_price_source="local",
            optimizer={
                "local": {
                    "n_components": 3,
                    "objective_weight": 0.7,
                    "return_risk_aversion_beta": 0.2,
                    "quote_size_min_per_side": 0.2,
                    "quote_capacity_ratio": 0.8,
                    "quote_imbalance_skew_strength": 0.25,
                    "soft_fill_temp": 0.11,
                    "soft_fill_activation": "hardsigmoid",
                    "soft_fill_queue_model": "pro_rata",
                },
                "global": {
                    "n_components": 2,
                    "objective_weight": 0.4,
                    "return_risk_aversion_beta": 0.8,
                    "quote_size_min_per_side": 1.0,
                    "quote_size_max_per_side": 5.0,
                    "quote_capacity_ratio": 0.6,
                    "quote_imbalance_skew_strength": 0.5,
                    "soft_fill_temp": 0.33,
                    "soft_fill_activation": "sigmoid",
                    "soft_fill_queue_model": "back_of_queue",
                },
            },
        ),
        execution=ExecutionRuntimeConfig(
            local=VenueExecutionConfig(max_orders_per_side=4),
            global_=VenueExecutionConfig(planner="replace", max_orders_per_side=2),
        ),
        liquidity_sourcing=sourcing,
    )

    controller = _build_controller(
        runtime.controller,
        runtime.binding,
        liquidity_sourcing=sourcing,
    )

    assert controller.local_optimizer.objective_weight == pytest.approx(0.7)
    assert controller.global_optimizer.objective_weight == pytest.approx(0.4)
    assert controller.local_n_components == 3
    assert controller.global_n_components == 2
    assert runtime.execution.local.max_orders_per_side == 4
    assert runtime.execution.global_.max_orders_per_side == 2
    assert controller.local_optimizer.return_risk_aversion_beta == pytest.approx(0.2)
    assert controller.global_optimizer.return_risk_aversion_beta == pytest.approx(0.8)
    assert controller.local_optimizer.soft_fill_temp == pytest.approx(0.11)
    assert controller.global_optimizer.soft_fill_temp == pytest.approx(0.33)
    assert controller.local_optimizer.quote_capacity_ratio == pytest.approx(0.8)
    assert controller.global_optimizer.quote_capacity_ratio == pytest.approx(0.6)
    assert controller.local_optimizer.quote_size_max_per_side is None
    assert controller.local_optimizer.quote_imbalance_skew_strength == pytest.approx(0.25)
    assert controller.global_optimizer.quote_imbalance_skew_strength == pytest.approx(0.5)
    assert controller.local_optimizer.soft_fill_queue_model == "pro_rata"
    assert controller.global_optimizer.soft_fill_queue_model == "back_of_queue"
    assert controller.reference_price_source == "local"
    assert controller._l_param_config("local").quote_size_min == pytest.approx(0.2)
    assert controller._l_param_config("global").quote_size_max == pytest.approx(5.0)


def test_build_controller_rejects_flat_venue_optimizer_params() -> None:
    from mm.runtime.config import ControllerRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig
    from mm.strategies.mpc_engine import _build_controller

    runtime = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="USDT_IRT"),
        controller=ControllerRuntimeConfig(
            enabled=True,
            optimizer={"quote_size_min_per_side": 0.1, "quote_size_max_per_side": 1.0},
        ),
    )

    with pytest.raises(ValueError, match="optimizer.local/global"):
        _build_controller(runtime.controller, runtime.binding)


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


def test_joint_optimizer_uses_global_component_count_override() -> None:
    state = _minimal_state(global_active=True)
    joint = _build_optimizer(
        liquidity_sourcing_enabled=True,
        local_optimizer={"n_components": 4},
        global_optimizer={"n_components": 2},
    )
    joint.adam_epochs = 1
    _, diagnostics = joint.optimize_with_diagnostics(state)

    local_flat = diagnostics.rollout.get("local_action_flat") or []
    global_flat = diagnostics.rollout.get("global_action_flat") or []
    assert joint.local_n_components == 4
    assert joint.global_n_components == 2
    assert len(local_flat) == action_param_len(4)
    assert len(global_flat) == action_param_len(2)
    assert diagnostics.config["local_optimizer"]["n_components"] == 4
    assert diagnostics.config["global_optimizer"]["n_components"] == 2
