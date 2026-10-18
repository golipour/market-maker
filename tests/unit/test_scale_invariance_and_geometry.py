import math
import pytest
import numpy as np
import torch

from mm.mpc.l_param import (
    LParamConfig,
    action_param_len,
    materialize_side_torch,
    spread_ratios_torch,
    validate_l_param_config,
)
from mm.mpc.mpc_state import State, OrderbookRelative
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched
from mm.mpc.objective_terms import ViabilityParams
from mm.runtime.config import (
    ControllerRuntimeConfig,
    ExecutionGridConfig,
    OrderbookGridConfig,
    SpreadGeometryConfig,
)


class DummyPriceModel:
    def sample_residual(self, shape, device=None, dtype=None):
        return (
            torch.zeros(shape, dtype=dtype, device=device),
            torch.zeros(shape, dtype=dtype, device=device),
        )


class DummyFlowModel:
    def sample(self, shape, device=None, dtype=None):
        return (
            torch.ones(shape, dtype=dtype, device=device),
            torch.ones(shape, dtype=dtype, device=device),
        )


class DummyExecutionModel:
    maker_fee = 0.0


def test_spread_geometry_derivation_and_validation() -> None:
    geom = SpreadGeometryConfig(half_spread_bps=5.0, skew_share=0.4, tightness_share=1.5)
    derived = geom.derived(spread_max_bps=38.0)
    assert derived["action_max_spread_bps"] == 5.0
    assert derived["spread_min_bps"] == pytest.approx(2.0)
    assert derived["max_abs_skew_bps"] == pytest.approx(2.0)
    assert derived["tightness_spread_scale_bps"] == pytest.approx(7.5)

    with pytest.raises(ValueError, match="spread_max_bps"):
        geom.derived(spread_max_bps=4.0)

    with pytest.raises(ValueError, match="skew_share"):
        SpreadGeometryConfig(half_spread_bps=5.0, skew_share=0.8).validate(38.0)


def test_controller_spread_coherence_validator_rejections() -> None:
    # 1. action_max_spread >= spread_max must be rejected
    with pytest.raises(ValueError, match="action_max_spread .* must be < spread_max"):
        ControllerRuntimeConfig(
            orderbook_grid=OrderbookGridConfig(spread_max_bps=38.0, spread_step_bps=1.9),
            optimizer={
                "local": {
                    "book_quality": {
                        "hard_structure": {"action_max_spread_bps": 50.0, "action_min_depth": 1.0}
                    }
                }
            },
        )

    # 2. spread_min >= action_max_spread must be rejected
    with pytest.raises(ValueError, match="spread_min .* must be < action_max_spread"):
        ControllerRuntimeConfig(
            orderbook_grid=OrderbookGridConfig(spread_max_bps=38.0, spread_step_bps=1.9),
            execution_grid=ExecutionGridConfig(spread_min_bps=5.0, spread_max_bps=38.0),
            optimizer={
                "local": {
                    "book_quality": {
                        "hard_structure": {"action_max_spread_bps": 3.8, "action_min_depth": 1.0}
                    }
                }
            },
        )

    # 3. reservation skew overpower (> 0.5 * action_max) must be rejected
    with pytest.raises(ValueError, match="reservation_skew .* must be <= 0.5 \\* action_max_spread"):
        ControllerRuntimeConfig(
            orderbook_grid=OrderbookGridConfig(spread_max_bps=38.0, spread_step_bps=1.9),
            optimizer={
                "local": {
                    "book_quality": {
                        "hard_structure": {"action_max_spread_bps": 3.8, "action_min_depth": 1.0}
                    },
                    "reservation_skew": {"max_abs_skew_bps": 20.0},
                }
            },
        )

    # 4. tightness_spread_scale > 2 * spread_max must be rejected
    with pytest.raises(ValueError, match="tightness_spread_scale .* must be <= 2 \\* spread_max"):
        ControllerRuntimeConfig(
            orderbook_grid=OrderbookGridConfig(spread_max_bps=38.0, spread_step_bps=1.9),
            optimizer={
                "local": {
                    "book_quality": {
                        "hard_structure": {"action_max_spread_bps": 3.8, "action_min_depth": 1.0},
                        "soft_scoring": {"tightness_spread_scale_bps": 100.0},
                    }
                }
            },
        )


def test_viability_ratio_and_bps_agreement() -> None:
    # Coherent configuration
    v = ViabilityParams.from_mapping({"min_viable_ratio": 0.1, "min_viable_bps": 1000.0})
    assert v.min_viable_ratio == 0.1

    # Disagreeing configuration
    with pytest.raises(ValueError, match="disagree"):
        ViabilityParams.from_mapping({"min_viable_ratio": 0.1, "min_viable_bps": 500.0})


def test_scale_invariance_across_100x_mid() -> None:
    spread_max_bps = 38.0
    spread_step_bps = 1.9
    action_max_spread_bps = 3.8

    optimizer = MonteCarloMPCTensorBatched(
        local_price_model=DummyPriceModel(),
        local_market_flow_model=DummyFlowModel(),
        execution_model=DummyExecutionModel(),
        n_components=2,
        n_samples=4,
        horizon=1,
        adam_epochs=1,
        spread_max=spread_max_bps,
        spread_step=spread_step_bps,
        is_bps=True,
        spread_max_bps=spread_max_bps,
        spread_step_bps=spread_step_bps,
        local_optimizer={
            "book_quality": {
                "hard_structure": {
                    "action_max_spread_bps": action_max_spread_bps,
                    "action_min_depth": 1.0,
                }
            },
            "reservation_skew": {"max_abs_skew_bps": 1.0},
        },
        compile_soft_fill=False,
    )

    # State at mid = 1,000
    mid1 = 1000.0
    ob1 = OrderbookRelative(
        bids=((0.001, 10.0), (0.002, 20.0)),
        asks=((0.001, 10.0), (0.002, 20.0)),
    )
    s1 = State(mid=mid1, ob_rel=ob1, buy_vol=1.0, sell_vol=1.0, inventory=0.0)
    action1 = optimizer.optimize(s1)

    # State at mid = 100,000 (100x mid)
    mid2 = 100000.0
    ob2 = OrderbookRelative(
        bids=((0.001, 10.0), (0.002, 20.0)),
        asks=((0.001, 10.0), (0.002, 20.0)),
    )
    s2 = State(mid=mid2, ob_rel=ob2, buy_vol=1.0, sell_vol=1.0, inventory=0.0)
    action2 = optimizer.optimize(s2)

    # Both actions must scale centers exactly by 100x in quote currency
    ratio = mid2 / mid1
    assert ratio == 100.0
    for x0_1, x0_2 in zip(action1.bid_x0, action2.bid_x0):
        assert x0_2 == pytest.approx(x0_1 * ratio, rel=1e-5)
    for x0_1, x0_2 in zip(action1.ask_x0, action2.ask_x0):
        assert x0_2 == pytest.approx(x0_1 * ratio, rel=1e-5)

        # First touch slope k_head scales inversely with spread
        assert action2.bid_k[0] == pytest.approx(action1.bid_k[0] / ratio, rel=0.05)


def test_crn_seed_reproducibility() -> None:
    opt1 = MonteCarloMPCTensorBatched(
        local_price_model=DummyPriceModel(),
        local_market_flow_model=DummyFlowModel(),
        execution_model=DummyExecutionModel(),
        n_components=2,
        n_samples=8,
        horizon=2,
        adam_epochs=2,
        crn_enabled=True,
        crn_seed=12345,
        compile_soft_fill=False,
    )
    opt2 = MonteCarloMPCTensorBatched(
        local_price_model=DummyPriceModel(),
        local_market_flow_model=DummyFlowModel(),
        execution_model=DummyExecutionModel(),
        n_components=2,
        n_samples=8,
        horizon=2,
        adam_epochs=2,
        crn_enabled=True,
        crn_seed=12345,
        compile_soft_fill=False,
    )
    ob = OrderbookRelative(
        bids=((1.0, 10.0),),
        asks=((1.0, 10.0),),
    )
    s = State(mid=1000.0, ob_rel=ob, buy_vol=1.0, sell_vol=1.0, inventory=0.0)
    act1 = opt1.optimize(s)
    act2 = opt2.optimize(s)
    assert act1.bid_x0 == pytest.approx(act2.bid_x0)
    assert act1.bid_L == pytest.approx(act2.bid_L)


def test_dynamo_single_compilation_across_mid_sweep() -> None:
    if not hasattr(torch, "_dynamo"):
        pytest.skip("torch._dynamo not available")

    torch._dynamo.reset()
    compile_counts = {"count": 0}

    def counting_backend(gm, sample_inputs):
        compile_counts["count"] += 1
        return gm

    from mm.mpc.mpc_tensor_batched import soft_fill_batched

    compiled_soft_fill = torch.compile(soft_fill_batched, backend=counting_backend)

    n = 2
    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=10.0,
        spread_max=38.0,
        n_components=n,
        quote_spread_max=3.8,
        reservation_r_max=0.001,
    )

    params = torch.zeros(action_param_len(n), dtype=torch.float64, requires_grad=True)
    book = torch.zeros((4, 20), dtype=torch.float64)
    vols = torch.ones((4, 1), dtype=torch.float64)
    size_min = torch.zeros((4, 1), dtype=torch.float64)
    size_max = torch.ones((4, 1), dtype=torch.float64)

    # Call with mid = 1,000
    mid1 = torch.full((4, 1), 1000.0, dtype=torch.float64)
    grid1 = torch.linspace(1.9 * 0.1, 38.0 * 0.1, 20, dtype=torch.float64)
    compiled_soft_fill(
        params,
        n,
        mid1,
        book,
        book,
        vols,
        vols,
        grid1,
        0.0,
        bid_quote_size_min=size_min,
        bid_quote_size_max=size_max,
        ask_quote_size_min=size_min,
        ask_quote_size_max=size_max,
        l_param=l_param,
    )
    initial_compiles = compile_counts["count"]
    assert initial_compiles > 0

    # Call with mid = 262,837.5 (different mid, different grid numbers, but same shape and constant canonical l_param)
    mid2 = torch.full((4, 1), 262837.5, dtype=torch.float64)
    grid2 = torch.linspace(1.9 * 26.28, 38.0 * 26.28, 20, dtype=torch.float64)
    compiled_soft_fill(
        params,
        n,
        mid2,
        book,
        book,
        vols,
        vols,
        grid2,
        0.0,
        bid_quote_size_min=size_min,
        bid_quote_size_max=size_max,
        ask_quote_size_min=size_min,
        ask_quote_size_max=size_max,
        l_param=l_param,
    )

    # Must NOT recompile!
    assert compile_counts["count"] == initial_compiles
