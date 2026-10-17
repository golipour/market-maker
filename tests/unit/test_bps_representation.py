import pytest
import numpy as np
import torch

from mm.runtime.config import OrderbookGridConfig, ExecutionGridConfig, MarketMakerRuntimeConfig
from mm.mpc.objective_terms import (
    ViabilityParams,
    wallet_quadratic_deficit,
    BookQualityHardStructure,
    BookQualitySoftScoring,
)
from mm.mpc.l_param import (
    ReservationSkewParams,
    SpreadStructureParams,
    curve_spread_for_market_spread,
    curve_spread_for_market_spread_np,
    LParamConfig,
)
from mm.mpc.mpc_state import EffectiveSigmoidMixtureAction


def test_orderbook_grid_bps_resolution() -> None:
    # 100 bps = 1.0%, 5 bps = 0.05%
    grid = OrderbookGridConfig(
        spread_max_bps=100.0,
        spread_step_bps=5.0,
    )
    assert grid.is_bps is True
    assert grid.levels == 20

    # Test resolution with mid = 260,000 IRT
    mid = 260_000.0
    res_max = grid.resolved_spread_max(mid=mid)
    res_step = grid.resolved_spread_step(mid=mid)
    assert res_max == pytest.approx(2600.0)
    assert res_step == pytest.approx(130.0)

    ladder = grid.spread_ladder(mid=mid)
    assert len(ladder) == 20
    assert ladder[0] == pytest.approx(130.0)
    assert ladder[-1] == pytest.approx(2600.0)


def test_execution_grid_bps_resolution() -> None:
    canon = OrderbookGridConfig(
        spread_max_bps=100.0,
        spread_step_bps=5.0,
    )
    exec_grid = ExecutionGridConfig(
        mode="log",
        levels=16,
        spread_min_bps=5.0,
        spread_max_bps=100.0,
    )
    assert exec_grid.is_bps is True
    mid = 260_000.0

    res_max = exec_grid.resolved_spread_max(canonical=canon, mid=mid)
    res_min = exec_grid.resolved_spread_min(mid=mid)
    assert res_max == pytest.approx(2600.0)
    assert res_min == pytest.approx(130.0)

    ladder = exec_grid.spread_ladder(canonical=canon, mid=mid)
    assert len(ladder) == 16
    assert ladder[0] == pytest.approx(130.0)
    assert ladder[-1] == pytest.approx(2600.0)


def test_viability_wealth_ratio_deficit() -> None:
    # 15% reserve floor
    params = ViabilityParams.from_mapping({
        "min_viable_ratio": 0.15,
        "recovery_fraction": 0.25,
    })
    assert params is not None
    assert params.min_viable_ratio == 0.15
    assert params.min_viable_bps == 1500.0

    # Total wealth: cash = 850M IRT, inventory = 550 USDT * 260,000 = 143M IRT.
    # Total wealth = 993M IRT.
    # 15% floor = 148.95M IRT.
    cash = torch.tensor(850_000_000.0)
    inventory = torch.tensor(550.0)
    mid = torch.tensor(260_000.0)

    deficit, details = wallet_quadratic_deficit(
        inventory=inventory,
        cash=cash,
        mid_price=mid,
        params=params,
    )
    # Cash is well above reserve (850M > 148.95M) -> bid deficit = 0
    assert details["viability_bid_deficit"].item() == pytest.approx(0.0)
    # Base notional is 143M < 148.95M -> mild deficit of ~4% shortfall
    ask_def = details["viability_ask_deficit"].item()
    assert 0.0 < ask_def < 0.1
    # Deficit is small, NOT blowing up to 61.26
    assert deficit.item() < 0.01


def test_reservation_skew_bps_and_clamp_guard() -> None:
    skew_params = ReservationSkewParams.from_mapping({"max_abs_skew_bps": 20.0})
    assert skew_params is not None
    assert skew_params.max_abs_skew == pytest.approx(0.0020)

    # Test that even if shift is huge (e.g. 5000 on a 1000 spread_max grid),
    # curve_spread_for_market_spread clamps the shift to 0.8 * spread_max,
    # ensuring the outer grid points never collapse to 0.0.
    mid = 260_000.0
    # Say reservation skew offset was huge: -0.01 (shift = -2600 on spread_max = 1000)
    market_spread = 1000.0
    spread_max = 1000.0
    clamped_spread = curve_spread_for_market_spread_np(
        market_spread,
        side="bid",
        mid_price=mid,
        reservation_r_offset=-0.01,
        spread_max=spread_max,
    )
    # Without clamp, 1000 - 2600 = -1600 -> 0.0 (wiping out book).
    # With clamp to 0.8 * 1000 = 800, 1000 - 800 = 200.0 > 0.0!
    assert clamped_spread == pytest.approx(200.0)

    # PyTorch version check
    clamped_torch = curve_spread_for_market_spread(
        torch.tensor(1000.0),
        side="bid",
        mid_price=torch.tensor(mid),
        reservation_r_offset=torch.tensor(-0.01),
        spread_max=torch.tensor(spread_max),
    )
    assert clamped_torch.item() == pytest.approx(200.0)


def test_effective_action_spread_max_guard() -> None:
    # Construct an EffectiveSigmoidMixtureAction with extreme skew
    action = EffectiveSigmoidMixtureAction(
        bid_L=(100.0,),
        bid_k=(0.01,),
        bid_x0=(100.0,),
        ask_L=(100.0,),
        ask_k=(0.01,),
        ask_x0=(100.0,),
        reservation_r_offset=-0.01,
        reservation_mid=260_000.0,
        spread_max=1000.0,
    )
    # The effective depth at spread_max must remain positive
    depth_at_max = action.effective_depth_bid(1000.0)
    assert depth_at_max > 0.0
