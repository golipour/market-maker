from __future__ import annotations

import pytest
import torch

from mm.mpc.l_param import LParamConfig, action_param_len
from mm.mpc.mpc_tensor_batched import _initial_action_uniform_marginal_qrange, _spread_grid_tensor
from mm.mpc.objective_terms import (
    BookQualityParams,
    InventoryZoneParams,
    ObjectiveNormalizeParams,
    apply_ce_objective_normalization,
    book_quality_score,
    combine_soft_objective_terms,
    inventory_zone_penalty,
    inventory_zone_path_mean,
    update_ce_ewma_scale,
    wallet_imbalance,
)


def _sigmoid_zone(**overrides: float) -> InventoryZoneParams:
    defaults = {
        "comfort_low": -0.1,
        "comfort_high": 0.1,
        "low_ramp_width": 0.02,
        "high_ramp_width": 0.02,
        "low_penalty_cap": 1.0,
        "high_penalty_cap": 0.6,
    }
    defaults.update(overrides)
    return InventoryZoneParams(**defaults)


def test_wallet_imbalance_is_negative_when_cash_heavy() -> None:
    imbalance = wallet_imbalance(
        torch.tensor([0.0, 1.0]),
        mid_price=torch.tensor([100.0, 100.0]),
        cash=torch.tensor([1000.0, 500.0]),
    )
    assert float(imbalance[0]) == pytest.approx(-1.0)
    assert float(imbalance[1]) < 0.0


def test_wallet_imbalance_is_positive_when_base_heavy() -> None:
    imbalance = wallet_imbalance(
        torch.tensor([10.0, 20.0]),
        mid_price=torch.tensor([100.0, 100.0]),
        cash=torch.tensor([100.0, 100.0]),
    )
    assert float(imbalance[0]) > 0.0
    assert float(imbalance[1]) > float(imbalance[0])


def test_inventory_zone_from_mapping_rejects_power_zone_keys() -> None:
    with pytest.raises(ValueError, match="power-zone keys were removed"):
        InventoryZoneParams.from_mapping(
            {"q_lo": -0.1, "q_hi": 0.1, "q_min": -0.5, "q_max": 0.5, "p": 2.0}
        )


def test_inventory_zone_penalty_is_near_zero_inside_comfort_band() -> None:
    zone = _sigmoid_zone()
    center = torch.tensor([0.0], dtype=torch.float64)
    extremes = torch.tensor([-0.9, 0.9], dtype=torch.float64)
    center_penalty = float(inventory_zone_penalty(center, zone))
    extreme_penalty = inventory_zone_penalty(extremes, zone)
    assert center_penalty < 0.02
    assert center_penalty < float(extreme_penalty.min()) * 0.1


def test_inventory_zone_penalty_saturates_at_extremes() -> None:
    zone = _sigmoid_zone()
    cap_sum = zone.low_penalty_cap + zone.high_penalty_cap
    imbalance = torch.tensor([-0.95, 0.95], dtype=torch.float64)
    penalty = inventory_zone_penalty(imbalance, zone)
    assert float(penalty[0]) == pytest.approx(zone.low_penalty_cap / cap_sum, rel=1e-3)
    assert float(penalty[1]) == pytest.approx(zone.high_penalty_cap / cap_sum, rel=1e-3)
    assert float(penalty.max()) <= 1.0 + 1e-9


def test_inventory_zone_low_side_penalizes_cash_heavy_imbalance() -> None:
    zone = _sigmoid_zone(comfort_low=-0.1)
    penalty, parts = inventory_zone_penalty(
        torch.tensor([-0.8], dtype=torch.float64),
        zone,
        return_components=True,
    )
    assert float(penalty[0]) > 0.5
    assert float(parts["inventory_zone_low_penalty"][0]) > float(
        parts["inventory_zone_high_penalty"][0]
    )


def test_inventory_zone_path_mean_reports_imbalance_and_side_components() -> None:
    zone = _sigmoid_zone()
    inventory_path = torch.tensor([[0.0, 5.0], [20.0, 0.0]], dtype=torch.float64)
    cash_path = torch.tensor([[1000.0, 500.0], [100.0, 800.0]], dtype=torch.float64)
    mid_path = torch.full_like(inventory_path, 100.0)
    path_mean, components = inventory_zone_path_mean(
        inventory_path,
        cash_path=cash_path,
        mid_path=mid_path,
        zone=zone,
    )
    assert 0.0 <= float(path_mean) <= 1.0
    assert float(components["inventory_zone_penalty"]) == pytest.approx(float(path_mean))
    assert float(components["inventory_zone_imbalance"]) != 0.0
    assert float(components["inventory_zone_low_penalty"]) > 0.0
    assert float(components["inventory_zone_high_penalty"]) > 0.0


def test_apply_ce_objective_normalization_scales_by_previous_ewma() -> None:
    normalize = ObjectiveNormalizeParams(enabled=True, ce_ewma_decay=0.95)
    ce = torch.tensor(0.002, dtype=torch.float64)
    objective, components, _ = apply_ce_objective_normalization(
        ce,
        {"risk_adjusted_return": ce},
        normalize=normalize,
        ce_ewma_scale=0.001,
    )
    assert float(components["risk_adjusted_return"]) == pytest.approx(2.0)
    assert float(objective) == pytest.approx(2.0)
    assert float(components["risk_adjusted_return_raw"]) == pytest.approx(0.002)


def test_update_ce_ewma_scale_tracks_magnitude() -> None:
    normalize = ObjectiveNormalizeParams(enabled=True, ce_ewma_decay=0.9)
    scale = update_ce_ewma_scale(
        0.001,
        {"risk_adjusted_return_raw": 0.002},
        normalize=normalize,
    )
    assert scale == pytest.approx(0.9 * 0.001 + 0.1 * 0.002)


def test_combine_soft_objective_terms_keeps_ce_component_pure() -> None:
    base = torch.tensor(0.5, dtype=torch.float64)
    book = torch.tensor(0.8, dtype=torch.float64)
    zone = torch.tensor(0.2, dtype=torch.float64)
    objective, components = combine_soft_objective_terms(
        base,
        {"risk_adjusted_return": base},
        w_book=0.1,
        w_inv=0.05,
        book_score=book,
        inventory_zone_mean=zone,
        book_components={"book_quality": book},
        zone_components={"inventory_zone_penalty": zone},
    )
    assert float(objective) == pytest.approx(0.5 + 0.1 * 0.8 - 0.05 * 0.2)
    assert float(components["risk_adjusted_return"]) == pytest.approx(0.5)
    assert float(components["book_quality"]) == pytest.approx(0.08)
    assert float(components["inventory_zone_penalty"]) == pytest.approx(-0.01)
    assert float(components["book_quality_score"]) == pytest.approx(0.8)
    assert float(components["inventory_zone_penalty_raw"]) == pytest.approx(0.2)


def test_book_quality_quoted_mode_gradients_flow() -> None:
    spread_max = 100.0
    spread_step = 20.0
    grid_levels = 5
    spread_grid = _spread_grid_tensor(
        torch,
        grid_levels,
        spread_step,
        torch.device("cpu"),
        torch.float64,
        spread_max=spread_max,
    )
    l_param = LParamConfig(
        quote_size_min=0.0,
        quote_size_max=10.0,
        spread_max=spread_max,
        n_components=2,
    )
    flat = torch.tensor(
        _initial_action_uniform_marginal_qrange(
            n_components=2,
            spread_max=spread_max,
            l_param=l_param,
        ),
        dtype=torch.float64,
        requires_grad=True,
    )
    score, components = book_quality_score(
        flat,
        l_param=l_param,
        n_components=2,
        spread_grid=spread_grid,
        bid_quote_size_min=0.0,
        bid_quote_size_max=10.0,
        ask_quote_size_min=0.0,
        ask_quote_size_max=10.0,
        mid_price=100.0,
        params=BookQualityParams(measure="quoted"),
    )
    score.backward()
    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()
    assert 0.0 <= float(score) <= 1.0
    assert 0.0 <= float(components["book_uniformity"]) <= 1.0


def test_book_quality_combined_mode_boosts_presence_from_market_depth() -> None:
    spread_max = 100.0
    spread_step = 20.0
    spread_grid = _spread_grid_tensor(
        torch,
        grid_levels=5,
        spread_step=spread_step,
        device=torch.device("cpu"),
        dtype=torch.float64,
        spread_max=spread_max,
    )
    l_param = LParamConfig(
        quote_size_min=0.0,
        quote_size_max=10.0,
        spread_max=spread_max,
        n_components=1,
    )
    flat = torch.zeros(action_param_len(1), dtype=torch.float64)
    flat[6] = -30.0
    flat[7] = -30.0
    market_cum_bid = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float64)
    market_cum_ask = torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0], dtype=torch.float64)
    quoted_score, quoted_components = book_quality_score(
        flat,
        l_param=l_param,
        n_components=1,
        spread_grid=spread_grid,
        bid_quote_size_min=0.0,
        bid_quote_size_max=10.0,
        ask_quote_size_min=0.0,
        ask_quote_size_max=10.0,
        mid_price=100.0,
        params=BookQualityParams(measure="quoted", depth_target=1.0),
    )
    combined_score, combined_components = book_quality_score(
        flat,
        l_param=l_param,
        n_components=1,
        spread_grid=spread_grid,
        bid_quote_size_min=0.0,
        bid_quote_size_max=10.0,
        ask_quote_size_min=0.0,
        ask_quote_size_max=10.0,
        mid_price=100.0,
        params=BookQualityParams(measure="combined", depth_target=1.0),
        market_cum_bid=market_cum_bid,
        market_cum_ask=market_cum_ask,
    )
    assert float(combined_components["book_presence"]) > float(quoted_components["book_presence"])
