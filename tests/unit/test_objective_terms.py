from __future__ import annotations

import math

import pytest
import torch

from mm.mpc.l_param import LParamConfig, action_param_len
from mm.mpc.mpc_tensor_batched import _initial_action_uniform_marginal_qrange, _spread_grid_tensor
from mm.mpc.objective_terms import (
    BookQualityParams,
    BookQualitySoftScoring,
    ObjectiveNormalizeParams,
    ValidityParams,
    ViabilityParams,
    apply_ce_objective_normalization,
    book_quality_score,
    combine_validity_term,
    combine_viability_term,
    soft_availability_margin,
    update_ce_ewma_scale,
    update_viability_lambda,
    validity_book_shortfall,
    viability_path_shortfall,
)


def _viability_params(**overrides: float) -> ViabilityParams:
    defaults = {
        "min_viable_quote": 100.0,
        "margin_buffer": 10.0,
        "availability_target": 0.99,
        "shortfall_tolerance": 0.05,
        "lambda_init": 0.0,
        "lambda_learning_rate": 0.02,
        "lambda_leakage": 0.0,
    }
    defaults.update(overrides)
    return ViabilityParams(**defaults)


def test_viability_from_mapping_rejects_inventory_zone_keys() -> None:
    with pytest.raises(ValueError, match="removed viability keys"):
        ViabilityParams.from_mapping({"comfort_low": -0.2, "comfort_high": 0.2})


def test_viability_from_mapping_rejects_legacy_keys() -> None:
    with pytest.raises(ValueError, match="removed viability keys"):
        ViabilityParams.from_mapping({"min_quote_qty": 1.0, "margin_sharpness": 0.1})
    with pytest.raises(ValueError, match="removed viability keys"):
        ViabilityParams.from_mapping({"min_free_quote": 1.0, "margin_buffer": 0.1})


def test_soft_availability_is_high_when_total_balances_exceed_viable_floor() -> None:
    params = _viability_params()
    availability, parts = soft_availability_margin(
        torch.tensor([10.0, 2.0]),
        cash=torch.tensor([2000.0, 500.0]),
        mid_price=torch.tensor([100.0, 100.0]),
        params=params,
    )
    assert float(availability[0]) > 0.9
    assert float(availability[1]) > float(availability[0]) * 0.5
    assert float(parts["viability_bid_margin"][0]) == pytest.approx(1900.0)
    assert float(parts["viability_ask_margin"][0]) == pytest.approx(900.0)


def test_soft_availability_goes_to_zero_when_one_side_lacks_total_base() -> None:
    params = _viability_params()
    availability, _ = soft_availability_margin(
        torch.tensor([0.0]),
        cash=torch.tensor([1000.0]),
        mid_price=torch.tensor([100.0]),
        params=params,
    )
    assert float(availability[0]) < 0.01


def test_soft_availability_penalizes_insufficient_total_quote() -> None:
    params = _viability_params()
    availability, _ = soft_availability_margin(
        torch.tensor([10.0]),
        cash=torch.tensor([50.0]),
        mid_price=torch.tensor([100.0]),
        params=params,
    )
    assert float(availability[0]) < 0.01


def test_viability_path_shortfall_penalizes_low_min_availability() -> None:
    params = _viability_params(availability_target=0.99)
    inventory_path = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float64)
    cash_path = torch.tensor([[1000.0, 1000.0], [1000.0, 1000.0]], dtype=torch.float64)
    mid_path = torch.full_like(inventory_path, 100.0)
    shortfall, components = viability_path_shortfall(
        inventory_path,
        cash_path=cash_path,
        mid_path=mid_path,
        params=params,
    )
    assert float(shortfall) > 0.0
    assert float(components["viability_shortfall"]) == pytest.approx(float(shortfall))
    assert float(components["viability_min_availability"]) < 0.99


def test_viability_path_shortfall_penalizes_post_fill_depletion() -> None:
    """Rollout paths must include post-orderflow wallet snapshots, not only pre-fill."""
    params = _viability_params(availability_target=0.99)
    mid = torch.tensor([[100.0, 100.0]], dtype=torch.float64)
    post_fill_path = {
        "inventory_path": torch.tensor([[10.0, 0.0]], dtype=torch.float64),
        "cash_path": torch.tensor([[1000.0, 50.0]], dtype=torch.float64),
    }
    pre_fill_only_path = {
        "inventory_path": torch.tensor([[10.0, 10.0]], dtype=torch.float64),
        "cash_path": torch.tensor([[1000.0, 1000.0]], dtype=torch.float64),
    }
    shortfall_post, _ = viability_path_shortfall(
        post_fill_path["inventory_path"],
        cash_path=post_fill_path["cash_path"],
        mid_path=mid,
        params=params,
    )
    shortfall_pre, _ = viability_path_shortfall(
        pre_fill_only_path["inventory_path"],
        cash_path=pre_fill_only_path["cash_path"],
        mid_path=mid,
        params=params,
    )
    assert float(shortfall_post) > float(shortfall_pre)


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


def test_update_viability_lambda_increases_on_shortfall() -> None:
    params = _viability_params(shortfall_tolerance=0.05, lambda_learning_rate=0.5)
    lam = update_viability_lambda(
        1.0,
        {"viability_min_availability": 0.7, "viability_shortfall": 0.2},
        params=params,
    )
    assert lam == pytest.approx(1.0 + 0.5 * (0.9 - 0.7))


def test_update_viability_lambda_decreases_when_above_target() -> None:
    params = _viability_params(
        shortfall_tolerance=0.05,
        lambda_learning_rate=0.02,
        lambda_reset_on_target=False,
        lambda_leakage=0.0,
    )
    lam = update_viability_lambda(
        1.0,
        {"viability_min_availability": 0.95},
        params=params,
    )
    assert lam == pytest.approx(1.0 + 0.02 * (0.99 - 0.95))


def test_update_viability_lambda_resets_when_target_met() -> None:
    params = _viability_params(
        shortfall_tolerance=0.05,
        lambda_learning_rate=1.0,
        lambda_init=0.0,
        lambda_reset_on_target=True,
    )
    lam = update_viability_lambda(
        3.0,
        {"viability_min_availability": 0.95},
        params=params,
    )
    assert lam == pytest.approx(0.0)


def test_update_viability_lambda_log_update_is_multiplicative() -> None:
    params = _viability_params(
        shortfall_tolerance=0.05,
        lambda_learning_rate=0.5,
        lambda_reset_on_target=False,
        lambda_log_update=True,
    )
    lam = update_viability_lambda(
        2.0,
        {"viability_min_availability": 0.7},
        params=params,
    )
    assert lam == pytest.approx(2.0 * math.exp(0.5 * (0.99 - 0.7)))


def test_update_viability_lambda_applies_leakage() -> None:
    params = _viability_params(
        shortfall_tolerance=0.1,
        lambda_learning_rate=0.02,
        lambda_leakage=0.01,
        lambda_reset_on_target=False,
    )
    lam = update_viability_lambda(
        1.0,
        {"viability_min_availability": 0.89, "viability_shortfall": 0.01},
        params=params,
    )
    assert lam == pytest.approx(0.99 * 1.0 + 0.02 * (0.99 - 0.89))


def test_update_viability_lambda_leakage_decays_when_on_target() -> None:
    params = _viability_params(
        shortfall_tolerance=0.1,
        lambda_learning_rate=0.02,
        lambda_leakage=0.01,
        lambda_reset_on_target=False,
    )
    lam = update_viability_lambda(
        2.0,
        {"viability_min_availability": 0.99},
        params=params,
    )
    assert lam == pytest.approx(0.99 * 2.0 + 0.02 * (0.99 - 0.99))


def test_combine_viability_term_with_lambda_init_zero_applies_no_penalty() -> None:
    params = _viability_params(lambda_init=0.0)
    shortfall = torch.tensor(0.2, dtype=torch.float64)
    objective, components = combine_viability_term(
        torch.tensor(0.5, dtype=torch.float64),
        {},
        viability_lambda=0.0,
        viability_params=params,
        viability_shortfall=shortfall,
        viability_components={"viability_shortfall": shortfall},
    )
    assert float(objective) == pytest.approx(0.5)
    assert float(components["viability_lambda"]) == pytest.approx(0.0)


def test_combine_viability_term_subtracts_lambda_times_shortfall() -> None:
    base = torch.tensor(0.5, dtype=torch.float64)
    shortfall = torch.tensor(0.2, dtype=torch.float64)
    objective, components = combine_viability_term(
        base,
        {"risk_adjusted_return": base},
        viability_lambda=2.0,
        viability_shortfall=shortfall,
        viability_components={"viability_shortfall": shortfall},
    )
    assert float(objective) == pytest.approx(0.5 - 2.0 * 0.2)
    assert float(components["viability_penalty"]) == pytest.approx(-0.4)
    assert float(components["viability_penalty_raw"]) == pytest.approx(0.2)
    assert float(components["viability_lambda"]) == pytest.approx(2.0)


def test_combine_viability_term_uses_lambda_init_when_dual_state_is_zero() -> None:
    params = _viability_params(lambda_init=1.5)
    shortfall = torch.tensor(0.2, dtype=torch.float64)
    objective, components = combine_viability_term(
        torch.tensor(0.0, dtype=torch.float64),
        {},
        viability_lambda=0.0,
        viability_params=params,
        viability_shortfall=shortfall,
        viability_components={"viability_shortfall": shortfall},
    )
    assert float(objective) == pytest.approx(-1.5 * 0.2)
    assert float(components["viability_lambda"]) == pytest.approx(1.5)


def test_update_viability_lambda_reads_penalty_raw_fallback() -> None:
    params = _viability_params(shortfall_tolerance=0.05, lambda_learning_rate=0.5)
    lam = update_viability_lambda(
        1.0,
        {"viability_penalty_raw": 0.2},
        params=params,
    )
    assert lam == pytest.approx(1.0 + 0.5 * 0.2)


def test_validity_from_mapping_rejects_w_book() -> None:
    with pytest.raises(ValueError, match="w_book"):
        ValidityParams.from_mapping({"w_book": 0.1, "min_score": 0.3})


def test_validity_book_shortfall_is_zero_when_score_meets_minimum() -> None:
    score = torch.tensor(0.5, dtype=torch.float64)
    params = ValidityParams(min_score=0.3, penalty_scale=100.0)
    shortfall, parts = validity_book_shortfall(score, params=params)
    assert float(shortfall) == pytest.approx(0.0)
    assert float(parts["validity_book_score"]) == pytest.approx(0.5)


def test_combine_validity_term_applies_fixed_penalty_below_minimum() -> None:
    base = torch.tensor(0.5, dtype=torch.float64)
    book = torch.tensor(0.1, dtype=torch.float64)
    params = ValidityParams(min_score=0.3, penalty_scale=100.0)
    objective, components = combine_validity_term(
        base,
        {"risk_adjusted_return": base},
        params=params,
        book_score=book,
        book_components={"book_quality": book, "book_presence": book},
    )
    assert float(objective) == pytest.approx(0.5 - 100.0 * 0.2)
    assert float(components["validity_penalty"]) == pytest.approx(-20.0)
    assert float(components["validity_penalty_raw"]) == pytest.approx(0.2)
    assert float(components["validity_book_score"]) == pytest.approx(0.1)
    assert float(components["risk_adjusted_return"]) == pytest.approx(0.5)


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
        params=BookQualityParams(soft_scoring=BookQualitySoftScoring(measure="quoted")),
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
        n_components=2,
    )
    flat = torch.tensor(
        _initial_action_uniform_marginal_qrange(
            n_components=2,
            spread_max=spread_max,
            l_param=l_param,
        ),
        dtype=torch.float64,
    )
    market_cum_bid = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64)
    market_cum_ask = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64)
    quoted_score, _ = book_quality_score(
        flat,
        l_param=l_param,
        n_components=2,
        spread_grid=spread_grid,
        bid_quote_size_min=0.0,
        bid_quote_size_max=10.0,
        ask_quote_size_min=0.0,
        ask_quote_size_max=10.0,
        mid_price=100.0,
        params=BookQualityParams(soft_scoring=BookQualitySoftScoring(measure="quoted")),
    )
    combined_score, _ = book_quality_score(
        flat,
        l_param=l_param,
        n_components=2,
        spread_grid=spread_grid,
        bid_quote_size_min=0.0,
        bid_quote_size_max=10.0,
        ask_quote_size_min=0.0,
        ask_quote_size_max=10.0,
        mid_price=100.0,
        params=BookQualityParams(soft_scoring=BookQualitySoftScoring(measure="combined")),
        market_cum_bid=market_cum_bid,
        market_cum_ask=market_cum_ask,
    )
    assert float(combined_score) >= float(quoted_score)
