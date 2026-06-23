"""Model-specific NLL helpers (aligned with runtime ``mm.models.scoring``)."""

from __future__ import annotations

import math
from typing import Any

from mm.models.scoring import ScoreResult, score_local_mid_delta, score_market_flow

from research.model_lab.core.types import IntervalStep, mpc_step_from_interval


def _gaussian_log_pdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(1e-12, float(sigma))
    z = (float(x) - float(mu)) / sigma
    return -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * z * z


def _student_t_logpdf(error: float, variance: float, *, df: float, sigma_min: float) -> float:
    nu = max(2.1, float(df))
    var = max(sigma_min * sigma_min, float(variance))
    scale2 = max(sigma_min * sigma_min, var * (nu - 2.0) / nu)
    return (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi * scale2)
        - ((nu + 1.0) / 2.0) * math.log1p((error * error) / (nu * scale2))
    )


def _logaddexp(a: float, b: float) -> float:
    high = max(a, b)
    low = min(a, b)
    return high + math.log1p(math.exp(low - high))


def score_residual_transition(
    model: Any,
    previous: IntervalStep,
    current: IntervalStep,
) -> ScoreResult:
    """Gaussian NLL for buy/sell VWAP residuals (production ``TorchResidualPriceModel``)."""

    from mm.pricing import reference_prices_from_transition_ticks

    depth = float(getattr(model, "_target_depth_lots", 1000.0))
    prev_ob = previous.orderbook
    cur_ob = current.orderbook
    buy_real, sell_real, buy_mof, sell_mof = reference_prices_from_transition_ticks(
        prev_ob,
        cur_ob,
        float(current.buy_volume),
        float(current.sell_volume),
        depth,
    )
    e_buy = float(buy_real - buy_mof)
    e_sell = float(sell_real - sell_mof)
    mu_buy = float(getattr(model, "_mu_buy", 0.0))
    mu_sell = float(getattr(model, "_mu_sell", 0.0))
    sigma_buy = max(
        float(getattr(model, "_sigma_min", 1e-6)),
        math.sqrt(max(0.0, float(getattr(model, "_sigma2_buy", 1.0)))),
    )
    sigma_sell = max(
        float(getattr(model, "_sigma_min", 1e-6)),
        math.sqrt(max(0.0, float(getattr(model, "_sigma2_sell", 1.0)))),
    )
    ll = _gaussian_log_pdf(e_buy, mu_buy, sigma_buy) + _gaussian_log_pdf(e_sell, mu_sell, sigma_sell)
    z = 0.5 * ((e_buy - mu_buy) / sigma_buy + (e_sell - mu_sell) / sigma_sell)
    return ScoreResult(
        log_likelihood=ll,
        nll=-ll,
        observation={
            "e_buy": e_buy,
            "e_sell": e_sell,
            "mu_buy": mu_buy,
            "mu_sell": mu_sell,
            "sigma_buy": sigma_buy,
            "sigma_sell": sigma_sell,
        },
        standardized_residual=float(z),
    )


def score_conditional_mixture_transition(
    model: Any,
    previous: IntervalStep,
    current: IntervalStep,
) -> ScoreResult:
    """Mixture NLL for component deltas (mirrors ``TorchConditionalMechanicalMixturePriceModel.update``)."""

    import numpy as np

    from mm.pricing.reference_prices_ticks import (
        reference_prices_from_transition_ticks,
        reference_prices_ticks_from_orderbook,
    )

    prev_step = mpc_step_from_interval(previous)
    curr_step = mpc_step_from_interval(current)
    _t_prev, prev_ob, _b_prev, _s_prev, *_ = prev_step
    _t_curr, cur_ob, buy_vol, sell_vol, *_ = curr_step

    try:
        prev_buy, prev_sell = reference_prices_ticks_from_orderbook(prev_ob, model.reference_vwap_vol)
        obs_buy, obs_sell, mech_buy, mech_sell = reference_prices_from_transition_ticks(
            prev_ob,
            cur_ob,
            float(buy_vol),
            float(sell_vol),
            model.reference_vwap_vol,
        )
        prev_mid = float(prev_ob.mid)
        obs_mid = float(cur_ob.mid)
        mech_bid, mech_ask = model._transition_mechanical_best(prev_ob, buy_vol, sell_vol)
    except (ValueError, ZeroDivisionError, TypeError, AttributeError):
        return ScoreResult(skip_reason="invalid_transition")

    prev_spread = model._spread_from_ob(prev_ob)
    obs_spread = model._spread_from_ob(cur_ob)
    mech_spread = max(model._param_eps, mech_ask - mech_bid)
    mech_mid = 0.5 * (mech_bid + mech_ask)
    prev = model._features_from_prices(
        mid=prev_mid, spread=prev_spread, buy_vwap=prev_buy, sell_vwap=prev_sell
    )
    obs = model._features_from_prices(
        mid=obs_mid, spread=obs_spread, buy_vwap=obs_buy, sell_vwap=obs_sell
    )
    mech = model._features_from_prices(
        mid=mech_mid, spread=mech_spread, buy_vwap=mech_buy, sell_vwap=mech_sell
    )
    component_names = model._component_names
    obs_components = {
        "mid": obs["mid"] - prev["mid"],
        "log_spread": math.log(obs["spread"]) - math.log(prev["spread"]),
        "log_buy_shape": math.log(obs["buy_shape"]) - math.log(prev["buy_shape"]),
        "log_sell_shape": math.log(obs["sell_shape"]) - math.log(prev["sell_shape"]),
    }
    mech_components = {
        "mid": mech["mid"] - prev["mid"],
        "log_spread": math.log(mech["spread"]) - math.log(prev["spread"]),
        "log_buy_shape": math.log(mech["buy_shape"]) - math.log(prev["buy_shape"]),
        "log_sell_shape": math.log(mech["sell_shape"]) - math.log(prev["sell_shape"]),
    }
    if not all(math.isfinite(v) for v in (*obs_components.values(), *mech_components.values())):
        return ScoreResult(skip_reason="non_finite_features")

    total_flow = max(0.0, float(buy_vol) + float(sell_vol))
    gate_log_flow = min(math.log1p(total_flow), 12.0)
    q_market = model._market_probability_np(gate_log_flow)
    q_refresh = 1.0 - q_market
    market_log = math.log(q_market)
    refresh_log = math.log(q_refresh)
    for name in component_names:
        x = np.array([1.0, mech_components[name]], dtype=np.float64)
        market_mean = float(x @ model._market_beta[name])
        refresh_mean = float(model._refresh_mean[name])
        market_err = obs_components[name] - market_mean
        refresh_err = obs_components[name] - refresh_mean
        market_log += model._normal_logpdf(market_err, model._market_var[name])
        refresh_log += model._student_t_logpdf(refresh_err, model._refresh_var[name])
    log_norm = _logaddexp(market_log, refresh_log)
    ll = float(log_norm)
    return ScoreResult(
        log_likelihood=ll,
        nll=-ll,
        observation={
            "obs_components": obs_components,
            "mech_components": mech_components,
            "market_prob": q_market,
            "total_flow": total_flow,
        },
    )


def score_flow_step(
    *,
    buy_volume: float,
    sell_volume: float,
    params_before: dict[str, Any],
    kind: str,
) -> ScoreResult:
    return score_market_flow(
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        params_before=params_before,
        kind=kind,
    )


def score_mid_delta_step(
    *,
    delta_mid: float,
    params_before: dict[str, Any],
) -> ScoreResult:
    return score_local_mid_delta(delta_mid=delta_mid, params_before=params_before)
