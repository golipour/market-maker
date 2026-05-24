"""Pre-update negative log-likelihood scoring for controller diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mm.diagnostics.models import ModelScore
from mm.mpc.market_order_flow_torch import (
    HurdleOrderFlow,
    _discrete_gamma_pmf,
    _discrete_weibull_pmf,
    _hurdle_gamma_pmf,
    _hurdle_weibull_pmf,
)
from mm.core.models.strategy import StrategyInputFrame
from mm.mpc.mpc_state import State

_LOG_EPS = 1e-300
_MIN_PROB = 1e-300


@dataclass(frozen=True, slots=True)
class ScoreResult:
    log_likelihood: float | None = None
    nll: float | None = None
    skip_reason: str | None = None
    observation: dict[str, Any] = field(default_factory=dict)
    standardized_residual: float | None = None


def _nll_from_log_prob(log_prob: float) -> tuple[float, float]:
    ll = float(log_prob)
    return ll, -ll


def _hurdle_lognormal_pmf(k: int, p0: float, mu: float, sigma: float) -> float:
    """P(X=k) with P(0)=p0; delegates to ``HurdleOrderFlow`` with p(nonzero)=1-p0."""
    if k < 0:
        return 0.0
    p0 = max(0.0, min(1.0, float(p0)))
    p_pos = max(0.0, min(1.0, 1.0 - p0))
    return float(HurdleOrderFlow._pmf_hurdle(k, p_pos, float(mu), float(sigma)))


def _flow_side_prob(k: int, params: dict[str, Any], *, side: str, kind: str) -> float:
    if side == "buy":
        p0 = params.get("p0_buy", params.get("p_buy"))
        if kind in ("hurdle_lognormal", "hurdle_lognormal_model"):
            return _hurdle_lognormal_pmf(
                k,
                float(p0 if p0 is not None else 0.3),
                float(params.get("mu_buy", 0.0)),
                float(params.get("sigma_buy", 1.0)),
            )
        if kind in ("hurdle_gamma",):
            return _hurdle_gamma_pmf(
                k,
                float(p0 if p0 is not None else 0.3),
                float(params.get("shape_buy", 1.0)),
                float(params.get("scale_buy", 1.0)),
            )
        if kind in ("hurdle_weibull",):
            return _hurdle_weibull_pmf(
                k,
                float(p0 if p0 is not None else 0.3),
                float(params.get("wb_k_buy", 2.0)),
                float(params.get("wb_lambda_buy", 1.0)),
            )
        if kind in ("gamma",):
            return _discrete_gamma_pmf(k, float(params.get("shape_buy", 1.0)), float(params.get("scale_buy", 1.0)))
        if kind in ("weibull",):
            return _discrete_weibull_pmf(
                k, float(params.get("wb_k_buy", 2.0)), float(params.get("wb_lambda_buy", 1.0))
            )
    else:
        p0 = params.get("p0_sell", params.get("p_sell"))
        if kind in ("hurdle_lognormal", "hurdle_lognormal_model"):
            return _hurdle_lognormal_pmf(
                k,
                float(p0 if p0 is not None else 0.3),
                float(params.get("mu_sell", 0.0)),
                float(params.get("sigma_sell", 1.0)),
            )
        if kind in ("hurdle_gamma",):
            return _hurdle_gamma_pmf(
                k,
                float(p0 if p0 is not None else 0.3),
                float(params.get("shape_sell", 1.0)),
                float(params.get("scale_sell", 1.0)),
            )
        if kind in ("hurdle_weibull",):
            return _hurdle_weibull_pmf(
                k,
                float(p0 if p0 is not None else 0.3),
                float(params.get("wb_k_sell", params.get("wb_k_buy", 2.0))),
                float(params.get("wb_lambda_sell", params.get("wb_lambda_buy", 1.0))),
            )
        if kind in ("gamma",):
            return _discrete_gamma_pmf(
                k, float(params.get("shape_sell", 1.0)), float(params.get("scale_sell", 1.0))
            )
        if kind in ("weibull",):
            return _discrete_weibull_pmf(
                k,
                float(params.get("wb_k_sell", params.get("wb_k_buy", 2.0))),
                float(params.get("wb_lambda_sell", params.get("wb_lambda_buy", 1.0))),
            )
    return 0.0


def _normalize_flow_kind(kind: str) -> str:
    k = kind.lower().strip()
    aliases = {
        "hurdle_lognormal": "hurdle_lognormal",
        "hurdle_gamma": "hurdle_gamma",
        "hurdle_weibull": "hurdle_weibull",
        "gamma": "gamma",
        "weibull": "weibull",
    }
    return aliases.get(k, k)


def score_market_flow(
    *,
    buy_volume: float,
    sell_volume: float,
    params_before: dict[str, Any],
    kind: str,
) -> ScoreResult:
    """Score realized buy/sell flow using pre-update parameters."""

    flow_kind = _normalize_flow_kind(kind)
    supported = {"hurdle_lognormal", "hurdle_gamma", "hurdle_weibull", "gamma", "weibull"}
    v_buy = int(round(float(buy_volume)))
    v_sell = int(round(float(sell_volume)))
    observation = {"buy_volume": v_buy, "sell_volume": v_sell}
    if flow_kind not in supported:
        return ScoreResult(skip_reason="unsupported_model", observation=observation)

    prob_buy = _flow_side_prob(v_buy, params_before, side="buy", kind=flow_kind)
    prob_sell = _flow_side_prob(v_sell, params_before, side="sell", kind=flow_kind)
    prob = max(_MIN_PROB, float(prob_buy) * float(prob_sell))
    ll, nll = _nll_from_log_prob(math.log(prob))
    return ScoreResult(
        log_likelihood=ll,
        nll=nll,
        observation={**observation, "prob_buy": prob_buy, "prob_sell": prob_sell},
    )


def _gaussian_log_pdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(1e-12, float(sigma))
    z = (float(x) - float(mu)) / sigma
    return -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * z * z


def score_primary_mid_delta(
    *,
    delta_mid: float,
    params_before: dict[str, Any],
) -> ScoreResult:
    """Gaussian NLL for realized primary mid change."""

    mu = float(params_before.get("mu_mid", 0.0))
    sigma = float(params_before.get("sigma_mid", params_before.get("sigma_buy", 1e-6)))
    if not math.isfinite(sigma) or sigma <= 0.0:
        return ScoreResult(
            skip_reason="invalid_sigma",
            observation={"delta_mid": float(delta_mid), "mu_mid": mu, "sigma_mid": sigma},
        )
    ll = _gaussian_log_pdf(delta_mid, mu, sigma)
    nll = -ll
    z = (float(delta_mid) - mu) / sigma
    return ScoreResult(
        log_likelihood=ll,
        nll=nll,
        observation={"delta_mid": float(delta_mid), "mu_mid_before": mu, "sigma_mid_before": sigma},
        standardized_residual=float(z),
    )


def _price_params_before(model: Any) -> dict[str, float]:
    sigma_buy = math.sqrt(max(0.0, float(getattr(model, "_sigma2_buy", 1.0))))
    sigma_sell = math.sqrt(max(0.0, float(getattr(model, "_sigma2_sell", 1.0))))
    sigma_min = float(getattr(model, "_sigma_min", 1e-6))
    sigma_buy = max(sigma_min, sigma_buy)
    sigma_sell = max(sigma_min, sigma_sell)
    mu_buy = float(getattr(model, "_mu_buy", 0.0))
    mu_sell = float(getattr(model, "_mu_sell", 0.0))
    mu_mid = 0.5 * (mu_buy + mu_sell)
    sigma_mid = math.sqrt(max(0.0, 0.5 * (sigma_buy * sigma_buy + sigma_sell * sigma_sell)))
    return {
        "mu_mid": mu_mid,
        "sigma_mid": max(sigma_min, sigma_mid),
        "mu_buy": mu_buy,
        "mu_sell": mu_sell,
        "sigma_buy": sigma_buy,
        "sigma_sell": sigma_sell,
    }


def score_hedge_vwap_deltas(
    *,
    delta_buy: float,
    delta_sell: float,
    params_before: dict[str, Any],
) -> ScoreResult:
    """Gaussian NLL for hedge buy/sell VWAP deltas (independent sides)."""

    mu_b = float(params_before.get("mu_delta_buy", 0.0))
    mu_s = float(params_before.get("mu_delta_sell", 0.0))
    sig_b = float(params_before.get("sigma_delta_buy", params_before.get("sigma_buy", 1e-6)))
    sig_s = float(params_before.get("sigma_delta_sell", params_before.get("sigma_sell", 1e-6)))
    if not math.isfinite(sig_b) or sig_b <= 0.0 or not math.isfinite(sig_s) or sig_s <= 0.0:
        return ScoreResult(
            skip_reason="invalid_sigma",
            observation={
                "delta_buy": float(delta_buy),
                "delta_sell": float(delta_sell),
                "mu_delta_buy": mu_b,
                "mu_delta_sell": mu_s,
                "sigma_delta_buy": sig_b,
                "sigma_delta_sell": sig_s,
            },
        )
    ll = _gaussian_log_pdf(delta_buy, mu_b, sig_b) + _gaussian_log_pdf(delta_sell, mu_s, sig_s)
    nll = -ll
    z_b = (float(delta_buy) - mu_b) / sig_b
    z_s = (float(delta_sell) - mu_s) / sig_s
    z = 0.5 * (z_b + z_s)
    return ScoreResult(
        log_likelihood=ll,
        nll=nll,
        observation={
            "delta_buy": float(delta_buy),
            "delta_sell": float(delta_sell),
            "mu_delta_buy": mu_b,
            "mu_delta_sell": mu_s,
            "sigma_delta_buy": sig_b,
            "sigma_delta_sell": sig_s,
        },
        standardized_residual=float(z),
    )


def _to_model_score(
    *,
    model: str,
    kind: str,
    params_before: dict[str, Any],
    params_after: dict[str, Any],
    result: ScoreResult,
    updated: bool,
) -> ModelScore:
    return ModelScore(
        model=model,
        kind=kind,
        updated=updated,
        skip_reason=result.skip_reason,
        observation=dict(result.observation),
        params_before=dict(params_before),
        params_after=dict(params_after),
        log_likelihood=result.log_likelihood,
        nll=result.nll,
        standardized_residual=result.standardized_residual,
    )


def score_frame_models(
    *,
    frame: StrategyInputFrame,
    previous_frame: StrategyInputFrame | None,
    state: State,
    market_flow_model: Any | None,
    price_model: Any | None,
    global_price_model: Any | None,
    market_flow_kind: str,
    price_kind: str,
    global_price_kind: str,
) -> dict[str, ModelScore]:
    """Score realized observations with pre-update params; caller runs updates after."""

    scores: dict[str, ModelScore] = {}

    if market_flow_model is not None and hasattr(market_flow_model, "get_params"):
        params_before = dict(market_flow_model.get_params())
        result = score_market_flow(
            buy_volume=frame.local.buy_volume,
            sell_volume=frame.local.sell_volume,
            params_before=params_before,
            kind=market_flow_kind,
        )
        scores["market_flow"] = _to_model_score(
            model="market_flow",
            kind=market_flow_kind,
            params_before=params_before,
            params_after=params_before,
            result=result,
            updated=False,
        )

    if price_model is not None:
        params_before = _price_params_before(price_model)
        if previous_frame is None:
            result = ScoreResult(
                skip_reason="missing_previous_frame",
                observation={"delta_mid": None},
            )
        else:
            from mm.core.order_constraints import mid_from_orderbook

            try:
                prev_mid = mid_from_orderbook(previous_frame.local.orderbook)
                curr_mid = mid_from_orderbook(frame.local.orderbook)
            except ValueError:
                result = ScoreResult(skip_reason="missing_previous_frame", observation={})
            else:
                delta = float(curr_mid - prev_mid)
                result = score_primary_mid_delta(delta_mid=delta, params_before=params_before)
        scores["price"] = _to_model_score(
            model="price",
            kind=price_kind,
            params_before=params_before,
            params_after=params_before,
            result=result,
            updated=False,
        )

    if global_price_model is not None and hasattr(global_price_model, "get_params"):
        params_before = dict(global_price_model.get_params())
        if state.global_buy_vwap is None or state.global_sell_vwap is None or not frame.has_global:
            result = ScoreResult(skip_reason="missing_hedge", observation={})
        elif getattr(global_price_model, "_last_buy_obs", None) is None:
            result = ScoreResult(skip_reason="missing_previous_frame", observation={})
        else:
            delta_buy = float(state.global_buy_vwap) - float(global_price_model._last_buy_obs)
            delta_sell = float(state.global_sell_vwap) - float(global_price_model._last_sell_obs)
            if global_price_kind == "gaussian":
                result = score_hedge_vwap_deltas(
                    delta_buy=delta_buy,
                    delta_sell=delta_sell,
                    params_before={
                        "mu_delta_buy": 0.0,
                        "mu_delta_sell": 0.0,
                        "sigma_delta_buy": params_before.get("sigma_buy", 0.0),
                        "sigma_delta_sell": params_before.get("sigma_sell", 0.0),
                    },
                )
            else:
                result = score_hedge_vwap_deltas(
                    delta_buy=delta_buy,
                    delta_sell=delta_sell,
                    params_before=params_before,
                )
        scores["global_price"] = _to_model_score(
            model="global_price",
            kind=global_price_kind,
            params_before=params_before,
            params_after=params_before,
            result=result,
            updated=False,
        )

    return scores


def refresh_score_params_after(
    scores: dict[str, ModelScore],
    *,
    key: str,
    params_after: dict[str, Any],
    updated: bool,
) -> dict[str, ModelScore]:
    """Replace one score entry with post-update parameters."""

    prev = scores[key]
    scores[key] = ModelScore(
        model=prev.model,
        kind=prev.kind,
        updated=updated,
        skip_reason=prev.skip_reason,
        observation=prev.observation,
        params_before=prev.params_before,
        params_after=dict(params_after),
        log_likelihood=prev.log_likelihood,
        nll=prev.nll,
        standardized_residual=prev.standardized_residual,
    )
    return scores
