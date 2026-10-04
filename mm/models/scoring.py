"""Pre-update negative log-likelihood scoring for controller diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mm.mpc_utils import (
    expit as _expit,
    log_clamped as _log_clamped,
    logaddexp as _logaddexp,
    log_normal_pdf as _log_normal_pdf,
    log_student_t_pdf as _log_student_t_pdf,
)
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
    ll, nll = _nll_from_log_prob(_log_clamped(prob))
    return ScoreResult(
        log_likelihood=ll,
        nll=nll,
        observation={**observation, "prob_buy": prob_buy, "prob_sell": prob_sell},
    )


def _gaussian_log_pdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(1e-12, float(sigma))
    z = (float(x) - float(mu)) / sigma
    return -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * z * z


def score_local_mid_delta(
    *,
    delta_mid: float,
    params_before: dict[str, Any],
) -> ScoreResult:
    """Gaussian NLL for realized local mid change (residual / EWMA price models)."""

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


_MIXTURE_PRICE_KINDS = frozenset(
    {"conditional_mechanical_mixture", "flow_gated_mixture"}
)


def _is_mixture_price_kind(kind: str, params: dict[str, Any] | None = None) -> bool:
    kind_l = str(kind or "").strip().lower()
    if kind_l in _MIXTURE_PRICE_KINDS:
        return True
    if params is None:
        return False
    return str(params.get("kind", "")).strip().lower() in _MIXTURE_PRICE_KINDS


def _normal_logpdf_var(error: float, variance: float, *, sigma_min: float = 1e-6) -> float:
    var = max(sigma_min * sigma_min, float(variance))
    return -0.5 * (math.log(2.0 * math.pi * var) + (error * error) / var)


def _student_t_logpdf_var(
    error: float,
    variance: float,
    *,
    student_df: float,
    sigma_min: float = 1e-6,
) -> float:
    nu = max(2.1, float(student_df))
    var = max(sigma_min * sigma_min, float(variance))
    scale2 = max(sigma_min * sigma_min, var * (nu - 2.0) / nu)
    return (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi * scale2)
        - ((nu + 1.0) / 2.0) * math.log1p((error * error) / (nu * scale2))
    )


def _mixture_market_probability(
    gate_beta: Any,
    total_flow: float,
    *,
    min_refresh_weight: float = 0.15,
) -> float:
    beta = list(gate_beta) if gate_beta is not None else [0.0, 0.0]
    b0 = float(beta[0]) if len(beta) > 0 else 0.0
    b1 = float(beta[1]) if len(beta) > 1 else 0.0
    log_flow = min(math.log1p(max(0.0, float(total_flow))), 12.0)
    score = b0 + b1 * log_flow
    if not math.isfinite(score):
        return 0.5
    # Match TorchConditionalMechanicalMixturePriceModel._market_probability_np.
    max_market = max(0.001, 1.0 - max(0.0, min(0.999, float(min_refresh_weight))))
    return min(max_market, max(0.001, _expit(score)))


def score_conditional_mechanical_mid_delta(
    *,
    delta_mid: float,
    mech_mid_delta: float,
    total_flow: float,
    params_before: dict[str, Any],
) -> ScoreResult:
    """
    Mixture NLL for realized mid change under conditional_mechanical_mixture.

    Predictive density (pre-update params):
      q · Normal(μ_mkt, σ_mkt²) + (1−q) · StudentT(μ_ref, σ_ref², ν)
    where μ_mkt = β₀ + β₁ · mech_mid_delta and q is the flow gate.
    """

    sigma_min = 1e-6
    market_beta = params_before.get("market_beta") or {}
    mid_beta = market_beta.get("mid") if isinstance(market_beta, dict) else None
    if mid_beta is None:
        mid_beta = [0.0, 0.0]
    b0 = float(mid_beta[0]) if len(mid_beta) > 0 else 0.0
    b1 = float(mid_beta[1]) if len(mid_beta) > 1 else 0.0
    market_mean = b0 + b1 * float(mech_mid_delta)

    refresh_mean_map = params_before.get("refresh_mean") or {}
    refresh_mean = (
        float(refresh_mean_map.get("mid", 0.0))
        if isinstance(refresh_mean_map, dict)
        else 0.0
    )

    market_sigma_map = params_before.get("market_sigma") or {}
    refresh_sigma_map = params_before.get("refresh_sigma") or {}
    market_sigma = (
        float(market_sigma_map.get("mid", 50.0))
        if isinstance(market_sigma_map, dict)
        else 50.0
    )
    refresh_sigma = (
        float(refresh_sigma_map.get("mid", 50.0))
        if isinstance(refresh_sigma_map, dict)
        else 50.0
    )
    if not math.isfinite(market_sigma) or market_sigma <= 0.0:
        return ScoreResult(
            skip_reason="invalid_sigma",
            observation={
                "delta_mid": float(delta_mid),
                "mech_mid_delta": float(mech_mid_delta),
                "market_sigma_mid": market_sigma,
            },
        )
    if not math.isfinite(refresh_sigma) or refresh_sigma <= 0.0:
        return ScoreResult(
            skip_reason="invalid_sigma",
            observation={
                "delta_mid": float(delta_mid),
                "mech_mid_delta": float(mech_mid_delta),
                "refresh_sigma_mid": refresh_sigma,
            },
        )

    student_df = float(params_before.get("student_df", 3.0))
    min_refresh_weight = float(params_before.get("min_refresh_weight", 0.15))
    q_market = _mixture_market_probability(
        params_before.get("gate_beta"),
        total_flow,
        min_refresh_weight=min_refresh_weight,
    )
    q_refresh = 1.0 - q_market

    market_err = float(delta_mid) - market_mean
    refresh_err = float(delta_mid) - refresh_mean
    market_ll = _log_clamped(max(_MIN_PROB, q_market)) + _normal_logpdf_var(
        market_err, market_sigma * market_sigma, sigma_min=sigma_min
    )
    refresh_ll = _log_clamped(max(_MIN_PROB, q_refresh)) + _student_t_logpdf_var(
        refresh_err,
        refresh_sigma * refresh_sigma,
        student_df=student_df,
        sigma_min=sigma_min,
    )
    # log(q N + (1-q) T) via log-sum-exp
    ll = _logaddexp(market_ll, refresh_ll)
    nll = -ll

    # Mixture mean / variance for a standardized residual diagnostic.
    mixture_mean = q_market * market_mean + q_refresh * refresh_mean
    # Student-t with ν>2 has variance = scale² * ν/(ν-2); model stores that variance in refresh_var.
    mixture_var = (
        q_market * (market_sigma * market_sigma + (market_mean - mixture_mean) ** 2)
        + q_refresh * (refresh_sigma * refresh_sigma + (refresh_mean - mixture_mean) ** 2)
    )
    mixture_sigma = math.sqrt(max(sigma_min * sigma_min, mixture_var))
    z = (float(delta_mid) - mixture_mean) / mixture_sigma

    return ScoreResult(
        log_likelihood=ll,
        nll=nll,
        observation={
            "delta_mid": float(delta_mid),
            "mech_mid_delta": float(mech_mid_delta),
            "total_flow": float(total_flow),
            "mu_mid_before": float(mixture_mean),
            "sigma_mid_before": float(mixture_sigma),
            "market_prob": float(q_market),
            "market_mean_mid": float(market_mean),
            "refresh_mean_mid": float(refresh_mean),
            "market_sigma_mid": float(market_sigma),
            "refresh_sigma_mid": float(refresh_sigma),
        },
        standardized_residual=float(z),
    )


def _mechanical_mid_delta_from_frames(
    previous_frame: StrategyInputFrame,
    frame: StrategyInputFrame,
    *,
    price_model: Any | None = None,
) -> float | None:
    """Mechanical mid move from MOF-only consumption of the previous book."""

    from types import SimpleNamespace

    from mm.core.order_constraints import mid_from_orderbook

    prev_ob = previous_frame.local.orderbook
    buy_vol = float(frame.local.buy_volume)
    sell_vol = float(frame.local.sell_volume)
    try:
        prev_mid = float(mid_from_orderbook(prev_ob))
    except (AttributeError, TypeError, ValueError):
        return None
    # Price-model helpers expect a mid-bearing book view (see mpc_engine._orderbook_with_mid).
    book_view = SimpleNamespace(bids=prev_ob.bids, asks=prev_ob.asks, mid=prev_mid)
    if hasattr(price_model, "_transition_mechanical_best"):
        try:
            mech_bid, mech_ask = price_model._transition_mechanical_best(
                book_view, buy_vol, sell_vol
            )
            return 0.5 * (float(mech_bid) + float(mech_ask)) - prev_mid
        except Exception:
            pass
    try:
        from mm.pricing.reference_prices_ticks import _consume_depth, _depth_by_price

        bid_prev = _depth_by_price(book_view, "bid")
        ask_prev = _depth_by_price(book_view, "ask")
        bid_after = _consume_depth(bid_prev, sell_vol, best_first_asc=False)
        ask_after = _consume_depth(ask_prev, buy_vol, best_first_asc=True)
        mech_bid = float(max(bid_after.keys())) if bid_after else prev_mid
        mech_ask = float(min(ask_after.keys())) if ask_after else prev_mid
        return 0.5 * (mech_bid + mech_ask) - prev_mid
    except Exception:
        return None


def _price_params_before(model: Any) -> dict[str, Any]:
    """Extract scoreable pre-update params; prefer ``get_params()`` when present."""

    if hasattr(model, "get_params"):
        try:
            params = dict(model.get_params())
        except Exception:
            params = {}
        if params:
            # Residual / EWMA-style models already expose mu_mid / sigma_mid.
            if "mu_mid" in params and "sigma_mid" in params:
                return {
                    "mu_mid": float(params["mu_mid"]),
                    "sigma_mid": float(params["sigma_mid"]),
                    "mu_buy": float(params.get("mu_buy", params["mu_mid"])),
                    "mu_sell": float(params.get("mu_sell", params["mu_mid"])),
                    "sigma_buy": float(params.get("sigma_buy", params["sigma_mid"])),
                    "sigma_sell": float(params.get("sigma_sell", params["sigma_mid"])),
                    "kind": params.get("kind"),
                }
            if _is_mixture_price_kind(str(params.get("kind", "")), params):
                return params

    # Legacy fallback for residual models that expose private EWMA state only.
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


def score_global_vwap_deltas(
    *,
    delta_buy: float,
    delta_sell: float,
    params_before: dict[str, Any],
) -> ScoreResult:
    """Gaussian NLL for global buy/sell VWAP deltas (independent sides)."""

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
    local_market_flow_model: Any | None,
    local_price_model: Any | None,
    global_market_flow_model: Any | None,
    global_price_model: Any | None,
    local_market_flow_kind: str,
    local_price_kind: str,
    global_market_flow_kind: str,
    global_price_kind: str,
) -> dict[str, ModelScore]:
    """Score realized observations with pre-update params; caller runs updates after."""

    scores: dict[str, ModelScore] = {}

    if local_market_flow_model is not None and hasattr(local_market_flow_model, "get_params"):
        params_before = dict(local_market_flow_model.get_params())
        result = score_market_flow(
            buy_volume=frame.local.buy_volume,
            sell_volume=frame.local.sell_volume,
            params_before=params_before,
            kind=local_market_flow_kind,
        )
        scores["local_market_flow"] = _to_model_score(
            model="local_market_flow",
            kind=local_market_flow_kind,
            params_before=params_before,
            params_after=params_before,
            result=result,
            updated=False,
        )

    if local_price_model is not None:
        params_before = _price_params_before(local_price_model)
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
                if _is_mixture_price_kind(local_price_kind, params_before):
                    mech_mid_delta = _mechanical_mid_delta_from_frames(
                        previous_frame,
                        frame,
                        price_model=local_price_model,
                    )
                    if mech_mid_delta is None:
                        result = ScoreResult(
                            skip_reason="missing_mechanical_mid",
                            observation={"delta_mid": delta},
                        )
                    else:
                        total_flow = max(
                            0.0,
                            float(frame.local.buy_volume) + float(frame.local.sell_volume),
                        )
                        result = score_conditional_mechanical_mid_delta(
                            delta_mid=delta,
                            mech_mid_delta=float(mech_mid_delta),
                            total_flow=total_flow,
                            params_before=params_before,
                        )
                else:
                    result = score_local_mid_delta(delta_mid=delta, params_before=params_before)
        scores["local_price"] = _to_model_score(
            model="local_price",
            kind=local_price_kind,
            params_before=params_before,
            params_after=params_before,
            result=result,
            updated=False,
        )

    if global_market_flow_model is not None and hasattr(global_market_flow_model, "get_params"):
        params_before = dict(global_market_flow_model.get_params())
        if frame.global_ is None or not frame.has_global:
            result = ScoreResult(skip_reason="missing_global", observation={})
        else:
            result = score_market_flow(
                buy_volume=frame.global_.buy_volume,
                sell_volume=frame.global_.sell_volume,
                params_before=params_before,
                kind=global_market_flow_kind,
            )
        scores["global_market_flow"] = _to_model_score(
            model="global_market_flow",
            kind=global_market_flow_kind,
            params_before=params_before,
            params_after=params_before,
            result=result,
            updated=False,
        )

    if global_price_model is not None and hasattr(global_price_model, "get_params"):
        params_before = dict(global_price_model.get_params())
        if state.global_buy_vwap is None or state.global_sell_vwap is None or not frame.has_global:
            result = ScoreResult(skip_reason="missing_global", observation={})
        elif getattr(global_price_model, "_last_buy_obs", None) is None:
            result = ScoreResult(skip_reason="missing_previous_frame", observation={})
        else:
            delta_buy = float(state.global_buy_vwap) - float(global_price_model._last_buy_obs)
            delta_sell = float(state.global_sell_vwap) - float(global_price_model._last_sell_obs)
            if global_price_kind == "gaussian":
                result = score_global_vwap_deltas(
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
                result = score_global_vwap_deltas(
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
