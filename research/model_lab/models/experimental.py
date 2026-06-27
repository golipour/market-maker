"""Experimental online models for research-only walk-forward comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mm.models.scoring import _hurdle_lognormal_pmf
from mm.mpc.market_order_flow_torch import TorchHurdleLogStudentTModel

from research.model_lab.core.types import IntervalStep, StepScore, mpc_step_from_interval


_MIN_PROB = 1e-300


def _student_t_log_pdf(x: float, mu: float, sigma: float, df: float) -> float:
    """Log density for Student-t parameterized by target standard deviation."""

    nu = max(2.1, float(df))
    sigma = max(1e-12, float(sigma))
    scale = sigma * math.sqrt((nu - 2.0) / nu)
    z = (float(x) - float(mu)) / scale
    return (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
        - math.log(scale)
        - ((nu + 1.0) / 2.0) * math.log1p((z * z) / nu)
    )


@dataclass
class HurdleLogStudentTFlowModel:
    """Hurdle flow model with log-Student-t positive-volume tails."""

    decay: float = 0.95
    init_p0: float = 0.3
    init_mu: float = 0.0
    init_sigma: float = 1.0
    init_df: float = 5.0
    name: str = "flow_hurdle_log_student_t"
    kind: str = "hurdle_log_student_t"
    family: str = "flow"
    _model: TorchHurdleLogStudentTModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._model = TorchHurdleLogStudentTModel(
            decay=self.decay,
            init_p0=self.init_p0,
            init_mu=self.init_mu,
            init_sigma=self.init_sigma,
            init_df=self.init_df,
        )

    def get_params(self) -> dict[str, Any]:
        return dict(self._model.get_params())

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        params = self.get_params()
        buy = max(0, int(round(float(current.buy_volume))))
        sell = max(0, int(round(float(current.sell_volume))))
        df = float(params.get("logt_df", self.init_df))
        prob_buy = TorchHurdleLogStudentTModel.pmf(
            buy,
            float(params.get("p0_buy", self.init_p0)),
            df,
            float(params.get("mu_buy", self.init_mu)),
            float(params.get("sigma_buy", self.init_sigma)),
        )
        prob_sell = TorchHurdleLogStudentTModel.pmf(
            sell,
            float(params.get("p0_sell", self.init_p0)),
            df,
            float(params.get("mu_sell", self.init_mu)),
            float(params.get("sigma_sell", self.init_sigma)),
        )
        ll = math.log(max(_MIN_PROB, float(prob_buy) * float(prob_sell)))
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=ll,
            nll=-ll,
            observation={
                "buy_volume": buy,
                "sell_volume": sell,
                "prob_buy": prob_buy,
                "prob_sell": prob_sell,
            },
            params_before=params,
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        self._model.update(mpc_step_from_interval(previous), mpc_step_from_interval(current))
        return True


@dataclass
class BivariateHurdleLogNormalFlowModel:
    """Joint hurdle pattern plus lognormal positive-volume sizes.

    This removes the production independence assumption for zero/non-zero
    co-occurrence while keeping tractable per-side positive-size likelihoods.
    """

    decay: float = 0.95
    init_mu: float = 0.0
    init_sigma: float = 1.0
    init_p00: float = 0.5
    init_p10: float = 0.2
    init_p01: float = 0.2
    init_p11: float = 0.1
    param_eps: float = 1e-6
    name: str = "flow_bivariate_hurdle_lognormal"
    kind: str = "bivariate_hurdle_lognormal"
    family: str = "flow"
    _pattern_probs: dict[str, float] = field(init=False, repr=False)
    _s_log_buy: float = field(default=0.0, init=False, repr=False)
    _s2_log_buy: float = field(default=0.0, init=False, repr=False)
    _w_buy: float = field(default=0.0, init=False, repr=False)
    _s_log_sell: float = field(default=0.0, init=False, repr=False)
    _s2_log_sell: float = field(default=0.0, init=False, repr=False)
    _w_sell: float = field(default=0.0, init=False, repr=False)
    _mu_buy: float = field(default=0.0, init=False, repr=False)
    _sigma_buy: float = field(default=1.0, init=False, repr=False)
    _mu_sell: float = field(default=0.0, init=False, repr=False)
    _sigma_sell: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        probs = {
            "00": max(0.0, float(self.init_p00)),
            "10": max(0.0, float(self.init_p10)),
            "01": max(0.0, float(self.init_p01)),
            "11": max(0.0, float(self.init_p11)),
        }
        total = sum(probs.values()) or 1.0
        self._pattern_probs = {key: value / total for key, value in probs.items()}
        self._s_log_buy = self._s2_log_buy = self._w_buy = 0.0
        self._s_log_sell = self._s2_log_sell = self._w_sell = 0.0
        self._mu_buy = float(self.init_mu)
        self._mu_sell = float(self.init_mu)
        self._sigma_buy = max(float(self.param_eps), float(self.init_sigma))
        self._sigma_sell = max(float(self.param_eps), float(self.init_sigma))

    def get_params(self) -> dict[str, Any]:
        return {
            "pattern_probs": dict(self._pattern_probs),
            "mu_buy": self._mu_buy,
            "sigma_buy": self._sigma_buy,
            "mu_sell": self._mu_sell,
            "sigma_sell": self._sigma_sell,
        }

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        params = self.get_params()
        buy = max(0, int(round(float(current.buy_volume))))
        sell = max(0, int(round(float(current.sell_volume))))
        pattern = self._pattern_for(buy, sell)
        pattern_prob = max(_MIN_PROB, self._pattern_probs.get(pattern, 0.0))
        prob = pattern_prob
        if buy > 0:
            prob *= max(_MIN_PROB, _hurdle_lognormal_pmf(buy, 0.0, self._mu_buy, self._sigma_buy))
        if sell > 0:
            prob *= max(_MIN_PROB, _hurdle_lognormal_pmf(sell, 0.0, self._mu_sell, self._sigma_sell))
        ll = math.log(max(_MIN_PROB, prob))
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=ll,
            nll=-ll,
            observation={
                "buy_volume": buy,
                "sell_volume": sell,
                "pattern": pattern,
                "pattern_prob": pattern_prob,
            },
            params_before=params,
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        buy = max(0, int(round(float(current.buy_volume))))
        sell = max(0, int(round(float(current.sell_volume))))
        pattern = self._pattern_for(buy, sell)
        gamma = max(1e-6, min(1.0, float(self.decay)))
        alpha = 1.0 - gamma
        for key in tuple(self._pattern_probs):
            self._pattern_probs[key] = gamma * self._pattern_probs[key] + alpha * (1.0 if key == pattern else 0.0)
        total = sum(self._pattern_probs.values()) or 1.0
        self._pattern_probs = {key: max(_MIN_PROB, value / total) for key, value in self._pattern_probs.items()}
        self._update_positive_side("buy", buy, gamma)
        self._update_positive_side("sell", sell, gamma)
        return True

    @staticmethod
    def _pattern_for(buy: int, sell: int) -> str:
        return ("1" if buy > 0 else "0") + ("1" if sell > 0 else "0")

    def _update_positive_side(self, side: str, value: int, gamma: float) -> None:
        if value <= 0:
            return
        log_x = math.log(max(1e-12, float(value)))
        if side == "buy":
            self._s_log_buy = gamma * self._s_log_buy + log_x
            self._s2_log_buy = gamma * self._s2_log_buy + log_x * log_x
            self._w_buy = gamma * self._w_buy + 1.0
            mean = self._s_log_buy / max(1e-12, self._w_buy)
            var = max(self.param_eps * self.param_eps, self._s2_log_buy / max(1e-12, self._w_buy) - mean * mean)
            self._mu_buy = float(mean)
            self._sigma_buy = float(math.sqrt(var))
            return
        self._s_log_sell = gamma * self._s_log_sell + log_x
        self._s2_log_sell = gamma * self._s2_log_sell + log_x * log_x
        self._w_sell = gamma * self._w_sell + 1.0
        mean = self._s_log_sell / max(1e-12, self._w_sell)
        var = max(self.param_eps * self.param_eps, self._s2_log_sell / max(1e-12, self._w_sell) - mean * mean)
        self._mu_sell = float(mean)
        self._sigma_sell = float(math.sqrt(var))


@dataclass
class StudentTMidDeltaModel:
    """EWMA Student-t model for one-step mid-price deltas."""

    decay: float = 0.99
    initial_sigma: float = 50.0
    df: float = 5.0
    name: str = "price_student_t_mid_delta"
    kind: str = "student_t_mid_delta"
    family: str = "price"
    _mu: float = field(default=0.0, init=False, repr=False)
    _sigma2: float = field(default=2500.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mu = 0.0
        sigma = max(1e-6, float(self.initial_sigma))
        self._sigma2 = sigma * sigma

    def get_params(self) -> dict[str, Any]:
        return {"mu_mid": self._mu, "sigma_mid": math.sqrt(max(1e-12, self._sigma2)), "df": self.df}

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        if previous is None:
            return StepScore(
                step_index=-1,
                ts=current.ts,
                model_name=self.name,
                model_kind=self.kind,
                family=self.family,
                skip_reason="missing_previous_frame",
            )
        delta = float(current.mid - previous.mid)
        params = self.get_params()
        sigma = float(params["sigma_mid"])
        ll = _student_t_log_pdf(delta, float(params["mu_mid"]), sigma, self.df)
        z = (delta - float(params["mu_mid"])) / max(1e-12, sigma)
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=ll,
            nll=-ll,
            standardized_residual=z,
            observation={"delta_mid": delta, "mu_mid_before": params["mu_mid"], "sigma_mid_before": sigma},
            params_before=params,
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        delta = float(current.mid - previous.mid)
        gamma = max(1e-6, min(1.0, float(self.decay)))
        alpha = 1.0 - gamma
        mu_new = gamma * self._mu + alpha * delta
        self._sigma2 = gamma * self._sigma2 + alpha * (delta - mu_new) ** 2
        self._mu = float(mu_new)
        return True


@dataclass
class FlowConditionedMidDeltaModel:
    """Online RLS Gaussian model for mid deltas using lagged flow state."""

    decay: float = 0.99
    variance_decay: float = 0.99
    initial_sigma: float = 50.0
    rls_p0: float = 1_000_000.0
    name: str = "price_flow_conditioned_mid_delta"
    kind: str = "flow_conditioned_mid_delta"
    family: str = "price"
    _beta: np.ndarray = field(init=False, repr=False)
    _p: np.ndarray = field(init=False, repr=False)
    _sigma2: float = field(default=2500.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._beta = np.zeros(4, dtype=np.float64)
        self._p = np.eye(4, dtype=np.float64) * float(self.rls_p0)
        sigma = max(1e-6, float(self.initial_sigma))
        self._sigma2 = sigma * sigma

    def get_params(self) -> dict[str, Any]:
        return {"beta": self._beta.tolist(), "sigma_mid": math.sqrt(max(1e-12, self._sigma2))}

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        if previous is None:
            return StepScore(
                step_index=-1,
                ts=current.ts,
                model_name=self.name,
                model_kind=self.kind,
                family=self.family,
                skip_reason="missing_previous_frame",
            )
        x = self._features(previous)
        mu = float(x @ self._beta)
        sigma = math.sqrt(max(1e-12, self._sigma2))
        delta = float(current.mid - previous.mid)
        ll = -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * ((delta - mu) / sigma) ** 2
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=ll,
            nll=-ll,
            standardized_residual=(delta - mu) / sigma,
            observation={"delta_mid": delta, "mu_mid_before": mu, "features": x.tolist()},
            params_before=self.get_params(),
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        x = self._features(previous)
        y = float(current.mid - previous.mid)
        gamma = max(1e-6, min(1.0, float(self.decay)))
        px = self._p @ x
        gain = px / (gamma + float(x @ px))
        residual = y - float(x @ self._beta)
        self._beta = self._beta + gain * residual
        self._p = (self._p - np.outer(gain, x) @ self._p) / gamma
        self._p = 0.5 * (self._p + self._p.T)
        var_gamma = max(1e-6, min(1.0, float(self.variance_decay)))
        self._sigma2 = var_gamma * self._sigma2 + (1.0 - var_gamma) * residual * residual
        return True

    @staticmethod
    def _features(step: IntervalStep) -> np.ndarray:
        total = max(0.0, float(step.total_volume))
        signed = 0.0 if total <= 0.0 else (float(step.buy_volume) - float(step.sell_volume)) / total
        spread_bps = 0.0 if step.mid <= 0.0 else 10_000.0 * float(step.spread) / float(step.mid)
        return np.array([1.0, math.log1p(total), signed, spread_bps], dtype=np.float64)


def _book_imbalance(step: IntervalStep) -> float:
    bids = getattr(step.orderbook, "bids", ()) or ()
    asks = getattr(step.orderbook, "asks", ()) or ()
    bid_depth = sum(float(getattr(level, "qty", 0.0)) for level in bids[:5])
    ask_depth = sum(float(getattr(level, "qty", 0.0)) for level in asks[:5])
    total = bid_depth + ask_depth
    if total <= 0.0:
        return 0.0
    return (bid_depth - ask_depth) / total


def _gaussian_log_pdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(1e-12, float(sigma))
    z = (float(x) - float(mu)) / sigma
    return -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * z * z


@dataclass
class StateConditionedHurdleFlowModel:
    """Hurdle lognormal with logistic gate on spread and book imbalance."""

    decay: float = 0.95
    init_p0: float = 0.3
    init_mu: float = 0.0
    init_sigma: float = 1.0
    gate_decay: float = 0.99
    name: str = "flow_state_conditioned_hurdle"
    kind: str = "state_conditioned_hurdle"
    family: str = "flow"
    _base: Any = field(init=False, repr=False)
    _gate_weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        from mm.mpc.market_order_flow_torch import TorchHurdleLogNormalModel

        self._base = TorchHurdleLogNormalModel(
            decay=self.decay,
            init_p0=self.init_p0,
            init_mu=self.init_mu,
            init_sigma=self.init_sigma,
        )
        self._gate_weights = np.zeros(3, dtype=np.float64)

    def get_params(self) -> dict[str, Any]:
        params = dict(self._base.get_params())
        params["gate_weights"] = self._gate_weights.tolist()
        return params

    def _gate_features(self, step: IntervalStep) -> np.ndarray:
        spread_bps = 0.0 if step.mid <= 0.0 else 10_000.0 * float(step.spread) / float(step.mid)
        return np.array([1.0, spread_bps, _book_imbalance(step)], dtype=np.float64)

    def _adjusted_p0(self, base_p0: float, features: np.ndarray) -> float:
        logit = float(features @ self._gate_weights)
        scale = 1.0 / (1.0 + math.exp(-logit))
        return max(1e-6, min(1.0 - 1e-6, float(base_p0) * (0.5 + scale)))

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        params = self._base.get_params()
        features = self._gate_features(previous) if previous is not None else np.array([1.0, 0.0, 0.0])
        buy = max(0, int(round(float(current.buy_volume))))
        sell = max(0, int(round(float(current.sell_volume))))
        p0_buy = self._adjusted_p0(float(params.get("p0_buy", self.init_p0)), features)
        p0_sell = self._adjusted_p0(float(params.get("p0_sell", self.init_p0)), features)
        prob_buy = max(_MIN_PROB, _hurdle_lognormal_pmf(buy, p0_buy, float(params.get("mu_buy", self.init_mu)), float(params.get("sigma_buy", self.init_sigma))))
        prob_sell = max(_MIN_PROB, _hurdle_lognormal_pmf(sell, p0_sell, float(params.get("mu_sell", self.init_mu)), float(params.get("sigma_sell", self.init_sigma))))
        ll = math.log(max(_MIN_PROB, prob_buy * prob_sell))
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=ll,
            nll=-ll,
            observation={"buy_volume": buy, "sell_volume": sell, "p0_buy": p0_buy, "p0_sell": p0_sell},
            params_before=self.get_params(),
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        self._base.update(mpc_step_from_interval(previous), mpc_step_from_interval(current))
        features = self._gate_features(previous)
        buy_zero = 1.0 if float(current.buy_volume) <= 0.0 else 0.0
        sell_zero = 1.0 if float(current.sell_volume) <= 0.0 else 0.0
        target = 0.5 * (buy_zero + sell_zero)
        gamma = max(1e-6, min(1.0, float(self.gate_decay)))
        pred = 1.0 / (1.0 + math.exp(-float(features @ self._gate_weights)))
        grad = (pred - target) * features
        self._gate_weights = gamma * self._gate_weights - (1.0 - gamma) * grad
        return True


@dataclass
class AR1MidDeltaModel:
    """AR(1) mean on mid deltas with EWMA innovation variance."""

    decay: float = 0.99
    phi_decay: float = 0.99
    initial_sigma: float = 50.0
    name: str = "price_ar1_mid_delta"
    kind: str = "ar1_mid_delta"
    family: str = "price"
    _phi: float = field(default=0.0, init=False, repr=False)
    _prev_delta: float = field(default=0.0, init=False, repr=False)
    _sigma2: float = field(default=2500.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phi = 0.0
        self._prev_delta = 0.0
        sigma = max(1e-6, float(self.initial_sigma))
        self._sigma2 = sigma * sigma

    def get_params(self) -> dict[str, Any]:
        return {"phi": self._phi, "sigma_mid": math.sqrt(max(1e-12, self._sigma2))}

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        if previous is None:
            return StepScore(step_index=-1, ts=current.ts, model_name=self.name, model_kind=self.kind, family=self.family, skip_reason="missing_previous_frame")
        delta = float(current.mid - previous.mid)
        mu = self._phi * self._prev_delta
        sigma = math.sqrt(max(1e-12, self._sigma2))
        ll = _gaussian_log_pdf(delta, mu, sigma)
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=ll,
            nll=-ll,
            standardized_residual=(delta - mu) / sigma,
            observation={"delta_mid": delta, "mu_mid_before": mu},
            params_before=self.get_params(),
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        delta = float(current.mid - previous.mid)
        mu = self._phi * self._prev_delta
        residual = delta - mu
        gamma = max(1e-6, min(1.0, float(self.decay)))
        self._sigma2 = gamma * self._sigma2 + (1.0 - gamma) * residual * residual
        phi_gamma = max(1e-6, min(1.0, float(self.phi_decay)))
        denom = max(1e-12, self._prev_delta * self._prev_delta)
        phi_obs = (self._prev_delta * delta) / denom if abs(self._prev_delta) > 1e-9 else 0.0
        self._phi = phi_gamma * self._phi + (1.0 - phi_gamma) * phi_obs
        self._phi = max(-0.99, min(0.99, self._phi))
        self._prev_delta = delta
        return True


@dataclass
class GARCHMidDeltaModel:
    """Online GARCH(1,1) volatility on mid deltas."""

    omega: float = 1.0
    alpha: float = 0.05
    beta: float = 0.90
    initial_sigma: float = 50.0
    name: str = "price_garch_mid_delta"
    kind: str = "garch_mid_delta"
    family: str = "price"
    _sigma2: float = field(default=2500.0, init=False, repr=False)
    _prev_eps: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        sigma = max(1e-6, float(self.initial_sigma))
        self._sigma2 = sigma * sigma
        self._prev_eps = 0.0

    def get_params(self) -> dict[str, Any]:
        return {"sigma_mid": math.sqrt(max(1e-12, self._sigma2))}

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        if previous is None:
            return StepScore(step_index=-1, ts=current.ts, model_name=self.name, model_kind=self.kind, family=self.family, skip_reason="missing_previous_frame")
        delta = float(current.mid - previous.mid)
        sigma = math.sqrt(max(1e-12, self._sigma2))
        ll = _gaussian_log_pdf(delta, 0.0, sigma)
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=ll,
            nll=-ll,
            standardized_residual=delta / sigma,
            observation={"delta_mid": delta, "sigma_mid_before": sigma},
            params_before=self.get_params(),
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        delta = float(current.mid - previous.mid)
        eps = delta
        self._sigma2 = max(1e-12, float(self.omega) + float(self.alpha) * self._prev_eps * self._prev_eps + float(self.beta) * self._sigma2)
        self._prev_eps = eps
        return True


@dataclass
class SpreadNormMidDeltaModel:
    """EWMA Gaussian on spread-normalized mid moves."""

    decay: float = 0.99
    initial_sigma: float = 1.0
    name: str = "price_spread_norm_mid_delta"
    kind: str = "spread_norm_mid_delta"
    family: str = "price"
    _mu: float = field(default=0.0, init=False, repr=False)
    _sigma2: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mu = 0.0
        sigma = max(1e-6, float(self.initial_sigma))
        self._sigma2 = sigma * sigma

    def get_params(self) -> dict[str, Any]:
        return {"mu_norm": self._mu, "sigma_norm": math.sqrt(max(1e-12, self._sigma2))}

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        if previous is None:
            return StepScore(step_index=-1, ts=current.ts, model_name=self.name, model_kind=self.kind, family=self.family, skip_reason="missing_previous_frame")
        spread = max(1e-12, float(previous.spread))
        target = float(current.mid - previous.mid) / spread
        sigma = math.sqrt(max(1e-12, self._sigma2))
        ll = _gaussian_log_pdf(target, self._mu, sigma)
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=ll,
            nll=-ll,
            standardized_residual=(target - self._mu) / sigma,
            observation={"spread_norm_delta": target},
            params_before=self.get_params(),
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        spread = max(1e-12, float(previous.spread))
        target = float(current.mid - previous.mid) / spread
        gamma = max(1e-6, min(1.0, float(self.decay)))
        mu_new = gamma * self._mu + (1.0 - gamma) * target
        self._sigma2 = gamma * self._sigma2 + (1.0 - gamma) * (target - mu_new) ** 2
        self._mu = float(mu_new)
        return True


@dataclass
class MicropriceDeltaModel:
    """EWMA Gaussian on microprice deltas."""

    decay: float = 0.99
    initial_sigma: float = 50.0
    name: str = "price_microprice_delta"
    kind: str = "microprice_delta"
    family: str = "price"
    _mu: float = field(default=0.0, init=False, repr=False)
    _sigma2: float = field(default=2500.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mu = 0.0
        sigma = max(1e-6, float(self.initial_sigma))
        self._sigma2 = sigma * sigma

    def get_params(self) -> dict[str, Any]:
        return {"mu_micro": self._mu, "sigma_micro": math.sqrt(max(1e-12, self._sigma2))}

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        if previous is None:
            return StepScore(step_index=-1, ts=current.ts, model_name=self.name, model_kind=self.kind, family=self.family, skip_reason="missing_previous_frame")
        delta = float(current.microprice - previous.microprice)
        sigma = math.sqrt(max(1e-12, self._sigma2))
        ll = _gaussian_log_pdf(delta, self._mu, sigma)
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=ll,
            nll=-ll,
            standardized_residual=(delta - self._mu) / sigma,
            observation={"delta_microprice": delta},
            params_before=self.get_params(),
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        delta = float(current.microprice - previous.microprice)
        gamma = max(1e-6, min(1.0, float(self.decay)))
        mu_new = gamma * self._mu + (1.0 - gamma) * delta
        self._sigma2 = gamma * self._sigma2 + (1.0 - gamma) * (delta - mu_new) ** 2
        self._mu = float(mu_new)
        return True


@dataclass
class TwoStateMidDeltaHMM:
    """Two-state Gaussian HMM on mid deltas (quiet vs active)."""

    decay: float = 0.99
    initial_sigma_quiet: float = 30.0
    initial_sigma_active: float = 80.0
    name: str = "price_hmm2_mid_delta"
    kind: str = "hmm2_mid_delta"
    family: str = "price"
    _trans: np.ndarray = field(init=False, repr=False)
    _state_prob: np.ndarray = field(init=False, repr=False)
    _sigma2: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._trans = np.array([[0.95, 0.05], [0.10, 0.90]], dtype=np.float64)
        self._state_prob = np.array([0.8, 0.2], dtype=np.float64)
        self._sigma2 = np.array(
            [float(self.initial_sigma_quiet) ** 2, float(self.initial_sigma_active) ** 2],
            dtype=np.float64,
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "state_prob": self._state_prob.tolist(),
            "sigma_quiet": math.sqrt(self._sigma2[0]),
            "sigma_active": math.sqrt(self._sigma2[1]),
        }

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        if previous is None:
            return StepScore(step_index=-1, ts=current.ts, model_name=self.name, model_kind=self.kind, family=self.family, skip_reason="missing_previous_frame")
        delta = float(current.mid - previous.mid)
        ll = self._predictive_log_pdf(delta)
        sigma = math.sqrt(max(1e-12, float(self._state_prob @ self._sigma2)))
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=ll,
            nll=-ll,
            standardized_residual=delta / max(1e-12, sigma),
            observation={"delta_mid": delta},
            params_before=self.get_params(),
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        delta = float(current.mid - previous.mid)
        gamma = max(1e-6, min(1.0, float(self.decay)))
        pred_prob = self._trans.T @ self._state_prob
        state_ll = np.array([_gaussian_log_pdf(delta, 0.0, math.sqrt(self._sigma2[i])) for i in range(2)])
        weights = pred_prob * np.exp(state_ll - max(state_ll))
        weights = weights / max(1e-300, weights.sum())
        for i in range(2):
            resid = delta
            self._sigma2[i] = gamma * self._sigma2[i] + (1.0 - gamma) * resid * resid
        self._state_prob = weights
        active = 1 if weights[1] > weights[0] else 0
        trans_obs = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                trans_obs[i, j] = 1.0 if (i == (1 if active else 0) and j == active) else 0.0
        self._trans = gamma * self._trans + (1.0 - gamma) * trans_obs
        self._trans = self._trans / np.maximum(1e-12, self._trans.sum(axis=1, keepdims=True))
        return True

    def _predictive_log_pdf(self, delta: float) -> float:
        pred_prob = self._trans.T @ self._state_prob
        components = []
        for i in range(2):
            sigma = math.sqrt(max(1e-12, self._sigma2[i]))
            components.append(pred_prob[i] + _gaussian_log_pdf(delta, 0.0, sigma))
        m = max(components)
        return m + math.log(sum(math.exp(c - m) for c in components))
