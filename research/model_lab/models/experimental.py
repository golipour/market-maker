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
