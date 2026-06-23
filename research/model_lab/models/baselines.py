"""Simple baseline models for benchmarking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from research.model_lab.core.scoring import score_flow_step, score_mid_delta_step
from research.model_lab.core.types import IntervalStep, StepScore


@dataclass
class ConstantZeroFlowModel:
    """Predict zero buy/sell volume with probability 1."""

    name: str = "baseline_zero_flow"
    kind: str = "constant_zero"
    family: str = "flow"
    _p0: float = 1.0

    def reset(self) -> None:
        self._p0 = 1.0

    def get_params(self) -> dict[str, Any]:
        return {"p0_buy": self._p0, "p0_sell": self._p0, "mu_buy": 0.0, "sigma_buy": 1.0, "mu_sell": 0.0, "sigma_sell": 1.0}

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        result = score_flow_step(
            buy_volume=current.buy_volume,
            sell_volume=current.sell_volume,
            params_before=self.get_params(),
            kind="hurdle_lognormal",
        )
        return self._to_step_score(current, result)

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        return False

    def _to_step_score(self, current: IntervalStep, result: Any) -> StepScore:
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=result.log_likelihood,
            nll=result.nll,
            skip_reason=result.skip_reason,
            standardized_residual=result.standardized_residual,
            observation=dict(result.observation),
            params_before=self.get_params(),
        )


@dataclass
class GaussianMidRandomWalkModel:
    """EWMA mean/variance on mid deltas (no flow features)."""

    name: str = "baseline_mid_random_walk"
    kind: str = "gaussian_mid_delta"
    family: str = "price"
    decay: float = 0.99
    initial_sigma: float = 50.0
    _mu: float = 0.0
    _sigma2: float = field(default=2500.0, init=False)
    _sigma_min: float = 1e-6

    def __post_init__(self) -> None:
        sigma0 = max(self._sigma_min, float(self.initial_sigma))
        self._sigma2 = sigma0 * sigma0

    def reset(self) -> None:
        self._mu = 0.0
        sigma0 = max(self._sigma_min, float(self.initial_sigma))
        self._sigma2 = sigma0 * sigma0

    def get_params(self) -> dict[str, Any]:
        sigma = max(self._sigma_min, math.sqrt(max(0.0, self._sigma2)))
        return {"mu_mid": self._mu, "sigma_mid": sigma}

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
        result = score_mid_delta_step(delta_mid=delta, params_before=self.get_params())
        return StepScore(
            step_index=-1,
            ts=current.ts,
            model_name=self.name,
            model_kind=self.kind,
            family=self.family,
            log_likelihood=result.log_likelihood,
            nll=result.nll,
            skip_reason=result.skip_reason,
            standardized_residual=result.standardized_residual,
            observation=dict(result.observation),
            params_before=self.get_params(),
        )

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        delta = float(current.mid - previous.mid)
        alpha = 1.0 - float(self.decay)
        mu_new = self.decay * self._mu + alpha * delta
        self._sigma2 = self.decay * self._sigma2 + alpha * (delta - mu_new) ** 2
        self._mu = float(mu_new)
        return True
