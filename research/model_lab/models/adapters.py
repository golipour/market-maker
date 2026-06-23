"""Adapters around production torch flow/price models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.model_lab.core.scoring import (
    score_conditional_mixture_transition,
    score_flow_step,
    score_residual_transition,
)
from research.model_lab.core.types import IntervalStep, StepScore, mpc_step_from_interval


@dataclass
class ProductionFlowModel:
    """Wrap ``mm.mpc.market_order_flow_torch`` hurdle/gamma/weibull models."""

    kind: str
    config: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    family: str = "flow"
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"flow_{self.kind}"
        from mm.strategies.mpc_engine import _build_market_flow_model

        self._model = _build_market_flow_model(
            dict(self.config) | {"kind": self.kind},
            config_name="local_market_flow_model",
        )

    def reset(self) -> None:
        from mm.strategies.mpc_engine import _build_market_flow_model

        self._model = _build_market_flow_model(
            dict(self.config) | {"kind": self.kind},
            config_name="local_market_flow_model",
        )

    def get_params(self) -> dict[str, Any]:
        return dict(self._model.get_params())

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        result = score_flow_step(
            buy_volume=current.buy_volume,
            sell_volume=current.sell_volume,
            params_before=self.get_params(),
            kind=self.kind,
        )
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
        prev_step = mpc_step_from_interval(previous)
        curr_step = mpc_step_from_interval(current)
        self._model.update(prev_step, curr_step)
        return True


@dataclass
class ProductionPriceModel:
    """Wrap ``mm.mpc.price_torch`` residual or conditional mixture models."""

    kind: str
    config: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    family: str = "price"
    reference_vwap_vol: float = 1000.0
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"price_{self.kind}"
        self.family = "price"
        from mm.strategies.mpc_engine import _build_price_model

        cfg = dict(self.config)
        cfg.setdefault("kind", self.kind)
        self._model = _build_price_model(cfg, market_flow_model=None, config_name="local_price_model")
        if self.kind == "residual":
            setattr(self._model, "_target_depth_lots", float(self.reference_vwap_vol))

    def reset(self) -> None:
        from mm.strategies.mpc_engine import _build_price_model

        cfg = dict(self.config)
        cfg.setdefault("kind", self.kind)
        self._model = _build_price_model(cfg, market_flow_model=None, config_name="local_price_model")
        if self.kind == "residual":
            setattr(self._model, "_target_depth_lots", float(self.reference_vwap_vol))

    def get_params(self) -> dict[str, Any]:
        if hasattr(self._model, "get_params"):
            return dict(self._model.get_params())
        return {}

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
        if self.kind == "residual":
            result = score_residual_transition(self._model, previous, current)
        elif self.kind in {"conditional_mechanical_mixture", "flow_gated_mixture"}:
            result = score_conditional_mixture_transition(self._model, previous, current)
        else:
            return StepScore(
                step_index=-1,
                ts=current.ts,
                model_name=self.name,
                model_kind=self.kind,
                family=self.family,
                skip_reason="unsupported_model",
            )
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
        prev_step = mpc_step_from_interval(previous)
        curr_step = mpc_step_from_interval(current)
        self._model.update(prev_step, curr_step)
        return True
