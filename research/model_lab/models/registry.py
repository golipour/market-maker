"""Model registry and factory."""

from __future__ import annotations

from typing import Any

from research.model_lab.core.protocol import OnlineModel
from research.model_lab.models.adapters import ProductionFlowModel, ProductionPriceModel
from research.model_lab.models.baselines import ConstantZeroFlowModel, GaussianMidRandomWalkModel
from research.model_lab.models.experimental import (
    BivariateHurdleLogNormalFlowModel,
    FlowConditionedMidDeltaModel,
    HurdleLogStudentTFlowModel,
    StudentTMidDeltaModel,
)

_FLOW_KINDS = frozenset({"hurdle_lognormal", "hurdle_gamma", "hurdle_weibull", "gamma", "weibull"})
_EXPERIMENTAL_FLOW_KINDS = frozenset({"hurdle_log_student_t", "bivariate_hurdle_lognormal"})
_PRICE_KINDS = frozenset({"residual", "conditional_mechanical_mixture", "flow_gated_mixture"})
_EXPERIMENTAL_PRICE_KINDS = frozenset({"student_t_mid_delta", "flow_conditioned_mid_delta"})

DEFAULT_FLOW_CONFIG = {"decay": 0.95, "init_p0": 0.3, "init_mu": 0.0, "init_sigma": 1.0}
DEFAULT_RESIDUAL_CONFIG = {"residual_decay": 0.95, "residual_sigma": 50.0}
DEFAULT_CONDITIONAL_CONFIG = {
    "reference_vwap_vol": 1000.0,
    "shape_alpha": 0.5,
    "decay": 0.99,
    "initial_sigma": 50.0,
    "initial_log_sigma": 0.25,
    "student_df": 3.0,
    "point_mode": "sample",
}


def parse_model_spec(spec: str) -> tuple[str, str, dict[str, Any]]:
    """Parse ``family:kind`` or ``family:kind{json}``."""

    raw = spec.strip()
    if ":" not in raw:
        raise ValueError(f"invalid model spec {spec!r}; expected family:kind")
    family, rest = raw.split(":", 1)
    family = family.strip().lower()
    if family not in {"flow", "price", "baseline"}:
        raise ValueError(f"unknown family {family!r} in {spec!r}")

    config: dict[str, Any] = {}
    if "{" in rest:
        kind, _, tail = rest.partition("{")
        kind = kind.strip()
        import json

        config = json.loads("{" + tail)
    else:
        kind = rest.strip()

    return family, kind, config


def build_model(spec: str, *, overrides: dict[str, Any] | None = None) -> OnlineModel:
    family, kind, inline_config = parse_model_spec(spec)
    config = dict(overrides or {})
    config.update(inline_config)

    if family == "baseline":
        if kind == "zero_flow":
            return ConstantZeroFlowModel()
        if kind in {"mid_random_walk", "gaussian_mid_delta"}:
            return GaussianMidRandomWalkModel(**{k: v for k, v in config.items() if k in {"decay", "initial_sigma"}})
        raise ValueError(f"unknown baseline kind {kind!r}")

    if family == "flow":
        if kind == "hurdle_log_student_t":
            merged = {**DEFAULT_FLOW_CONFIG, **{"init_df": 5.0}, **config}
            return HurdleLogStudentTFlowModel(**merged)
        if kind == "bivariate_hurdle_lognormal":
            merged = {**DEFAULT_FLOW_CONFIG, **config}
            return BivariateHurdleLogNormalFlowModel(
                decay=float(merged["decay"]),
                init_mu=float(merged["init_mu"]),
                init_sigma=float(merged["init_sigma"]),
                init_p00=float(merged.get("init_p00", 0.5)),
                init_p10=float(merged.get("init_p10", 0.2)),
                init_p01=float(merged.get("init_p01", 0.2)),
                init_p11=float(merged.get("init_p11", 0.1)),
            )
        if kind not in _FLOW_KINDS:
            choices = sorted(_FLOW_KINDS | _EXPERIMENTAL_FLOW_KINDS)
            raise ValueError(f"unsupported flow kind {kind!r}; choices: {choices}")
        merged = {**DEFAULT_FLOW_CONFIG, **config}
        return ProductionFlowModel(kind=kind, config=merged, name=f"flow_{kind}")

    if family == "price":
        if kind == "student_t_mid_delta":
            return StudentTMidDeltaModel(**{k: v for k, v in config.items() if k in {"decay", "initial_sigma", "df"}})
        if kind == "flow_conditioned_mid_delta":
            return FlowConditionedMidDeltaModel(
                **{
                    k: v
                    for k, v in config.items()
                    if k in {"decay", "variance_decay", "initial_sigma", "rls_p0"}
                }
            )
        if kind not in _PRICE_KINDS:
            choices = sorted(_PRICE_KINDS | _EXPERIMENTAL_PRICE_KINDS)
            raise ValueError(f"unsupported price kind {kind!r}; choices: {choices}")
        if kind == "residual":
            merged = {**DEFAULT_RESIDUAL_CONFIG, **config}
            ref = float(merged.pop("reference_vwap_vol", 1000.0))
            return ProductionPriceModel(kind=kind, config=merged, reference_vwap_vol=ref, name=f"price_{kind}")
        merged = {**DEFAULT_CONDITIONAL_CONFIG, **config}
        return ProductionPriceModel(kind=kind, config=merged, name=f"price_{kind}")

    raise ValueError(f"unsupported family {family!r}")


def build_flow_model(kind: str = "hurdle_lognormal", **config: Any) -> ProductionFlowModel:
    merged = {**DEFAULT_FLOW_CONFIG, **config}
    return ProductionFlowModel(kind=kind, config=merged, name=f"flow_{kind}")


def build_price_model(kind: str = "conditional_mechanical_mixture", **config: Any) -> ProductionPriceModel:
    if kind == "residual":
        merged = {**DEFAULT_RESIDUAL_CONFIG, **config}
        ref = float(merged.pop("reference_vwap_vol", 1000.0))
        return ProductionPriceModel(kind=kind, config=merged, reference_vwap_vol=ref, name=f"price_{kind}")
    merged = {**DEFAULT_CONDITIONAL_CONFIG, **config}
    return ProductionPriceModel(kind=kind, config=merged, name=f"price_{kind}")


def list_builtin_models() -> list[str]:
    out = [
        "baseline:zero_flow",
        "baseline:mid_random_walk",
    ]
    for kind in sorted(_FLOW_KINDS):
        out.append(f"flow:{kind}")
    for kind in sorted(_EXPERIMENTAL_FLOW_KINDS):
        out.append(f"flow:{kind}")
    for kind in sorted(_PRICE_KINDS):
        out.append(f"price:{kind}")
    for kind in sorted(_EXPERIMENTAL_PRICE_KINDS):
        out.append(f"price:{kind}")
    return out
