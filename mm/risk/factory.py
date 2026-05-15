"""Construct risk engines from runtime configuration."""

from __future__ import annotations

from collections.abc import Callable

from mm.core.interfaces.actor import AbstractRiskEngine
from mm.risk.engine import PassThroughRiskEngine, RiskLimits, RuntimeRiskEngine
from mm.runtime.config import RiskRuntimeConfig


def build_risk_engine(
    config: RiskRuntimeConfig | None,
    *,
    clock_ms: Callable[[], int] | None = None,
) -> AbstractRiskEngine:
    if config is None or not config.enabled:
        return PassThroughRiskEngine()
    return RuntimeRiskEngine(
        RiskLimits(
            max_inventory=config.max_inventory,
            max_data_age_ms=config.max_data_age_ms,
        ),
        clock_ms=clock_ms,
    )
