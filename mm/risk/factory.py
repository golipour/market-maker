"""Construct risk engines from runtime configuration."""

from __future__ import annotations

from collections.abc import Callable

from mm.core.interfaces.risk import AbstractRiskEngine
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
        RiskLimits(primary=config.primary, hedge=config.hedge),
        clock_ms=clock_ms,
    )
