"""Construct risk engines from runtime configuration."""

from __future__ import annotations

from collections.abc import Callable

from mm.core.interfaces.risk import AbstractRiskEngine
from mm.core.models.instrument import SymbolBinding
from mm.risk.engine import PassThroughRiskEngine, RiskLimits, RuntimeRiskEngine
from mm.risk.net_delta import NetDeltaRiskEngine
from mm.runtime.config import LiquiditySourcingRuntimeConfig, RiskRuntimeConfig


def build_risk_engine(
    config: RiskRuntimeConfig | None,
    *,
    binding: SymbolBinding | None = None,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig | None = None,
    clock_ms: Callable[[], int] | None = None,
) -> AbstractRiskEngine:
    if config is None or not config.enabled:
        return PassThroughRiskEngine()
    limits = RiskLimits(primary=config.primary, hedge=config.hedge)
    if config.engine == "net_delta":
        if binding is None:
            raise ValueError("binding is required when controller.risk.engine is 'net_delta'")
        return NetDeltaRiskEngine(
            limits=limits,
            binding=binding,
            liquidity_sourcing=liquidity_sourcing or LiquiditySourcingRuntimeConfig(),
            clock_ms=clock_ms,
        )
    return RuntimeRiskEngine(limits, clock_ms=clock_ms)
