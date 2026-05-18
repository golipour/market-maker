"""Strategy runner wiring for the market-maker runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mm.application.strategy_runner import StrategyRunner
from mm.bus.base import AbstractEventBus
from mm.exchanges.shared.provider_adapter import now_ms
from mm.risk.factory import build_risk_engine
from mm.runtime.config import MarketMakerRuntimeConfig
from mm.runtime.control import RuntimeControlPlane
from mm.strategies.mpc_engine import MpcStrategyEngine

if TYPE_CHECKING:
    from mm.core.interfaces.order_manager import AbstractOrderManager

__all__ = ["MpcStrategyEngine", "StrategyRunner", "build_strategy_runner"]


def build_strategy_runner(
    config: MarketMakerRuntimeConfig,
    *,
    bus: AbstractEventBus,
    frame_topic: str,
    order_managers: dict[str, AbstractOrderManager],
    control_plane: RuntimeControlPlane | None = None,
) -> StrategyRunner | None:
    """Instantiate ``StrategyRunner`` when controller config is present and enabled."""

    ctrl = config.controller
    if ctrl is None or not ctrl.enabled:
        return None
    return StrategyRunner(
        config=ctrl,
        binding=config.binding,
        frame_queue=bus.subscribe_queue(frame_topic),
        order_managers=order_managers,
        risk_engine=build_risk_engine(ctrl.risk, clock_ms=now_ms),
        bus=bus,
        control_plane=control_plane,
        run_id=config.observability.run_id,
    )
