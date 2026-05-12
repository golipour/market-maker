"""MPC strategy decision path (engine + async bus runner)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mm.bus.base import AbstractEventBus
from mm.core.models.instrument import SymbolBinding
from mm.runtime.config import ControllerRuntimeConfig, MarketMakerRuntimeConfig
from mm.strategies.mpc_controller import MpcControllerRunner
from mm.strategies.mpc_engine import MpcStrategyEngine

if TYPE_CHECKING:
    from mm.core.interfaces.actor import AbstractOrderManager

__all__ = ["MpcControllerRunner", "MpcStrategyEngine", "build_controller_runner"]


def build_controller_runner(
    config: MarketMakerRuntimeConfig,
    *,
    bus: AbstractEventBus,
    frame_topic: str,
    order_managers: dict[str, AbstractOrderManager],
) -> MpcControllerRunner | None:
    """Instantiate ``MpcControllerRunner`` when controller config is present and enabled."""

    ctrl = config.controller
    if ctrl is None or not ctrl.enabled:
        return None
    return MpcControllerRunner(
        config=ctrl,
        binding=config.binding,
        frame_queue=bus.subscribe_queue(frame_topic),
        order_managers=order_managers,
        bus=bus,
    )
