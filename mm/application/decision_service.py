"""Strategy runner wiring for the market-maker runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mm.application.strategy_runner import StrategyRunner
from mm.bus.base import AbstractEventBus
from mm.exchanges.shared.provider_adapter import now_ms
from mm.portfolio.limits import AllocationCaps
from mm.portfolio.wallet import StrategyWallet
from mm.registry.models import AllocationRecord
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
    allocation: AllocationRecord | None = None,
    allocation_caps_provider: Callable[[], AllocationCaps | None] | None = None,
    strategy_wallet_provider: Callable[[], StrategyWallet | None] | None = None,
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
        risk_engine=build_risk_engine(
            ctrl.risk,
            binding=config.binding,
            liquidity_sourcing=config.liquidity_sourcing,
            clock_ms=now_ms,
            allocation=allocation,
            allocation_caps_provider=allocation_caps_provider,
            strategy_wallet_provider=strategy_wallet_provider,
        ),
        liquidity_sourcing=config.liquidity_sourcing,
        execution=config.execution,
        bus=bus,
        control_plane=control_plane,
        run_id=config.observability.run_id,
        strategy_wallet_provider=strategy_wallet_provider,
    )
