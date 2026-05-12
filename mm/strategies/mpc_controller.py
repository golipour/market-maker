"""Async MPC controller worker on the runtime bus."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.actor import AbstractOrderManager
from mm.core.models.execution import Action
from mm.core.models.instrument import SymbolBinding
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.runtime.config import ControllerRuntimeConfig
from mm.strategies.mpc_engine import MpcStrategyEngine, normalize_mpc_optimizer_config

__all__ = ["MpcControllerRunner", "MpcStrategyEngine", "normalize_mpc_optimizer_config"]


@dataclass(slots=True)
class MpcControllerRunner:
    """Async bus worker around ``MpcStrategyEngine``."""

    config: ControllerRuntimeConfig
    binding: SymbolBinding
    frame_queue: asyncio.Queue[StrategyInputFrame]
    order_managers: dict[str, AbstractOrderManager]
    bus: AbstractEventBus | None = None
    _engine: MpcStrategyEngine = field(init=False, repr=False)
    _frame_step: int = field(default=0, init=False, repr=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _stopped: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    def __post_init__(self) -> None:
        self._engine = MpcStrategyEngine(config=self.config, binding=self.binding)

    async def start(self) -> None:
        self._stopped.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        try:
            while not self._stopped.is_set():
                frame = await self.frame_queue.get()
                await self.on_frame(frame)
        except asyncio.CancelledError:
            raise

    async def on_frame(self, frame: StrategyInputFrame) -> Action:
        try:
            self._frame_step += 1
            decision = self._engine.decide(frame, frame_step=self._frame_step)
            await self._publish_decision(decision)
            if not self.config.observe_only:
                manager = self.order_managers.get(frame.primary.exchange.lower().strip())
                if manager is not None and self._frame_step > self.config.action_start_after_steps:
                    await manager.sync([decision.action])
            return decision.action
        except Exception as exc:
            await self._publish(
                self.config.normalized_decisions_topic,
                {
                    "type": "ControllerError",
                    "actor_id": self.config.actor_id,
                    "exchange": frame.primary.exchange,
                    "symbol": frame.primary.symbol,
                    "t_end_ms": frame.primary.t_end_ms,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )
            return Action(actor_id=self.config.actor_id)

    async def _publish_decision(self, decision: StrategyDecision) -> None:
        await self._publish(self.config.normalized_decisions_topic, decision.as_payload())
        await self._publish(self.config.normalized_action_topic, decision.action)

    async def _publish(self, topic: str, event: Any) -> None:
        if self.bus is not None:
            await self.bus.publish(topic, event)
