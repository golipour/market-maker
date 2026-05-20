"""Unified strategy frame → decision → risk → execution pipeline."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from mm.application.action_routing import route_actions_to_managers, route_actions_to_managers_with_report
from mm.execution.plan import stage_report_with_rejections
from mm.application.frame_preparation import clean_strategy_frame
from mm.application.trading_context import live_orders_from_managers, trading_state_from_frame
from mm.bus.base import AbstractEventBus
from mm.controller.context import DecisionContext, execution_blocked_reasons
from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.interfaces.risk import AbstractRiskEngine
from mm.core.models.execution import Action
from mm.core.models.instrument import SymbolBinding
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.diagnostics.publisher import DiagnosticsPublisher
from mm.risk.engine import RuntimeRiskEngine
from mm.runtime.config import ControllerRuntimeConfig
from mm.runtime.control import RuntimeControlPlane
from mm.strategies.mpc_engine import MpcStrategyEngine, order_summary_dict


@dataclass(slots=True)
class StrategyRunner:
    """Consumes ``StrategyInputFrame``, runs MPC, risk, and optional live execution.

    Pipeline per frame:

    1. Collect live orders from ``OrderManager`` instances.
    2. ``clean_strategy_frame`` — subtract own orders from primary/hedge books.
    3. ``MpcStrategyEngine.decide`` on the clean frame.
    4. ``AbstractRiskEngine.validate_frame`` on the proposed action.
    5. Publish decision + approved action; ``route_actions_to_managers`` when allowed.
    6. Publish ``ControllerDiagnostics`` (one per frame, including controller errors).
    """

    config: ControllerRuntimeConfig
    binding: SymbolBinding
    frame_queue: asyncio.Queue[StrategyInputFrame]
    order_managers: dict[str, AbstractOrderManager]
    risk_engine: AbstractRiskEngine
    bus: AbstractEventBus | None = None
    control_plane: RuntimeControlPlane | None = None
    run_id: str = ""
    _engine: MpcStrategyEngine = field(init=False, repr=False)
    _diagnostics: DiagnosticsPublisher = field(init=False, repr=False)
    _frame_step: int = field(default=0, init=False, repr=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _stopped: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _last_primary_exchange: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self._engine = MpcStrategyEngine(config=self.config, binding=self.binding)
        self._diagnostics = DiagnosticsPublisher(
            self.bus,
            topic=self.config.normalized_diagnostics_topic,
        )

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

    async def on_frame(self, frame: StrategyInputFrame) -> list[Action]:
        t0 = time.monotonic_ns()
        self._frame_step += 1
        self._last_primary_exchange = frame.primary.exchange.lower().strip()
        self._sync_control_plane_to_risk()

        open_orders = live_orders_from_managers(self.order_managers)
        clean_frame = clean_strategy_frame(frame, open_orders)
        trading_state = trading_state_from_frame(
            frame,
            self.binding,
            actor_id=self.config.actor_id,
            open_orders=open_orders,
        )

        ctx = DecisionContext(
            config=self.config,
            frame=frame,
            clean_frame=clean_frame,
            trading_state=trading_state,
            frame_step=self._frame_step,
            run_id=self.run_id,
        )

        try:
            decision = self._engine.decide(clean_frame, frame_step=self._frame_step)
            ctx.record_decision(decision)
        except Exception as exc:
            ctx.record_controller_error(exc)
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
            ctx.latency_ms = (time.monotonic_ns() - t0) / 1_000_000
            await self._publish_diagnostics(ctx)
            return []

        proposed = [decision.action]
        risk_result = self.risk_engine.validate_frame_with_result(proposed, trading_state, clean_frame)
        approved = risk_result.approved_list
        approved_orders = tuple(order for action in approved for order in action.orders)
        ctx.record_risk_result(
            proposed,
            approved,
            rejection_counts=dict(risk_result.rejection_counts),
            exposure_before=dict(risk_result.exposure_before),
            exposure_after=dict(risk_result.exposure_after),
        )
        ctx.apply_risk_to_pipeline(stage_report_with_rejections(approved_orders, risk_result.rejection_counts))
        if len(approved) < len(proposed) and self.bus is not None:
            await self.bus.publish(
                "risk:events",
                {
                    "type": "RiskActionsFiltered",
                    "actor_id": self.config.actor_id,
                    "exchange": frame.primary.exchange,
                    "symbol": frame.primary.symbol,
                    "t_end_ms": frame.primary.t_end_ms,
                    "proposed_actions": len(proposed),
                    "approved_actions": len(approved),
                },
            )

        observe_only_control = (
            self.control_plane is not None and self.control_plane.state.observe_only
        )
        executed = self._should_execute_live(approved)
        blocked = execution_blocked_reasons(
            config=self.config,
            frame_step=self._frame_step,
            approved=approved,
            observe_only_control=observe_only_control,
            primary_manager_present=self.order_managers.get(self._last_primary_exchange) is not None,
        )
        if executed:
            blocked = []
        ctx.record_execution_gates(executed=executed, blocked_reasons=blocked)

        executed_orders: tuple[dict, ...] = ()
        if executed:
            executed_orders = tuple(
                order_summary_dict(order)
                for action in approved
                for order in action.orders
            )

        await self._publish_decision(decision, executed_orders=executed_orders)
        for action in approved:
            await self._publish(self.config.normalized_action_topic, action)

        execution_report = None
        if approved:
            if executed:
                execution_report = await route_actions_to_managers_with_report(approved, self.order_managers)
            else:
                primary_manager = self.order_managers.get(self._last_primary_exchange)
                preview = getattr(primary_manager, "preview_placement", None) if primary_manager else None
                if preview is not None:
                    execution_report = preview(approved)
        if execution_report is not None:
            ctx.apply_execution_to_pipeline(execution_report, executed=executed)

        ctx.latency_ms = (time.monotonic_ns() - t0) / 1_000_000
        await self._publish_diagnostics(ctx)

        if self.bus is not None:
            await self.bus.publish(
                f"strategy:journal:{self.config.actor_id}",
                {
                    "actor_id": self.config.actor_id,
                    "exchange": frame.primary.exchange,
                    "symbol": frame.primary.symbol,
                    "t_end_ms": frame.primary.t_end_ms,
                    "hedge_status": frame.hedge_status,
                    "proposed_actions": len(proposed),
                    "approved_actions": len(approved),
                    "executed": executed,
                    "latency_ms": ctx.latency_ms,
                },
            )
        return approved

    def _sync_control_plane_to_risk(self) -> None:
        if self.control_plane is None:
            return
        if isinstance(self.risk_engine, RuntimeRiskEngine):
            self.risk_engine.set_kill_switch(self.control_plane.state.kill_switch)

    def _should_execute_live(self, approved: list[Action]) -> bool:
        if not approved:
            return False
        if self.config.observe_only:
            return False
        if self.control_plane is not None and self.control_plane.state.observe_only:
            return False
        if self._frame_step <= self.config.action_start_after_steps:
            return False
        if self.order_managers.get(self._last_primary_exchange) is None:
            return False
        return True

    async def _publish_decision(
        self,
        decision: StrategyDecision,
        *,
        executed_orders: tuple[dict[str, object], ...] = (),
    ) -> None:
        await self._publish(
            self.config.normalized_decisions_topic,
            decision.as_payload(executed_orders=executed_orders),
        )
        await self._publish(self.config.normalized_action_topic, decision.action)

    async def _publish_diagnostics(self, ctx: DecisionContext) -> None:
        await self._diagnostics.publish(ctx.to_diagnostics())

    async def _publish(self, topic: str, event: Any) -> None:
        if self.bus is not None:
            await self.bus.publish(topic, event)
