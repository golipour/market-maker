"""Unified strategy frame → decision → risk → execution pipeline."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any

from mm.application.venue_execution import (
    compute_venue_execution_gates,
    prepare_execution_joint_book,
    route_joint_target_book_with_report,
)
from mm.execution.plan import ExecutionSyncReport, stage_report_with_rejections
from mm.application.frame_preparation import clean_strategy_frame
from mm.application.trading_context import trading_state_from_ledger
from mm.bus.base import AbstractEventBus
from mm.controller.context import DecisionContext, execution_blocked_reasons
from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.interfaces.risk import AbstractRiskEngine
from mm.core.models.execution import Action, Order
from mm.core.models.instrument import SymbolBinding
from mm.core.models.joint_strategy import JointStrategyFrame
from mm.core.models.ledger import JointLedgerSnapshot
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.diagnostics.publisher import DiagnosticsPublisher
from mm.observability import log_event
from mm.risk.net_delta import NetDeltaRiskEngine, target_book_result_to_risk_result
from mm.runtime.config import (
    ControllerRuntimeConfig,
    ExecutionRuntimeConfig,
    LiquiditySourcingRuntimeConfig,
)
from mm.runtime.control import RuntimeControlPlane
from mm.strategies.mpc_engine import MpcStrategyEngine, order_summary_dict


@dataclass(slots=True)
class StrategyRunner:
    """Consumes ``JointStrategyFrame``, runs MPC, risk, and optional live execution.

    Pipeline per frame:

    1. Collect live orders from the ledger.
    2. ``clean_strategy_frame`` — subtract own orders from local/global books.
    3. ``MpcStrategyEngine.decide`` on the clean frame.
    4. ``AbstractRiskEngine.validate_frame`` on the proposed action.
    5. Publish decision + approved action; ``route_actions_to_managers`` when allowed.
    6. Publish ``ControllerDiagnostics`` once per frame in ``finally`` (including errors).
    """

    config: ControllerRuntimeConfig
    binding: SymbolBinding
    frame_queue: asyncio.Queue[JointStrategyFrame]
    order_managers: dict[str, AbstractOrderManager]
    risk_engine: AbstractRiskEngine
    liquidity_sourcing: LiquiditySourcingRuntimeConfig = field(
        default_factory=LiquiditySourcingRuntimeConfig
    )
    execution: ExecutionRuntimeConfig = field(default_factory=ExecutionRuntimeConfig)
    bus: AbstractEventBus | None = None
    control_plane: RuntimeControlPlane | None = None
    run_id: str = ""
    _engine: MpcStrategyEngine = field(init=False, repr=False)
    _diagnostics: DiagnosticsPublisher = field(init=False, repr=False)
    _frame_step: int = field(default=0, init=False, repr=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _stopped: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _last_local_exchange: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self._engine = MpcStrategyEngine(
            config=self.config,
            binding=self.binding,
            liquidity_sourcing=self.liquidity_sourcing,
            execution=self.execution,
        )
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
                try:
                    await self.on_joint_frame(frame)
                except Exception as exc:
                    log_event(
                        "strategy_runner_frame_error",
                        {
                            "actor_id": self.config.actor_id,
                            "exchange": frame.local_market.exchange,
                            "symbol": frame.local_market.symbol,
                            "t_end_ms": frame.local_market.t_end_ms,
                            "error": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
        except asyncio.CancelledError:
            raise

    async def on_joint_frame(self, joint_frame: JointStrategyFrame) -> list[Action]:
        frame = joint_frame.input_frame
        if frame is None:
            raise ValueError("JointStrategyFrame.input_frame is required for strategy pipeline")
        if joint_frame.ledger is None:
            raise ValueError("JointStrategyFrame.ledger is required for strategy pipeline")
        if frame.created_at_ms <= 0:
            created_at_ms = joint_frame.created_at_ms or joint_frame.ledger.created_at_ms
            if created_at_ms > 0:
                frame = replace(frame, created_at_ms=created_at_ms)
        return await self._process_input_frame(frame, joint_frame=joint_frame)

    async def _process_input_frame(
        self,
        frame: StrategyInputFrame,
        *,
        joint_frame: JointStrategyFrame,
    ) -> list[Action]:
        t0 = time.monotonic_ns()
        self._frame_step += 1
        self._last_local_exchange = frame.local.exchange.lower().strip()
        self._sync_control_plane_to_risk()

        clean_started = time.monotonic_ns()
        ledger = joint_frame.ledger
        open_orders = _live_orders_from_ledger(ledger)
        clean_frame = clean_strategy_frame(frame, open_orders)
        trading_state = trading_state_from_ledger(
            ledger,
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
            binding=self.binding,
            ledger=ledger,
            frame_sync_status=joint_frame.sync_status.as_payload(),
        )
        ctx.record_stage_latency("clean", _elapsed_ms(clean_started))
        approved: list[Action] = []

        try:
            decide_started = time.monotonic_ns()
            try:
                self._engine.set_ledger_context(ledger)
                decision = self._engine.decide(clean_frame, frame_step=self._frame_step)
                ctx.record_decision(decision)
                ctx.record_stage_latency("decide", _elapsed_ms(decide_started))
            except Exception as exc:
                ctx.record_stage_latency("decide", _elapsed_ms(decide_started))
                ctx.record_controller_error(exc)
                await self._publish_controller_error(frame, exc)
                return approved

            risk_started = time.monotonic_ns()
            proposed = [decision.action]
            execution_book = decision.joint_target_book
            if execution_book is None:
                raise ValueError("StrategyDecision.joint_target_book is required")
            if execution_book is not None:
                execution_book = prepare_execution_joint_book(
                    execution_book,
                    execution=self.execution,
                    liquidity_sourcing=self.liquidity_sourcing,
                    frame=clean_frame,
                    binding=self.binding,
                )

            intent_gates = compute_venue_execution_gates(
                config=self.config,
                execution=self.execution,
                liquidity_sourcing=self.liquidity_sourcing,
                binding=self.binding,
                frame_step=self._frame_step,
                joint_target_book=execution_book,
                order_managers=self.order_managers,
                control_plane=self.control_plane,
                frame=clean_frame,
                ignore_empty_book=True,
            )
            global_exchange = (
                self.binding.global_.exchange.lower().strip()
                if self.binding.global_ is not None
                else ""
            )
            global_live_execution = bool(
                global_exchange
                and intent_gates.get(global_exchange) is not None
                and intent_gates[global_exchange].live
                and intent_gates[global_exchange].has_orders
            )
            live_execution = any(
                gate.live and gate.has_orders for gate in intent_gates.values()
            )

            if isinstance(self.risk_engine, NetDeltaRiskEngine):
                target_risk_result = self.risk_engine.validate_target_book_with_result(
                    decision.joint_target_book,
                    ledger,
                    clean_frame,
                    live_execution=live_execution,
                    global_live_execution=global_live_execution,
                )
                risk_result = target_book_result_to_risk_result(
                    target_risk_result,
                    proposed_actions=proposed,
                )
                if target_risk_result.approved is not None:
                    execution_book = prepare_execution_joint_book(
                        target_risk_result.approved,
                        execution=self.execution,
                        liquidity_sourcing=self.liquidity_sourcing,
                        frame=clean_frame,
                        binding=self.binding,
                    )
            else:
                risk_result = self.risk_engine.validate_frame_with_result(
                    proposed, trading_state, clean_frame
                )
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
            ctx.record_stage_latency("risk", _elapsed_ms(risk_started))
            if len(approved) < len(proposed) and self.bus is not None:
                await self.bus.publish(
                    "risk:events",
                    {
                        "type": "RiskActionsFiltered",
                        "actor_id": self.config.actor_id,
                        "exchange": frame.local.exchange,
                        "symbol": frame.local.symbol,
                        "t_end_ms": frame.local.t_end_ms,
                        "proposed_actions": len(proposed),
                        "approved_actions": len(approved),
                    },
                )

            publish_started = time.monotonic_ns()
            observe_only_control = (
                self.control_plane is not None and self.control_plane.state.observe_only
            )
            venue_gates = compute_venue_execution_gates(
                config=self.config,
                execution=self.execution,
                liquidity_sourcing=self.liquidity_sourcing,
                binding=self.binding,
                frame_step=self._frame_step,
                joint_target_book=execution_book,
                order_managers=self.order_managers,
                control_plane=self.control_plane,
                frame=clean_frame,
            )
            executed = self._should_execute_live(approved, venue_gates=venue_gates)
            blocked = execution_blocked_reasons(
                config=self.config,
                frame_step=self._frame_step,
                approved=approved,
                observe_only_control=observe_only_control,
                local_manager_present=self.order_managers.get(self._last_local_exchange) is not None,
            )
            if executed:
                blocked = []
            elif approved and venue_gates:
                blocked = sorted(
                    {
                        *blocked,
                        *(
                            reason
                            for gate in venue_gates.values()
                            for reason in gate.blocked_reasons
                            if reason != "no_orders" or gate.has_orders
                        ),
                    }
                )
            ctx.record_execution_gates(
                executed=executed,
                blocked_reasons=blocked,
                venue_gates={exchange: gate.as_dict() for exchange, gate in venue_gates.items()},
            )

            executed_orders: tuple[dict, ...] = ()
            if executed:
                executed_orders = tuple(
                    order_summary_dict(order)
                    for action in approved
                    for order in action.orders
                )

            await self._publish_decision(decision, executed_orders=executed_orders)
            if decision.joint_target_book is not None:
                await self._publish(
                    self.config.normalized_target_books_topic,
                    {
                        "type": "JointTargetBook",
                        **decision.joint_target_book.as_payload(),
                        "exchange": frame.local.exchange,
                        "symbol": frame.local.symbol,
                        "t_end_ms": frame.local.t_end_ms,
                        "frame_step": self._frame_step,
                    },
                )
            for action in approved:
                await self._publish(self.config.normalized_action_topic, action)
            ctx.record_stage_latency("publish", _elapsed_ms(publish_started))

            execute_started = time.monotonic_ns()
            execution_report: ExecutionSyncReport | None = None
            sync_executed = executed
            if approved:
                if executed:
                    execution_report = await self._sync_joint_with_timeout(
                        execution_book,
                        venue_gates,
                        ctx,
                    )
                    if execution_report is None:
                        sync_executed = False
                else:
                    execution_report = await route_joint_target_book_with_report(
                        execution_book,
                        order_managers=self.order_managers,
                        venue_gates=venue_gates,
                        execute=False,
                    )
            ctx.record_stage_latency("execute", _elapsed_ms(execute_started))
            if execution_report is not None:
                ctx.apply_execution_to_pipeline(execution_report, executed=sync_executed)

            if self.bus is not None:
                await self.bus.publish(
                    f"strategy:journal:{self.config.actor_id}",
                    {
                        "actor_id": self.config.actor_id,
                        "exchange": frame.local.exchange,
                        "symbol": frame.local.symbol,
                        "t_end_ms": frame.local.t_end_ms,
                        "global_status": frame.global_status,
                        "proposed_actions": len(proposed),
                        "approved_actions": len(approved),
                        "executed": sync_executed,
                        "latency_ms": (time.monotonic_ns() - t0) / 1_000_000,
                    },
                )
            return approved
        except Exception as exc:
            if "risk" not in ctx.stage_timings_ms:
                ctx.record_stage_latency("risk", _elapsed_ms(risk_started) if "risk_started" in locals() else 0.0)
            ctx.record_controller_error(exc)
            await self._publish_controller_error(frame, exc)
            return approved
        finally:
            ctx.latency_ms = (time.monotonic_ns() - t0) / 1_000_000
            await self._publish_diagnostics(ctx)

    async def _sync_joint_with_timeout(
        self,
        execution_book,
        venue_gates: dict,
        ctx: DecisionContext,
    ) -> ExecutionSyncReport | None:
        timeout_ms = int(self.config.placement_timeout_ms)
        try:
            coro = route_joint_target_book_with_report(
                execution_book,
                order_managers=self.order_managers,
                venue_gates=venue_gates,
                execute=True,
            )
            if timeout_ms <= 0:
                return await coro
            return await asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            message = f"ExecutionSyncTimeout: exceeded {timeout_ms}ms placement_timeout_ms"
            ctx.record_warning(message)
            ctx.record_execution_gates(
                executed=False,
                blocked_reasons=[*ctx.blocked_reasons, "execution_sync_timeout"],
            )
            await self._publish_execution_error(ctx, message)
            return None
        except Exception as exc:
            message = f"ExecutionSyncError: {type(exc).__name__}: {exc}"
            ctx.record_warning(message)
            ctx.record_execution_gates(
                executed=False,
                blocked_reasons=[*ctx.blocked_reasons, "execution_sync_error"],
            )
            await self._publish_execution_error(ctx, message)
            return None

    async def _publish_execution_error(self, ctx: DecisionContext, message: str) -> None:
        if self.bus is None:
            return
        payload = {
            "type": "ExecutionSyncFailed",
            "actor_id": self.config.actor_id,
            "exchange": ctx.frame.local.exchange,
            "symbol": ctx.frame.local.symbol,
            "t_end_ms": ctx.frame.t_end_ms,
            "frame_step": ctx.frame_step,
            "error": message.split(":", 1)[0],
            "message": message,
            "retryable": True,
        }
        log_event("execution_sync_failed", payload)
        await self.bus.publish("execution:errors", payload)

    async def _publish_controller_error(
        self,
        frame: StrategyInputFrame,
        exc: BaseException,
    ) -> None:
        await self._publish(
            self.config.normalized_decisions_topic,
            {
                "type": "ControllerError",
                "actor_id": self.config.actor_id,
                "exchange": frame.local.exchange,
                "symbol": frame.local.symbol,
                "t_end_ms": frame.local.t_end_ms,
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )

    def _sync_control_plane_to_risk(self) -> None:
        if self.control_plane is None:
            return
        set_kill_switch = getattr(self.risk_engine, "set_kill_switch", None)
        if callable(set_kill_switch):
            set_kill_switch(self.control_plane.state.kill_switch)

    def _should_execute_live(
        self,
        approved: list[Action],
        *,
        venue_gates: dict,
    ) -> bool:
        if not approved:
            return False
        return any(gate.live and gate.has_orders for gate in venue_gates.values())

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

    async def _publish_diagnostics(self, ctx: DecisionContext) -> None:
        await self._diagnostics.publish(ctx.to_diagnostics())

    async def _publish(self, topic: str, event: Any) -> None:
        if self.bus is not None:
            await self.bus.publish(topic, event)


def _live_orders_from_ledger(ledger: JointLedgerSnapshot) -> dict[str, Order]:
    out: dict[str, Order] = {}
    for venue in (ledger.local, ledger.global_):
        if venue is None:
            continue
        exchange_key = venue.exchange.lower().strip()
        for exchange_order_id, open_order in venue.live_orders.items():
            out[f"{exchange_key}:{exchange_order_id}"] = open_order.order
    return out


def _elapsed_ms(started_ns: int) -> float:
    return (time.monotonic_ns() - started_ns) / 1_000_000
