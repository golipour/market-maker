"""Unified strategy frame → decision → risk → execution pipeline."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from mm.application.frame_freshness import local_market_age_ms, stale_market_frame_reason
from mm.application.venue_execution import (
    compute_venue_execution_gates,
    prepare_execution_joint_book,
    route_joint_target_book_with_report,
)
from mm.core.adapters.joint import joint_target_book_to_action
from mm.execution.plan import (
    ExecutionSyncReport,
    stage_report_with_rejections,
)
from mm.application.frame_preparation import clean_strategy_frame
from mm.application.trading_context import trading_state_from_ledger, trading_state_from_wallet
from mm.bus.base import AbstractEventBus
from mm.controller.context import DecisionContext, execution_blocked_reasons
from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.interfaces.risk import AbstractRiskEngine
from mm.core.models.execution import Action, Order
from mm.core.models.instrument import SymbolBinding
from mm.core.models.joint_strategy import JointStrategyFrame
from mm.core.payload_schema import joint_strategy_frame_from_payload
from mm.core.models.ledger import JointLedgerSnapshot
from mm.core.models.strategy import StrategyDecision, StrategyInputFrame
from mm.diagnostics.publisher import DiagnosticsPublisher
from mm.observability import log_event
from mm.observability.heartbeat import HeartbeatPublisher, StrategyErrorPublisher
from mm.risk.net_delta import NetDeltaRiskEngine, target_book_result_to_risk_result
from mm.portfolio.wallet import StrategyWallet, joint_ledger_with_virtual_local
from mm.runtime.config import (
    ControllerRuntimeConfig,
    ExecutionRuntimeConfig,
    LiquiditySourcingRuntimeConfig,
)
from mm.runtime.control import RuntimeControlPlane
from mm.strategies.mpc_engine import MpcStrategyEngine, order_summary_dict


class OptimizerDecisionTimeout(RuntimeError):
    """Raised when MPC decision compute exceeds the configured frame budget."""


class OptimizerDecisionInProgress(RuntimeError):
    """Raised when the previous timed-out optimizer call is still running."""


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
    strategy_wallet_provider: Callable[[], StrategyWallet | None] | None = None
    max_local_market_age_ms: int = 0
    max_global_market_age_ms: int = 0
    _engine: MpcStrategyEngine = field(init=False, repr=False)
    _diagnostics: DiagnosticsPublisher = field(init=False, repr=False)
    _heartbeat: HeartbeatPublisher = field(init=False, repr=False)
    _error_publisher: StrategyErrorPublisher = field(init=False, repr=False)
    _frame_step: int = field(default=0, init=False, repr=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _decide_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _decide_pool: concurrent.futures.ThreadPoolExecutor = field(init=False, repr=False)
    _decide_cancel_event: threading.Event = field(init=False, repr=False)
    _stopped: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _last_local_exchange: str = field(default="", init=False, repr=False)
    _last_portfolio_mark: float | None = field(default=None, init=False, repr=False)

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
        interval_ms = int(self.config.frame_interval_ms)
        self._heartbeat = HeartbeatPublisher(
            bus=self.bus,
            actor_id=self.config.actor_id,
            interval_ms=interval_ms,
        )
        self._error_publisher = StrategyErrorPublisher(
            bus=self.bus,
            actor_id=self.config.actor_id,
        )
        # Dedicated single-worker thread pool for the optimizer call so a
        # wedged decide cannot starve other to_thread callers (the
        # 2026-09-01 Bitwana failure mode). The cancel event is checked
        # cooperatively by the optimizer's Adam loop, so a timeout
        # actually stops the work instead of leaking it into the pool.
        self._decide_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mm-decide-{self.config.actor_id}"
        )
        self._decide_cancel_event = threading.Event()

    async def start(self) -> None:
        self._stopped.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped.set()
        self._decide_cancel_event.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        pool = getattr(self, "_decide_pool", None)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if self.bus is not None:
            await self.bus.unsubscribe_queue(self.frame_queue)

    async def _run(self) -> None:
        try:
            while not self._stopped.is_set():
                raw_frame = await self.frame_queue.get()
                frame = _coerce_joint_strategy_frame(raw_frame)
                if frame is None:
                    log_event(
                        "strategy_runner_frame_skipped",
                        {
                            "actor_id": self.config.actor_id,
                            "reason": "unparseable_joint_strategy_frame",
                            "payload_type": type(raw_frame).__name__,
                        },
                    )
                    continue
                skip_reason = self._stale_frame_reason(frame)
                if skip_reason is not None:
                    log_event(
                        "strategy_runner_frame_skipped",
                        {
                            "actor_id": self.config.actor_id,
                            "reason": skip_reason,
                            "t_end_ms": frame.t_end_ms,
                            "local_market_age_ms": self._local_market_age_ms(frame),
                            "max_local_market_age_ms": self.max_local_market_age_ms,
                            "max_global_market_age_ms": self.max_global_market_age_ms,
                        },
                    )
                    continue
                try:
                    await self.on_joint_frame(frame)
                except Exception as exc:
                    context = _joint_frame_log_context(frame)
                    log_event(
                        "strategy_runner_frame_error",
                        {
                            "actor_id": self.config.actor_id,
                            **context,
                            "error": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
        except asyncio.CancelledError:
            raise

    def _local_market_age_ms(self, joint_frame: JointStrategyFrame) -> int | None:
        input_frame = joint_frame.input_frame
        if input_frame is None:
            return None
        return local_market_age_ms(input_frame, now_ms=int(time.time() * 1000))

    def _stale_frame_reason(self, joint_frame: JointStrategyFrame) -> str | None:
        input_frame = joint_frame.input_frame
        if input_frame is None:
            return None
        return stale_market_frame_reason(
            input_frame,
            now_ms=int(time.time() * 1000),
            max_local_market_age_ms=self.max_local_market_age_ms,
            max_global_market_age_ms=self.max_global_market_age_ms,
        )

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
        lineage = _lineage_ids(
            actor_id=self.config.actor_id,
            exchange=frame.local.exchange,
            symbol=frame.local.symbol,
            t_end_ms=frame.local.t_end_ms,
            frame_step=self._frame_step,
            existing_frame_id=frame.frame_id,
        )
        frame = replace(frame, frame_id=lineage["frame_id"])
        self._last_local_exchange = frame.local.exchange.lower().strip()
        self._sync_control_plane_to_risk()

        clean_started = time.monotonic_ns()
        physical_ledger = joint_frame.physical_ledger or joint_frame.ledger
        wallet = self.strategy_wallet_provider() if self.strategy_wallet_provider is not None else None
        if joint_frame.physical_ledger is not None:
            strategy_ledger = joint_frame.ledger
            open_orders = _live_orders_from_ledger(strategy_ledger)
            if wallet is not None:
                trading_state = trading_state_from_wallet(
                    wallet,
                    actor_id=self.config.actor_id,
                    open_orders=open_orders,
                )
            else:
                trading_state = trading_state_from_ledger(
                    strategy_ledger,
                    actor_id=self.config.actor_id,
                    open_orders=open_orders,
                )
        elif wallet is not None:
            strategy_ledger = joint_ledger_with_virtual_local(physical_ledger, wallet)
            open_orders = _live_orders_from_ledger(strategy_ledger)
            trading_state = trading_state_from_wallet(
                wallet,
                actor_id=self.config.actor_id,
                open_orders=open_orders,
            )
        else:
            strategy_ledger = physical_ledger
            open_orders = _live_orders_from_ledger(physical_ledger)
            trading_state = trading_state_from_ledger(
                physical_ledger,
                actor_id=self.config.actor_id,
                open_orders=open_orders,
            )
        clean_frame = clean_strategy_frame(frame, open_orders)

        ctx = DecisionContext(
            config=self.config,
            frame=frame,
            clean_frame=clean_frame,
            trading_state=trading_state,
            frame_step=self._frame_step,
            run_id=self.run_id,
            binding=self.binding,
            ledger=strategy_ledger,
            physical_ledger=physical_ledger,
            ledger_source=joint_frame.ledger_source,
            frame_sync_status=joint_frame.sync_status.as_payload(),
            last_portfolio_mark=self._last_portfolio_mark,
        )
        ctx.record_stage_latency("clean", _elapsed_ms(clean_started))
        approved: list[Action] = []

        try:
            decide_started = time.monotonic_ns()
            try:
                self._engine.set_ledger_context(strategy_ledger)
                decision = await self._decide_with_timeout(clean_frame, frame_step=self._frame_step)
                decision = _with_decision_lineage(decision, lineage)
                ctx.record_decision(decision)
                ctx.record_stage_latency("decide", _elapsed_ms(decide_started))
            except (OptimizerDecisionTimeout, OptimizerDecisionInProgress) as exc:
                ctx.record_stage_latency("decide", _elapsed_ms(decide_started))
                ctx.record_warning(str(exc))
                await self._publish_controller_fallback(frame, reason=_fallback_reason(exc), message=str(exc))
                return approved
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
                and intent_gates[global_exchange].can_sync
            )
            live_execution = any(
                gate.live and gate.can_sync for gate in intent_gates.values()
            )

            if isinstance(self.risk_engine, NetDeltaRiskEngine):
                target_risk_result = self.risk_engine.validate_target_book_with_result(
                    decision.joint_target_book,
                    strategy_ledger,
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
                risk_approved_execution_book = target_risk_result.approved is not None
            else:
                risk_result = self.risk_engine.validate_frame_with_result(
                    proposed, trading_state, clean_frame
                )
                risk_approved_execution_book = False
            approved = risk_result.approved_list
            approved_orders = tuple(order for action in approved for order in action.orders)
            risk_stage_orders = approved_orders
            local_risk_rejection_counts = _local_risk_rejection_counts(risk_result.rejection_counts)
            risk_rejections: tuple[dict[str, Any], ...] = ()
            if isinstance(self.risk_engine, NetDeltaRiskEngine):
                if target_risk_result.approved is not None:
                    risk_stage_orders = tuple(
                        joint_target_book_to_action(target_risk_result.approved).orders
                    )
                    if not risk_stage_orders and local_risk_rejection_counts:
                        risk_rejections = _risk_rejection_rows(
                            ctx,
                            local_risk_rejection_counts,
                            fallback_orders=tuple(
                                order for action in proposed for order in action.orders
                            ),
                        )
                else:
                    risk_stage_orders = ()
                    risk_rejections = _risk_rejection_rows(
                        ctx,
                        local_risk_rejection_counts,
                        fallback_orders=tuple(
                            order for action in proposed for order in action.orders
                        ),
                    )
            ctx.record_risk_result(
                proposed,
                approved,
                rejection_counts=dict(risk_result.rejection_counts),
                exposure_before=dict(risk_result.exposure_before),
                exposure_after=dict(risk_result.exposure_after),
            )
            ctx.apply_risk_to_pipeline(
                stage_report_with_rejections(
                    risk_stage_orders,
                    local_risk_rejection_counts,
                    rejections=risk_rejections,
                )
            )
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
            dry_run_control = (
                self.control_plane is not None and self.control_plane.state.dry_run
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
            execution_allowed_by_risk = bool(approved) or (
                (risk_approved_execution_book or not risk_result.rejection_counts)
                and any(gate.live and gate.can_sync for gate in venue_gates.values())
            )
            executed = self._should_execute_live(
                execution_allowed_by_risk=execution_allowed_by_risk,
                venue_gates=venue_gates,
            )
            blocked = execution_blocked_reasons(
                config=self.config,
                frame_step=self._frame_step,
                approved=approved,
                dry_run_control=dry_run_control,
                local_manager_present=self.order_managers.get(self._last_local_exchange) is not None,
            )
            if executed:
                blocked = []
            elif execution_allowed_by_risk and venue_gates:
                blocked = sorted(
                    {
                        *blocked,
                        *(
                            reason
                            for gate in venue_gates.values()
                            for reason in gate.blocked_reasons
                            if reason != "no_orders" or gate.can_sync
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
                from mm.core.payload_schema import joint_target_book_event

                await self._publish(
                    self.config.normalized_target_books_topic,
                    joint_target_book_event(
                        decision.joint_target_book,
                        exchange=frame.local.exchange,
                        symbol=frame.local.symbol,
                        t_end_ms=frame.local.t_end_ms,
                        frame_step=self._frame_step,
                    ),
                )
            for action in approved:
                await self._publish(self.config.normalized_action_topic, action)
            ctx.record_stage_latency("publish", _elapsed_ms(publish_started))

            execute_started = time.monotonic_ns()
            execution_report: ExecutionSyncReport | None = None
            sync_executed = executed
            if execution_allowed_by_risk:
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
                ctx.record_stage_latency("cancel", float(execution_report.cancel_ms))
                ctx.record_stage_latency("place", float(execution_report.place_ms))
                ctx.apply_execution_to_pipeline(execution_report, executed=sync_executed)

            if self.bus is not None:
                await self.bus.publish(
                    f"strategy:journal:{self.config.actor_id}",
                    {
                        "actor_id": self.config.actor_id,
                        "exchange": frame.local.exchange,
                        "symbol": frame.local.symbol,
                        "t_end_ms": frame.local.t_end_ms,
                        "frame_id": frame.frame_id,
                        "decision_id": decision.decision_id,
                        "target_book_id": decision.target_book_id,
                        "global_status": frame.global_status,
                        "proposed_actions": len(proposed),
                        "approved_actions": len(approved),
                        "executed": sync_executed,
                        "latency_ms": (time.monotonic_ns() - t0) / 1_000_000,
                    },
                )
            return approved
        except Exception as exc:
            if "risk" not in ctx.timing.stage_timings_ms:
                ctx.record_stage_latency("risk", _elapsed_ms(risk_started) if "risk_started" in locals() else 0.0)
            ctx.record_controller_error(exc)
            await self._publish_controller_error(frame, exc)
            return approved
        finally:
            ctx.timing.latency_ms = (time.monotonic_ns() - t0) / 1_000_000
            await self._publish_diagnostics(ctx)
            await self._heartbeat.publish(
                frame_id=frame.frame_id,
                frame_step=self._frame_step,
                decided=ctx.decision is not None,
                decision_latency_ms=(
                    ctx.timing.stage_timings_ms.get("decide") if ctx.timing is not None else None
                ),
            )

    async def _decide_with_timeout(
        self,
        frame: StrategyInputFrame,
        *,
        frame_step: int,
    ) -> StrategyDecision:
        timeout_ms = int(self.config.decision_timeout_ms)
        if timeout_ms <= 0:
            return self._engine.decide(frame, frame_step=frame_step)
        if self._decide_task is not None:
            if not self._decide_task.done():
                raise OptimizerDecisionInProgress(
                    "OptimizerDecisionInProgress: previous optimizer decision is still running; "
                    "holding current orders"
                )
            self._decide_task = None

        # Reset the cancel event and wire it into the optimizer backend so
        # a timeout can interrupt the Adam loop cooperatively. The pool
        # is dedicated (max_workers=1) so a wedged call cannot saturate
        # the default executor.
        self._decide_cancel_event.clear()
        backend = getattr(self._engine, "_optimizer_backend", None)
        if backend is not None and hasattr(backend, "set_cancel_event"):
            backend.set_cancel_event(self._decide_cancel_event)

        loop = asyncio.get_running_loop()
        try:
            future = self._decide_pool.submit(
                self._engine.decide, frame, frame_step=frame_step
            )
        except RuntimeError as exc:
            raise OptimizerDecisionInProgress(
                "OptimizerDecisionInProgress: decide pool is shut down"
            ) from exc
        self._decide_task = asyncio.ensure_future(
            asyncio.wrap_future(future, loop=loop)
        )
        timed_out = False
        try:
            return await asyncio.wait_for(
                asyncio.shield(self._decide_task),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError as exc:
            timed_out = True
            # Signal the optimizer to stop gracefully. The Adam loop
            # checks the event at the top of every epoch and returns the
            # best-so-far result, so the pool worker is freed.
            self._decide_cancel_event.set()
            self._decide_task.add_done_callback(self._on_decide_task_done)
            raise OptimizerDecisionTimeout(
                f"OptimizerDecisionTimeout: exceeded {timeout_ms}ms decision_timeout_ms; "
                "holding current orders"
            ) from exc
        finally:
            if not timed_out:
                self._decide_task = None

    def _on_decide_task_done(self, task: asyncio.Task) -> None:
        if self._decide_task is task:
            self._decide_task = None
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log_event(
                "optimizer_decision_task_failed",
                {
                    "actor_id": self.config.actor_id,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )

    async def _sync_joint_with_timeout(
        self,
        execution_book,
        venue_gates: dict,
        ctx: DecisionContext,
    ) -> ExecutionSyncReport | None:
        timeout_ms = int(self.config.placement_timeout_ms)
        started_ns = time.monotonic_ns()
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
                blocked_reasons=[*ctx.execution_state.blocked_reasons, "execution_sync_timeout"],
            )
            await self._publish_execution_error(ctx, message, latency_ms=_elapsed_ms(started_ns))
            return None
        except Exception as exc:
            message = f"ExecutionSyncError: {type(exc).__name__}: {exc}"
            ctx.record_warning(message)
            ctx.record_execution_gates(
                executed=False,
                blocked_reasons=[*ctx.execution_state.blocked_reasons, "execution_sync_error"],
            )
            await self._publish_execution_error(ctx, message, latency_ms=_elapsed_ms(started_ns))
            return None

    async def _publish_execution_error(
        self,
        ctx: DecisionContext,
        message: str,
        *,
        latency_ms: float | None = None,
    ) -> None:
        if self.bus is None:
            return
        from mm.core.payload_schema import execution_sync_failed_event

        payload = execution_sync_failed_event(
            actor_id=self.config.actor_id,
            exchange=ctx.frame.local.exchange,
            symbol=ctx.frame.local.symbol,
            t_end_ms=ctx.frame.t_end_ms,
            frame_step=ctx.frame_step,
            frame_id=ctx.frame.frame_id,
            decision_id=ctx.decision.decision_id if ctx.decision is not None else "",
            target_book_id=ctx.decision.target_book_id if ctx.decision is not None else "",
            error=message.split(":", 1)[0],
            message=message,
            timestamp_ms=int(time.time() * 1000),
            latency_ms=latency_ms,
        )
        log_event("execution_sync_failed", payload)
        await self.bus.publish("execution:errors", payload)

    async def _publish_controller_fallback(
        self,
        frame: StrategyInputFrame,
        *,
        reason: str,
        message: str,
    ) -> None:
        if self.bus is not None:
            await self._error_publisher.publish(
                kind=reason,
                message=message,
                frame_id=frame.frame_id,
                frame_step=self._frame_step,
            )
        await self._publish(
            self.config.normalized_decisions_topic,
            {
                "type": "ControllerFallback",
                "actor_id": self.config.actor_id,
                "exchange": frame.local.exchange,
                "symbol": frame.local.symbol,
                "t_end_ms": frame.local.t_end_ms,
                "frame_step": self._frame_step,
                "frame_id": frame.frame_id,
                "reason": reason,
                "fallback_policy": "hold_current_orders",
                "message": message,
            },
        )

    async def _publish_controller_error(
        self,
        frame: StrategyInputFrame,
        exc: BaseException,
    ) -> None:
        if self.bus is not None:
            await self._error_publisher.publish(
                kind="controller_error",
                message=f"{type(exc).__name__}: {exc}",
                frame_id=frame.frame_id,
                frame_step=self._frame_step,
                exc=exc,
            )
        await self._publish(
            self.config.normalized_decisions_topic,
            {
                "type": "ControllerError",
                "actor_id": self.config.actor_id,
                "exchange": frame.local.exchange,
                "symbol": frame.local.symbol,
                "t_end_ms": frame.local.t_end_ms,
                "frame_id": frame.frame_id,
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
        *,
        execution_allowed_by_risk: bool,
        venue_gates: dict,
    ) -> bool:
        if not execution_allowed_by_risk:
            return False
        return any(gate.live and gate.can_sync for gate in venue_gates.values())

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
        diagnostics = ctx.to_diagnostics()
        if ctx.last_portfolio_mark is not None:
            self._last_portfolio_mark = float(ctx.last_portfolio_mark)
        await self._diagnostics.publish(diagnostics)

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


def _lineage_ids(
    *,
    actor_id: str,
    exchange: str,
    symbol: str,
    t_end_ms: int,
    frame_step: int,
    existing_frame_id: str = "",
) -> dict[str, str]:
    frame_id = existing_frame_id.strip() or _lineage_id(
        "frame",
        actor_id,
        exchange,
        symbol,
        str(int(t_end_ms)),
        str(int(frame_step)),
    )
    decision_id = _lineage_id("decision", frame_id)
    target_book_id = _lineage_id("target", frame_id)
    return {
        "frame_id": frame_id,
        "decision_id": decision_id,
        "target_book_id": target_book_id,
    }


def _with_decision_lineage(
    decision: StrategyDecision,
    lineage: dict[str, str],
) -> StrategyDecision:
    target_book = (
        _with_target_book_lineage(decision.joint_target_book, lineage)
        if decision.joint_target_book is not None
        else None
    )
    return replace(
        decision,
        frame_id=lineage["frame_id"],
        decision_id=lineage["decision_id"],
        target_book_id=lineage["target_book_id"] if target_book is not None else "",
        joint_target_book=target_book,
    )


def _with_target_book_lineage(
    book,
    lineage: dict[str, str],
):
    local = _with_venue_book_lineage(book.local, role="local", lineage=lineage)
    global_book = (
        _with_venue_book_lineage(book.global_, role="global", lineage=lineage)
        if book.global_ is not None
        else None
    )
    return replace(
        book,
        local=local,
        global_=global_book,
        frame_id=lineage["frame_id"],
        decision_id=lineage["decision_id"],
        target_book_id=lineage["target_book_id"],
    )


def _with_venue_book_lineage(
    book,
    *,
    role: str,
    lineage: dict[str, str],
):
    exchange = str(getattr(book, "exchange", role))
    execution_plan_id = _lineage_id(
        "execplan",
        lineage["target_book_id"],
        role,
        exchange,
    )
    return replace(
        book,
        frame_id=lineage["frame_id"],
        decision_id=lineage["decision_id"],
        target_book_id=lineage["target_book_id"],
        execution_plan_id=execution_plan_id,
    )


def _lineage_id(prefix: str, *parts: str) -> str:
    normalized = [
        str(part or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        for part in parts
    ]
    return ":".join((prefix, *normalized))


def _fallback_reason(exc: BaseException) -> str:
    if isinstance(exc, OptimizerDecisionTimeout):
        return "optimizer_decision_timeout"
    if isinstance(exc, OptimizerDecisionInProgress):
        return "optimizer_decision_in_progress"
    return "controller_fallback"


def _risk_rejection_rows(
    ctx: DecisionContext,
    rejection_counts: dict[str, int],
    *,
    fallback_orders: tuple[Order, ...] = (),
) -> tuple[dict[str, Any], ...]:
    if not rejection_counts:
        return ()
    source_orders: tuple[dict[str, Any], ...] = ()
    report = ctx.pipeline_state.report
    if report is not None:
        source_orders = (
            report.risk_approved_target.orders
            or report.capped.orders
            or report.target_book.orders
        )
    if not source_orders and fallback_orders:
        source_orders = tuple(order_summary_dict(order) for order in fallback_orders)
    if not source_orders:
        return ()
    rows: list[dict[str, Any]] = []
    for reason in sorted(rejection_counts):
        reason_orders = _risk_rejection_source_orders(source_orders, reason)
        for order in reason_orders:
            rows.append({**dict(order), "reason": reason, "stage": "risk"})
    return tuple(rows)


def _risk_rejection_source_orders(
    source_orders: tuple[dict[str, Any], ...],
    reason: str,
) -> tuple[dict[str, Any], ...]:
    sides = _risk_rejection_sides(reason)
    if sides is None:
        return source_orders
    filtered = tuple(
        dict(order)
        for order in source_orders
        if str(order.get("side", "")).lower().strip() in sides
    )
    return filtered or source_orders


def _risk_rejection_sides(reason: str) -> set[str] | None:
    normalized = reason.removeprefix("global_")
    if normalized in {
        "insufficient_quote_for_bid_orders",
        "insufficient_quote_for_buys",
        "min_cash",
        "max_inventory",
    }:
        return {"buy"}
    if normalized in {
        "insufficient_base_for_ask_orders",
        "insufficient_base_for_sells",
        "min_inventory",
    }:
        return {"sell"}
    return None


def _local_risk_rejection_counts(rejection_counts: dict[str, int]) -> dict[str, int]:
    return {
        reason: count
        for reason, count in rejection_counts.items()
        if not reason.startswith("global_")
    }


def _coerce_joint_strategy_frame(raw_frame: Any) -> JointStrategyFrame | None:
    if isinstance(raw_frame, JointStrategyFrame):
        return raw_frame
    return joint_strategy_frame_from_payload(raw_frame)


def _joint_frame_log_context(frame: Any) -> dict[str, Any]:
    if isinstance(frame, JointStrategyFrame):
        return {
            "exchange": frame.local_market.exchange,
            "symbol": frame.local_market.symbol,
            "t_end_ms": frame.local_market.t_end_ms,
        }
    if isinstance(frame, dict):
        local = frame.get("local_market")
        if isinstance(local, dict):
            return {
                "exchange": local.get("exchange"),
                "symbol": local.get("symbol"),
                "t_end_ms": local.get("t_end_ms"),
            }
    return {}


def _elapsed_ms(started_ns: int) -> float:
    return (time.monotonic_ns() - started_ns) / 1_000_000
