from __future__ import annotations

import asyncio
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.models.execution import Action
from mm.core.models.strategy import StrategyInputFrame
from mm.execution.order_manager import ExecutionSnapshot
from mm.observability import counter_inc, gauge, histogram_observe, log_event, span


class RuntimeObservabilitySubscriber:
    """Observes domain/runtime events and emits telemetry outside business logic."""

    def __init__(self, *, bus: AbstractEventBus, topics: tuple[str, ...]):
        self.bus = bus
        self.topics = tuple(dict.fromkeys(topic for topic in topics if topic))
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._sync_pending_ts_ms: dict[str, int] = {}

    async def start(self) -> None:
        self._stopped.clear()
        self._tasks = [asyncio.create_task(self._run_topic(topic)) for topic in self.topics]

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _run_topic(self, topic: str) -> None:
        queue = self.bus.subscribe_queue(topic)
        try:
            while not self._stopped.is_set():
                event = await queue.get()
                self.observe(topic, event)
        except asyncio.CancelledError:
            raise

    def observe(self, topic: str, event: Any) -> None:
        if isinstance(event, StrategyInputFrame):
            self._observe_strategy_input_frame(topic, event)
            return
        if isinstance(event, Action):
            self._observe_action(topic, event)
            return
        if isinstance(event, ExecutionSnapshot):
            self._observe_execution_snapshot(event)
            return
        if isinstance(event, dict):
            kind = str(event.get("type", ""))
            if kind in {"StrategyDecision", "ControllerDecision"}:
                self._observe_strategy_decision(event)
                return
            if kind == "OrderValidationRejected":
                self._observe_order_validation_rejection(event)
                return
            if kind == "ExecutionSyncRequested":
                self._observe_execution_sync_requested(event)
                self._observe_execution_event(topic, event)
                return
            if kind.endswith("Error") or "error" in event:
                self._observe_error(topic, event)
                return
            if kind.startswith(("Execution", "Place", "Cancel")):
                self._observe_execution_event(topic, event)
                return
            if event.get("exchange") is not None and "live_order_count" in event:
                self._observe_execution_snapshot_dict(event)
                return
        live_order_count = getattr(event, "live_order_count", None)
        exchange = getattr(event, "exchange", None)
        timestamp_ms = getattr(event, "timestamp_ms", None)
        if live_order_count is not None and exchange is not None:
            self._observe_execution_snapshot_compat(exchange, live_order_count, timestamp_ms)

    def _observe_strategy_input_frame(self, topic: str, frame: StrategyInputFrame) -> None:
        primary = frame.primary
        labels = {"exchange": primary.exchange, "symbol": primary.symbol, "hedge_status": frame.hedge_status}
        _ = topic
        with span("mm.frame_assembler.build", exchange=primary.exchange, symbol=primary.symbol):
            counter_inc("mm_strategy_frames_total", labels=labels, help="Strategy frames assembled")
            if frame.hedge_lag_ms is not None:
                gauge(
                    "mm_hedge_lag_ms",
                    float(frame.hedge_lag_ms),
                    labels=labels,
                    help="Selected hedge event lag",
                )
            log_event(
                "strategy_frame",
                {
                    "exchange": primary.exchange,
                    "symbol": primary.symbol,
                    "t_end_ms": primary.t_end_ms,
                    "hedge_status": frame.hedge_status,
                    "hedge_lag_ms": frame.hedge_lag_ms,
                },
            )

    def _observe_execution_sync_requested(self, payload: dict[str, Any]) -> None:
        ex = str(payload.get("exchange", "")).strip()
        ts = payload.get("timestamp_ms")
        if ex and ts is not None:
            self._sync_pending_ts_ms[ex] = int(ts)

    def _observe_execution_snapshot(self, snap: ExecutionSnapshot) -> None:
        self._finish_sync_latency(str(snap.exchange), snap.timestamp_ms)
        gauge(
            "mm_execution_live_orders",
            float(snap.live_order_count),
            labels={"exchange": str(snap.exchange)},
            help="Current live orders by exchange",
        )

    def _observe_execution_snapshot_dict(self, payload: dict[str, Any]) -> None:
        ex = str(payload.get("exchange", ""))
        ts_ms = payload.get("timestamp_ms")
        if ts_ms is not None:
            self._finish_sync_latency(ex, int(ts_ms))
        lc = payload.get("live_order_count")
        if lc is not None:
            gauge(
                "mm_execution_live_orders",
                float(lc),
                labels={"exchange": ex},
                help="Current live orders by exchange",
            )

    def _observe_execution_snapshot_compat(self, exchange: Any, live_order_count: Any, timestamp_ms: Any) -> None:
        ex = str(exchange)
        if timestamp_ms is not None:
            self._finish_sync_latency(ex, int(timestamp_ms))
        gauge(
            "mm_execution_live_orders",
            float(live_order_count),
            labels={"exchange": ex},
            help="Current live orders by exchange",
        )

    def _finish_sync_latency(self, exchange: str, snapshot_ts_ms: int) -> None:
        pending = self._sync_pending_ts_ms.pop(exchange, None)
        if pending is None:
            return
        latency_ms = max(0.0, float(snapshot_ts_ms - pending))
        histogram_observe(
            "mm_order_sync_latency_ms",
            latency_ms,
            labels={"exchange": exchange},
            help="Wall time from ExecutionSyncRequested timestamp to execution snapshot",
        )

    def _observe_strategy_decision(self, payload: dict[str, Any]) -> None:
        labels = {
            "actor_id": str(payload.get("actor_id", "")),
            "exchange": str(payload.get("exchange", "")),
            "symbol": str(payload.get("symbol", "")),
        }
        econ = payload.get("economics")
        if isinstance(econ, dict) and econ:
            self._maybe_persist_step_economics(payload, econ)
        with span("mm.controller.decision_observed", **labels):
            gauge(
                "mm_controller_latency_ms",
                float(payload.get("latency_ms", 0.0) or 0.0),
                labels=labels,
                help="Controller decision latency",
            )
            gauge(
                "mm_controller_order_count",
                float(payload.get("order_count", 0.0) or 0.0),
                labels=labels,
                help="Controller order count",
            )
            gauge(
                "mm_inventory",
                float(payload.get("inventory", 0.0) or 0.0),
                labels=labels,
                help="Actor inventory",
            )
            gauge(
                "mm_cash",
                float(payload.get("cash", 0.0) or 0.0),
                labels=labels,
                help="Actor cash",
            )
            counter_inc("mm_controller_decisions_total", labels=labels, help="Controller decisions")
            log_event(
                "controller_decision",
                {
                    "actor_id": labels["actor_id"],
                    "exchange": labels["exchange"],
                    "symbol": labels["symbol"],
                    "t_end_ms": payload.get("t_end_ms"),
                    "order_count": payload.get("order_count"),
                    "latency_ms": payload.get("latency_ms"),
                    "observe_only": payload.get("observe_only"),
                    "decision_hash": payload.get("decision_hash"),
                },
            )
            try:
                from mm.diagnostics_bridge import emit_controller_diagnostics

                emit_controller_diagnostics(payload)
            except Exception:
                return

    def _maybe_persist_step_economics(self, payload: dict[str, Any], econ: dict[str, Any]) -> None:
        try:
            from datetime import datetime, timezone

            from mm.persistence.step_economics_pg import persist_step_economics_row

            t_end_ms = int(econ.get("t_end_ms", payload.get("t_end_ms", 0)))
            ts = datetime.fromtimestamp(t_end_ms / 1000.0, tz=timezone.utc)
            persist_step_economics_row(
                t_fill=ts,
                t_end_ms=t_end_ms,
                step_counter=int(econ.get("step_counter", 0)),
                exchange=str(econ.get("exchange", payload.get("exchange", ""))),
                symbol=str(econ.get("symbol", payload.get("symbol", ""))),
                d_w=float(econ.get("d_w", 0.0)),
                spread_pnl=float(econ.get("spread_pnl", 0.0)),
                inventory_pnl=float(econ.get("inventory_pnl", 0.0)),
                cross_pnl=float(econ.get("cross_pnl", 0.0)),
                fee_pnl=float(econ.get("fee_pnl", 0.0)),
                wealth_pnl=float(econ.get("wealth_pnl", 0.0)),
            )
        except Exception:
            return

    def _observe_action(self, topic: str, action: Action) -> None:
        labels = {"actor_id": action.actor_id, "topic": topic}
        gauge(
            "mm_strategy_action_order_count",
            float(len(action.orders_to_create)),
            labels=labels,
            help="Orders requested by strategy action",
        )

    def _observe_execution_event(self, topic: str, payload: dict[str, Any]) -> None:
        labels = {
            "exchange": str(payload.get("exchange", "")),
            "event_type": str(payload.get("type", "")),
        }
        counter_inc("mm_execution_events_total", labels=labels, help="Execution events observed")
        if payload.get("type") == "ExecutionSyncRequested":
            counter_inc(
                "mm_order_sync_total",
                labels={"exchange": labels["exchange"]},
                help="Order manager sync rounds observed",
            )
        if "create_count" in payload:
            gauge(
                "mm_order_sync_create_count",
                float(payload.get("create_count", 0.0) or 0.0),
                labels={"exchange": labels["exchange"]},
                help="Orders requested for create",
            )
        if "cancel_count" in payload:
            gauge(
                "mm_order_sync_cancel_count",
                float(payload.get("cancel_count", 0.0) or 0.0),
                labels={"exchange": labels["exchange"]},
                help="Orders requested for cancel",
            )

    def _observe_order_validation_rejection(self, payload: dict[str, Any]) -> None:
        labels = {
            "exchange": str(payload.get("exchange", "")),
            "reason": str(payload.get("reason", "")),
        }
        counter_inc(
            "mm_order_validation_rejected_total",
            labels=labels,
            help="Orders dropped before REST due to venue limits",
        )
        log_event("order_validation_rejected", payload)

    def _observe_error(self, topic: str, payload: dict[str, Any]) -> None:
        labels = {
            "topic": topic,
            "error": str(payload.get("error", payload.get("type", "unknown"))),
        }
        counter_inc("mm_runtime_errors_total", labels=labels, help="Runtime errors observed")
        log_event("runtime_error", {"topic": topic, **payload})
