from __future__ import annotations

import asyncio
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.models.execution import Action
from mm.core.models.joint_strategy import JointStrategyFrame
from mm.core.models.strategy import StrategyInputFrame
from mm.execution.order_manager import ExecutionSnapshot
from mm.observability import schema as obs_schema
from mm.observability.metrics import counter_inc, gauge, histogram_observe
from mm.observability.logs import log_event
from mm.observability.traces import span


class RuntimeObservabilitySubscriber:
    """Observes domain/runtime events and emits telemetry outside business logic."""

    def __init__(self, *, bus: AbstractEventBus, topics: tuple[str, ...]):
        self.bus = bus
        self.topics = tuple(dict.fromkeys(topic for topic in topics if topic))
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._sync_pending_ts_ms: dict[str, int] = {}
        # Latest local frame per exchange: (t_end_ms, created_at_ms wall) for latency vs execution sync.
        self._last_local_frame_by_exchange: dict[str, tuple[int, int]] = {}

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
        if isinstance(event, JointStrategyFrame):
            self._observe_joint_strategy_frame(topic, event)
            return
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
        primary = frame.local
        labels = {"exchange": primary.exchange, "symbol": primary.symbol, "global_status": frame.global_status}
        _ = topic
        with span("mm.frame_assembler.build", exchange=primary.exchange, symbol=primary.symbol):
            counter_inc(obs_schema.MM_STRATEGY_FRAMES_TOTAL, labels=labels, help="Strategy frames assembled")
            self._last_local_frame_by_exchange[primary.exchange.lower().strip()] = (
                int(primary.t_end_ms),
                int(frame.created_at_ms),
            )
            if frame.global_lag_ms is not None:
                gauge(
                    "mm_global_lag_ms",
                    float(frame.global_lag_ms),
                    labels=labels,
                    help="Selected global event lag",
                )
            log_event(
                obs_schema.EVENT_STRATEGY_FRAME,
                {
                    "exchange": primary.exchange,
                    "symbol": primary.symbol,
                    "t_end_ms": primary.t_end_ms,
                    "global_status": frame.global_status,
                    "global_lag_ms": frame.global_lag_ms,
                },
            )

    def _observe_joint_strategy_frame(self, topic: str, frame: JointStrategyFrame) -> None:
        primary = frame.local_market
        sync = frame.sync_status
        labels = {
            "exchange": primary.exchange,
            "symbol": primary.symbol,
            "mode": frame.mode,
            "global_status": sync.global_status,
        }
        _ = topic
        self._last_local_frame_by_exchange[primary.exchange.lower().strip()] = (
            int(primary.t_end_ms),
            int(frame.created_at_ms),
        )
        counter_inc(
            "mm_joint_strategy_frames_total",
            labels=labels,
            help="Ledger-backed joint strategy frames assembled",
        )
        if sync.latency_skew_ms is not None:
            gauge(
                "mm_latency_skew_ms",
                float(sync.latency_skew_ms),
                labels=labels,
                help="Local/global market event latency skew",
            )
        gauge(
            "mm_frame_local_market_ready",
            1.0 if sync.local_market_ready else 0.0,
            labels=labels,
            help="Local market readiness for joint frame",
        )
        gauge(
            "mm_frame_global_market_ready",
            1.0 if sync.global_market_ready else 0.0,
            labels=labels,
            help="Global market readiness for joint frame",
        )
        gauge(
            "mm_frame_ledger_fresh",
            1.0 if sync.ledger_fresh else 0.0,
            labels=labels,
            help="Joint ledger freshness for frame assembly",
        )
        log_event(
            "joint_strategy_frame",
            {
                "exchange": primary.exchange,
                "symbol": primary.symbol,
                "t_end_ms": primary.t_end_ms,
                "mode": frame.mode,
                "sync_status": sync.as_payload(),
            },
        )

    def _observe_execution_sync_requested(self, payload: dict[str, Any]) -> None:
        ex = str(payload.get("exchange", "")).strip()
        ts = payload.get("timestamp_ms")
        if ex and ts is not None:
            sync_ts = int(ts)
            exchange_key = ex.lower().strip()
            self._sync_pending_ts_ms[exchange_key] = sync_ts
            last = self._last_local_frame_by_exchange.get(exchange_key)
            if last is not None:
                t_end_ms, created_at_ms = last
                histogram_observe(
                    obs_schema.MM_FRAME_TO_EXEC_SYNC_LATENCY_MS,
                    max(0.0, float(sync_ts - created_at_ms)),
                    labels={"exchange": ex},
                    help="Wall ms from StrategyInputFrame.created_at_ms to ExecutionSyncRequested",
                )
                histogram_observe(
                    obs_schema.MM_MARKET_WINDOW_END_TO_EXEC_SYNC_LATENCY_MS,
                    max(0.0, float(sync_ts - t_end_ms)),
                    labels={"exchange": ex},
                    help="Ms from local MarketEvent.t_end_ms to ExecutionSyncRequested (runtime wall clock)",
                )

    def _observe_execution_snapshot(self, snap: ExecutionSnapshot) -> None:
        self._finish_sync_latency(str(snap.exchange), snap.timestamp_ms)
        gauge(
            obs_schema.MM_EXECUTION_LIVE_ORDERS,
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
                obs_schema.MM_EXECUTION_LIVE_ORDERS,
                float(lc),
                labels={"exchange": ex},
                help="Current live orders by exchange",
            )

    def _observe_execution_snapshot_compat(self, exchange: Any, live_order_count: Any, timestamp_ms: Any) -> None:
        ex = str(exchange)
        if timestamp_ms is not None:
            self._finish_sync_latency(ex, int(timestamp_ms))
        gauge(
            obs_schema.MM_EXECUTION_LIVE_ORDERS,
            float(live_order_count),
            labels={"exchange": ex},
            help="Current live orders by exchange",
        )

    def _finish_sync_latency(self, exchange: str, snapshot_ts_ms: int) -> None:
        pending = self._sync_pending_ts_ms.pop(exchange.lower().strip(), None)
        if pending is None:
            return
        latency_ms = max(0.0, float(snapshot_ts_ms - pending))
        histogram_observe(
            obs_schema.MM_ORDER_SYNC_LATENCY_MS,
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
                obs_schema.MM_CONTROLLER_LATENCY_MS,
                float(payload.get("latency_ms", 0.0) or 0.0),
                labels=labels,
                help="Controller decision latency",
            )
            gauge(
                obs_schema.MM_CONTROLLER_ORDER_COUNT,
                float(payload.get("order_count", 0.0) or 0.0),
                labels=labels,
                help="Controller order count",
            )
            gauge(
                obs_schema.MM_INVENTORY,
                float(payload.get("inventory", 0.0) or 0.0),
                labels=labels,
                help="Strategy inventory",
            )
            gauge(
                obs_schema.MM_CASH,
                float(payload.get("cash", 0.0) or 0.0),
                labels=labels,
                help="Strategy cash",
            )
            counter_inc(obs_schema.MM_CONTROLLER_DECISIONS_TOTAL, labels=labels, help="Controller decisions")
            log_event(
                obs_schema.EVENT_STRATEGY_DECISION,
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
            obs_schema.MM_STRATEGY_ACTION_ORDER_COUNT,
            float(len(action.orders)),
            labels=labels,
            help="Orders requested by strategy action",
        )

    def _observe_execution_event(self, topic: str, payload: dict[str, Any]) -> None:
        labels = {
            "exchange": str(payload.get("exchange", "")),
            "event_type": str(payload.get("type", "")),
        }
        counter_inc(obs_schema.MM_EXECUTION_EVENTS_TOTAL, labels=labels, help="Execution events observed")
        etype = str(payload.get("type", ""))
        if etype == "ExecutionSyncRequested":
            counter_inc(
                obs_schema.MM_ORDER_SYNC_TOTAL,
                labels={"exchange": labels["exchange"]},
                help="Order manager sync rounds observed",
            )
        if etype == "PlaceOrdersAccepted":
            oids = payload.get("order_ids") or ()
            n = len(oids) if isinstance(oids, (list, tuple)) else 0
            if n:
                counter_inc(
                    obs_schema.MM_ORDER_PLACE_ACCEPTED_TOTAL,
                    float(n),
                    labels={"exchange": labels["exchange"]},
                    help="Venue-accepted new orders (len order_ids)",
                )
        if etype == "CancelOrdersAccepted":
            oids = payload.get("order_ids") or ()
            n = len(oids) if isinstance(oids, (list, tuple)) else 0
            if n:
                counter_inc(
                    obs_schema.MM_ORDER_CANCEL_ACCEPTED_TOTAL,
                    float(n),
                    labels={"exchange": labels["exchange"]},
                    help="Venue-accepted cancels (len order_ids)",
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
            obs_schema.MM_ORDER_VALIDATION_REJECTED_TOTAL,
            labels=labels,
            help="Orders dropped before REST due to venue limits",
        )
        log_event(obs_schema.EVENT_ORDER_VALIDATION_REJECTED, payload)

    def _observe_error(self, topic: str, payload: dict[str, Any]) -> None:
        labels = {
            "topic": topic,
            "error": str(payload.get("error", payload.get("type", "unknown"))),
        }
        counter_inc(obs_schema.MM_RUNTIME_ERRORS_TOTAL, labels=labels, help="Runtime errors observed")
        log_event(obs_schema.EVENT_RUNTIME_ERROR, {"topic": topic, **payload})
