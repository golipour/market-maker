from __future__ import annotations

import asyncio
import math
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.models.execution import Action
from mm.core.models.joint_strategy import JointStrategyFrame
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
        if isinstance(event, Action):
            self._observe_action(topic, event)
            return
        if isinstance(event, ExecutionSnapshot):
            self._observe_execution_snapshot(event)
            return
        if isinstance(event, dict):
            kind = str(event.get("type", ""))
            if kind == "ControllerDiagnostics":
                self._observe_controller_diagnostics(event)
                return
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
            if kind.startswith(("Execution", "Place", "Cancel")):
                self._observe_execution_event(topic, event)
                if kind.endswith("Error") or "error" in event:
                    self._observe_error(topic, event)
                return
            if kind == "RiskActionsFiltered":
                self._observe_risk_event(event)
                return
            if kind.endswith("Error") or "error" in event:
                self._observe_error(topic, event)
                return
            if event.get("exchange") is not None and "live_order_count" in event:
                self._observe_execution_snapshot_dict(event)
                return

    def _observe_joint_strategy_frame(self, topic: str, frame: JointStrategyFrame) -> None:
        local = frame.local_market
        sync = frame.sync_status
        labels = {
            "exchange": local.exchange,
            "symbol": local.symbol,
            "mode": frame.mode,
            "global_status": sync.global_status,
        }
        _ = topic
        self._last_local_frame_by_exchange[local.exchange.lower().strip()] = (
            int(local.t_end_ms),
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
                "exchange": local.exchange,
                "symbol": local.symbol,
                "t_end_ms": local.t_end_ms,
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
                    help="Wall ms from JointStrategyFrame.created_at_ms to ExecutionSyncRequested",
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

    def _observe_controller_diagnostics(self, payload: dict[str, Any]) -> None:
        labels = _controller_labels(payload)
        frame = _dict(payload.get("frame"))
        frame_age_ms = _as_float(frame.get("frame_age_ms"))
        if frame_age_ms is not None:
            gauge(
                obs_schema.MM_FRAME_AGE_MS,
                frame_age_ms,
                labels={k: labels[k] for k in ("exchange", "symbol")},
                help="Wall ms between frame creation and diagnostics publication",
            )

        sync_status = _dict(frame.get("sync_status"))
        for field in (
            "local_market_ready",
            "global_market_ready",
            "local_private_ready",
            "global_private_ready",
            "ledger_fresh",
        ):
            if field in sync_status:
                gauge(
                    obs_schema.MM_FRAME_SYNC_READY,
                    1.0 if bool(sync_status.get(field)) else 0.0,
                    labels={**{k: labels[k] for k in ("exchange", "symbol")}, "field": field},
                    help="Controller frame sync readiness flags",
                )

        optimizer = _dict(payload.get("optimizer"))
        timing = _dict(optimizer.get("timing"))
        for stage, value_ms in _timing_ms(timing).items():
            histogram_observe(
                obs_schema.MM_CONTROLLER_STAGE_LATENCY_MS,
                value_ms,
                labels={**labels, "stage": stage},
                help="Controller or optimizer stage latency in milliseconds",
            )

        for warning in payload.get("warnings") or ():
            reason = _bounded_reason(warning)
            counter_inc(
                obs_schema.MM_CONTROLLER_WARNINGS_TOTAL,
                labels={**labels, "reason": reason},
                help="Controller diagnostics warnings",
            )

        models = _dict(payload.get("models"))
        for model_key, score_any in models.items():
            score = _dict(score_any)
            model = str(score.get("model") or model_key)
            model_labels = {**labels, "model": model}
            nll = _as_float(score.get("nll"))
            if nll is not None:
                histogram_observe(
                    obs_schema.MM_MODEL_NLL,
                    nll,
                    labels=model_labels,
                    help="Pre-update model negative log likelihood",
                )
            log_likelihood = _as_float(score.get("log_likelihood"))
            if log_likelihood is not None:
                gauge(
                    obs_schema.MM_MODEL_LOG_LIKELIHOOD,
                    log_likelihood,
                    labels=model_labels,
                    help="Model log likelihood for latest observation",
                )
            residual = _as_float(score.get("standardized_residual"))
            if residual is not None:
                gauge(
                    obs_schema.MM_MODEL_STANDARDIZED_RESIDUAL,
                    residual,
                    labels=model_labels,
                    help="Model standardized residual for latest observation",
                )
            skip_reason = score.get("skip_reason")
            if skip_reason:
                counter_inc(
                    obs_schema.MM_MODEL_SCORE_SKIPPED_TOTAL,
                    labels={**model_labels, "reason": _bounded_reason(skip_reason)},
                    help="Model score skipped by reason",
                )

        action_pipeline = _dict(payload.get("action_pipeline"))
        for stage, stage_any in action_pipeline.items():
            stage_payload = _dict(stage_any)
            if not stage_payload:
                continue
            stage_labels = {**labels, "stage": str(stage)}
            order_count = _as_float(stage_payload.get("order_count"))
            if order_count is not None:
                gauge(
                    obs_schema.MM_ACTION_STAGE_ORDERS,
                    order_count,
                    labels={**stage_labels, "side": "all"},
                    help="Order count at action pipeline stage",
                )
            for side in ("buy", "sell"):
                qty = _as_float(stage_payload.get(f"{side}_qty"))
                if qty is not None:
                    gauge(
                        obs_schema.MM_ACTION_STAGE_QTY,
                        qty,
                        labels={**stage_labels, "side": side},
                        help="Base quantity at action pipeline stage",
                    )
                notional = _as_float(stage_payload.get(f"{side}_notional"))
                if notional is not None:
                    gauge(
                        obs_schema.MM_ACTION_STAGE_NOTIONAL,
                        notional,
                        labels={**stage_labels, "side": side},
                        help="Quote notional at action pipeline stage",
                    )
            for reason, count in _dict(stage_payload.get("rejection_counts")).items():
                inc = _as_float(count)
                if inc is None or inc <= 0:
                    continue
                counter_inc(
                    obs_schema.MM_ACTION_REJECTED_TOTAL,
                    inc,
                    labels={**stage_labels, "reason": _bounded_reason(reason)},
                    help="Action pipeline rejections by stage and reason",
                )

        risk = _dict(payload.get("risk"))
        if risk:
            gauge(
                obs_schema.MM_RISK_APPROVED,
                1.0 if bool(risk.get("approved")) else 0.0,
                labels=labels,
                help="Risk approval state for latest controller frame",
            )
            for reason, count in _dict(risk.get("rejection_counts")).items():
                inc = _as_float(count)
                if inc is None or inc <= 0:
                    continue
                counter_inc(
                    obs_schema.MM_RISK_REJECT_TOTAL,
                    inc,
                    labels={**labels, "reason": _bounded_reason(reason)},
                    help="Risk rejections by reason",
                )
            for phase in ("exposure_before", "exposure_after"):
                for field, value in _dict(risk.get(phase)).items():
                    v = _as_float(value)
                    if v is None:
                        continue
                    gauge(
                        obs_schema.MM_RISK_EXPOSURE,
                        v,
                        labels={**labels, "phase": phase.replace("exposure_", ""), "field": str(field)},
                        help="Risk exposure fields before and after approval",
                    )

        gates = _dict(payload.get("execution_gates"))
        for reason in gates.get("blocked_reasons") or ():
            counter_inc(
                obs_schema.MM_EXECUTION_GATE_BLOCK_TOTAL,
                labels={**labels, "reason": _bounded_reason(reason)},
                help="Live execution gate blocked reasons",
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
        if etype == "ExecutionSyncCompleted":
            counter_inc(
                obs_schema.MM_EXECUTION_SYNC_COMPLETED_TOTAL,
                labels={"exchange": labels["exchange"]},
                help="Order manager sync rounds completed",
            )
            latency = payload.get("latency_ms")
            if latency is not None:
                self._sync_pending_ts_ms.pop(labels["exchange"].lower().strip(), None)
                histogram_observe(
                    obs_schema.MM_ORDER_SYNC_LATENCY_MS,
                    max(0.0, float(latency)),
                    labels={"exchange": labels["exchange"]},
                    help="Wall time for completed order manager sync rounds",
                )
        if etype == "ExecutionSyncFailed":
            self._sync_pending_ts_ms.pop(labels["exchange"].lower().strip(), None)
            counter_inc(
                obs_schema.MM_EXECUTION_SYNC_FAILED_TOTAL,
                labels={
                    "exchange": labels["exchange"],
                    "error": str(payload.get("error", "unknown")),
                },
                help="Order manager sync rounds failed",
            )
            latency = payload.get("latency_ms")
            if latency is not None:
                histogram_observe(
                    obs_schema.MM_ORDER_SYNC_LATENCY_MS,
                    max(0.0, float(latency)),
                    labels={"exchange": labels["exchange"]},
                    help="Wall time for failed order manager sync rounds",
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

    def _observe_risk_event(self, payload: dict[str, Any]) -> None:
        labels = {
            "actor_id": str(payload.get("actor_id", "")),
            "exchange": str(payload.get("exchange", "")),
            "symbol": str(payload.get("symbol", "")),
        }
        proposed = _as_float(payload.get("proposed_actions")) or 0.0
        approved = _as_float(payload.get("approved_actions")) or 0.0
        if approved < proposed:
            counter_inc(
                obs_schema.MM_RISK_REJECT_TOTAL,
                max(1.0, proposed - approved),
                labels={**labels, "reason": "actions_filtered"},
                help="Risk rejections by reason",
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


def _controller_labels(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "actor_id": str(payload.get("actor_id", "")),
        "exchange": str(payload.get("exchange", "")),
        "symbol": str(payload.get("symbol", "")),
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _bounded_reason(value: Any) -> str:
    text = str(value or "unknown").strip()
    if not text:
        return "unknown"
    text = text.split(":", 1)[0]
    chars: list[str] = []
    last_underscore = False
    for ch in text.lower()[:96]:
        ok = ch.isalnum()
        if ok:
            chars.append(ch)
            last_underscore = False
        elif not last_underscore:
            chars.append("_")
            last_underscore = True
    reason = "".join(chars).strip("_")
    return reason or "unknown"


def _timing_ms(timing: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in timing.items():
        v = _as_float(value)
        if v is None:
            continue
        stage = str(key)
        if stage.endswith("_sec"):
            out[stage[:-4]] = v * 1000.0
        elif stage.endswith("_ms"):
            out[stage[:-3]] = v
        elif stage in {"best_epoch"}:
            continue
    return out
