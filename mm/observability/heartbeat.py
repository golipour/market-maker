"""Per-actor heartbeat stream + a dedicated errors stream.

The runner publishes a heartbeat to ``strategy:heartbeat:<actor_id>`` once
per frame, regardless of whether decide succeeded. Anything that watches
data freshness (admin API, Grafana alerts, cluster health monitor) reads
this stream instead of inferring liveness from ControllerDiagnostics, which
silently stops arriving when the decide path errors out.

The errors stream is a first-class event with a stable schema, so alerts
do not have to parse JSON inside ControllerDiagnostics payloads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

from mm.bus.base import AbstractEventBus


HEARTBEAT_TOPIC_PREFIX = "strategy:heartbeat:"
ERRORS_TOPIC_PREFIX = "strategy:errors:"


def heartbeat_topic(actor_id: str) -> str:
    return f"{HEARTBEAT_TOPIC_PREFIX}{actor_id}"


def errors_topic(actor_id: str) -> str:
    return f"{ERRORS_TOPIC_PREFIX}{actor_id}"


@dataclass(slots=True)
class HeartbeatPublisher:
    """Emits per-frame heartbeat events for one actor."""

    bus: AbstractEventBus
    actor_id: str
    interval_ms: int

    async def publish(
        self,
        *,
        frame_id: str,
        frame_step: int,
        decided: bool,
        decision_latency_ms: float | None,
    ) -> None:
        if self.bus is None:
            return
        payload: dict[str, Any] = {
            "type": "StrategyHeartbeat",
            "payload_schema": "mm.strategy_heartbeat",
            "payload_version": 1,
            "actor_id": self.actor_id,
            "frame_id": frame_id,
            "frame_step": frame_step,
            "decided": bool(decided),
            "decision_latency_ms": float(decision_latency_ms) if decision_latency_ms is not None else None,
            "interval_ms": int(self.interval_ms),
            "ts_ms": int(time.time() * 1000),
        }
        await self.bus.publish(heartbeat_topic(self.actor_id), payload)


@dataclass(slots=True)
class StrategyErrorPublisher:
    """Emits one structured error event per failure."""

    bus: AbstractEventBus
    actor_id: str

    async def publish(
        self,
        *,
        kind: str,
        message: str,
        frame_id: str = "",
        frame_step: int | None = None,
        exc: BaseException | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.bus is None:
            return
        payload: dict[str, Any] = {
            "type": "StrategyError",
            "payload_schema": "mm.strategy_error",
            "payload_version": 1,
            "actor_id": self.actor_id,
            "kind": str(kind),
            "message": str(message),
            "frame_id": str(frame_id or ""),
            "frame_step": int(frame_step) if frame_step is not None else None,
            "error_class": type(exc).__name__ if exc is not None else None,
            "ts_ms": int(time.time() * 1000),
        }
        if extra:
            payload["extra"] = dict(extra)
        await self.bus.publish(errors_topic(self.actor_id), payload)


def heartbeat_age_ms(latest: dict[str, Any] | None, *, now_ms: int) -> int | None:
    """Age of a heartbeat event in ms; None when no heartbeat was ever seen."""
    if not isinstance(latest, dict):
        return None
    ts = latest.get("ts_ms")
    if ts is None:
        return None
    try:
        return max(0, int(now_ms) - int(ts))
    except (TypeError, ValueError):
        return None


def classify_data_status(
    *,
    age_ms: int | None,
    interval_ms: int,
    never_seen_as_lost: bool = True,
) -> str:
    """Map heartbeat age to a stable status string used by admin + UI.

    - ``fresh``: age < 3 * interval_ms
    - ``stale``: 3 * interval_ms <= age < 5 * interval_ms
    - ``lost``:  age >= 5 * interval_ms, or heartbeat never seen for an
      actor that the worker says it is running
    """
    if age_ms is None:
        return "lost" if never_seen_as_lost else "unknown"
    threshold_stale = 3 * max(1, int(interval_ms))
    threshold_lost = 5 * max(1, int(interval_ms))
    if age_ms >= threshold_lost:
        return "lost"
    if age_ms >= threshold_stale:
        return "stale"
    return "fresh"


def collect_recent_errors(
    events: Iterable[dict[str, Any]],
    *,
    since_ms: int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Filter an iterable of error events to the most recent N since a cutoff."""
    bucket: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        ts = int(event.get("ts_ms") or 0)
        if ts < since_ms:
            continue
        bucket.append(event)
    bucket.sort(key=lambda e: int(e.get("ts_ms") or 0), reverse=True)
    return bucket[:limit]
