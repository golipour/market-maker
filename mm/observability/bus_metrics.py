from __future__ import annotations

from typing import Any

from mm.observability.metrics import counter_inc, gauge, histogram_observe
from mm.observability.schema import (
    MM_BUS_EVENTS_DROPPED_TOTAL,
    MM_BUS_PUBLISH_BLOCKED_MS,
    MM_BUS_SUBSCRIBER_LAG_MS,
    MM_BUS_SUBSCRIBER_QUEUE_DEPTH,
)


def record_subscriber_queue(
    topic: str,
    subscriber_id: str,
    *,
    depth: int,
    lag_ms: int,
) -> None:
    labels = {"topic": topic, "subscriber_id": subscriber_id}
    gauge(
        MM_BUS_SUBSCRIBER_QUEUE_DEPTH,
        float(max(0, depth)),
        labels=labels,
        help="Current subscriber queue depth by topic",
    )
    gauge(
        MM_BUS_SUBSCRIBER_LAG_MS,
        float(max(0, lag_ms)),
        labels=labels,
        help="Age in ms of the oldest queued event for a subscriber",
    )


def record_event_dropped(topic: str, subscriber_id: str, *, policy: str) -> None:
    counter_inc(
        MM_BUS_EVENTS_DROPPED_TOTAL,
        labels={"topic": topic, "subscriber_id": subscriber_id, "policy": policy},
        help="Bus events dropped due to subscriber backpressure",
    )


def record_publish_blocked_ms(topic: str, subscriber_id: str, wait_ms: float) -> None:
    histogram_observe(
        MM_BUS_PUBLISH_BLOCKED_MS,
        max(0.0, float(wait_ms)),
        labels={"topic": topic, "subscriber_id": subscriber_id},
        help="Time spent blocked while publishing to a full subscriber queue",
    )
