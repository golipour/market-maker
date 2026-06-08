from __future__ import annotations

from typing import Any

from mm.core.events import EventEnvelope, envelope_event, envelope_size_bytes, json_size_bytes
from mm.observability.metrics import counter_inc, histogram_observe
from mm.observability import schema as obs_schema

EVENT_BYTE_BUCKETS = (
    128.0,
    256.0,
    512.0,
    1024.0,
    2048.0,
    4096.0,
    8192.0,
    16384.0,
    32768.0,
    65536.0,
    131072.0,
    262144.0,
    524288.0,
    1048576.0,
    float("inf"),
)


def observe_event_payload(
    *,
    topic: str,
    event_type: str,
    stage: str,
    payload_bytes: int,
    envelope_bytes: int | None = None,
) -> None:
    labels: dict[str, Any] = {
        "topic": topic,
        "event_type": event_type or "event",
        "stage": stage,
    }
    counter_inc(
        obs_schema.MM_EVENT_PAYLOADS_TOTAL,
        labels=labels,
        help="Observed event payload count by topic, event type, and pipeline stage",
    )
    counter_inc(
        obs_schema.MM_EVENT_PAYLOAD_BYTES_TOTAL,
        float(max(0, int(payload_bytes))),
        labels=labels,
        help="Observed event payload bytes by topic, event type, and pipeline stage",
    )
    histogram_observe(
        obs_schema.MM_EVENT_PAYLOAD_BYTES,
        float(max(0, int(payload_bytes))),
        labels=labels,
        help="Observed event payload byte distribution",
        buckets=EVENT_BYTE_BUCKETS,
    )
    if envelope_bytes is None:
        return
    counter_inc(
        obs_schema.MM_EVENT_ENVELOPE_BYTES_TOTAL,
        float(max(0, int(envelope_bytes))),
        labels=labels,
        help="Observed event envelope bytes by topic, event type, and pipeline stage",
    )
    histogram_observe(
        obs_schema.MM_EVENT_ENVELOPE_BYTES,
        float(max(0, int(envelope_bytes))),
        labels=labels,
        help="Observed event envelope byte distribution",
        buckets=EVENT_BYTE_BUCKETS,
    )


def observe_runtime_event(topic: str, event: Any, *, stage: str) -> None:
    try:
        envelope = envelope_event(topic, event)
        observe_event_payload(
            topic=envelope.topic,
            event_type=envelope.event_type,
            stage=stage,
            payload_bytes=json_size_bytes(envelope.payload),
        )
    except Exception:
        return


def observe_envelope(envelope: EventEnvelope, *, stage: str) -> None:
    try:
        observe_event_payload(
            topic=envelope.topic,
            event_type=envelope.event_type,
            stage=stage,
            payload_bytes=json_size_bytes(envelope.payload),
            envelope_bytes=envelope_size_bytes(envelope),
        )
    except Exception:
        return
