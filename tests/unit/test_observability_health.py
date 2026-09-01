from __future__ import annotations

import asyncio
import time

import pytest

from mm.observability.heartbeat import (
    ERRORS_TOPIC_PREFIX,
    HEARTBEAT_TOPIC_PREFIX,
    HeartbeatPublisher,
    StrategyErrorPublisher,
    classify_data_status,
    collect_recent_errors,
    heartbeat_age_ms,
)


def test_classify_data_status_thresholds() -> None:
    assert classify_data_status(age_ms=5_000, interval_ms=30_000) == "fresh"
    assert classify_data_status(age_ms=120_000, interval_ms=30_000) == "stale"
    assert classify_data_status(age_ms=300_000, interval_ms=30_000) == "lost"
    assert classify_data_status(age_ms=None, interval_ms=30_000) == "lost"
    assert classify_data_status(age_ms=None, interval_ms=30_000, never_seen_as_lost=False) == "unknown"


def test_heartbeat_age_ms_handles_missing() -> None:
    assert heartbeat_age_ms(None, now_ms=1_000) is None
    assert heartbeat_age_ms({}, now_ms=1_000) is None
    assert heartbeat_age_ms({"ts_ms": 900}, now_ms=1_000) == 100


def test_collect_recent_errors_sorts_and_caps() -> None:
    events = [
        {"ts_ms": 100, "kind": "warn", "message": "old"},
        {"ts_ms": 200, "kind": "warn", "message": "newer"},
        {"ts_ms": 50, "kind": "warn", "message": "before_window"},
    ]
    out = collect_recent_errors(events, since_ms=75, limit=2)
    assert [e["message"] for e in out] == ["newer", "old"]


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.events.append((topic, payload))


@pytest.mark.asyncio
async def test_heartbeat_publisher_emits_to_dedicated_topic() -> None:
    bus = _RecordingBus()
    pub = HeartbeatPublisher(bus=bus, actor_id="bitwana-controller", interval_ms=30_000)
    await pub.publish(
        frame_id="frame:1",
        frame_step=7,
        decided=True,
        decision_latency_ms=120.0,
    )
    topic, payload = bus.events[0]
    assert topic == f"{HEARTBEAT_TOPIC_PREFIX}bitwana-controller"
    assert payload["actor_id"] == "bitwana-controller"
    assert payload["decided"] is True
    assert payload["interval_ms"] == 30_000
    assert payload["frame_id"] == "frame:1"
    assert payload["frame_step"] == 7
    assert payload["decision_latency_ms"] == 120.0
    assert payload["ts_ms"] <= int(time.time() * 1000)


@pytest.mark.asyncio
async def test_error_publisher_emits_to_dedicated_topic() -> None:
    bus = _RecordingBus()
    pub = StrategyErrorPublisher(bus=bus, actor_id="mpc-controller")
    await pub.publish(
        kind="controller_error",
        message="ValueError: bad input",
        frame_id="frame:1",
        frame_step=4,
        exc=ValueError("bad input"),
    )
    topic, payload = bus.events[0]
    assert topic == f"{ERRORS_TOPIC_PREFIX}mpc-controller"
    assert payload["kind"] == "controller_error"
    assert payload["error_class"] == "ValueError"
    assert payload["frame_step"] == 4
