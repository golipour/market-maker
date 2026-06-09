from __future__ import annotations

from typing import Any

from mm.observability.metrics import counter_inc, histogram_observe
from mm.observability.schema import (
    MM_SCHEDULER_CALLBACK_LATENCY_MS,
    MM_SCHEDULER_CALLBACK_MISSED_DEADLINE_TOTAL,
)


def record_callback(
    *,
    interval_ms: int,
    callback: str,
    latency_ms: float,
    missed_deadline: bool,
) -> None:
    labels: dict[str, Any] = {
        "interval_ms": str(interval_ms),
        "callback": callback,
    }
    histogram_observe(
        MM_SCHEDULER_CALLBACK_LATENCY_MS,
        max(0.0, float(latency_ms)),
        labels=labels,
        help="Interval scheduler callback wall time in milliseconds",
    )
    if missed_deadline:
        counter_inc(
            MM_SCHEDULER_CALLBACK_MISSED_DEADLINE_TOTAL,
            labels=labels,
            help="Scheduler callbacks that exceeded their interval budget",
        )
