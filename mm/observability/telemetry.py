"""Aggregate re-export of observability primitives (used by legacy imports)."""

from mm.observability.logs import log_event
from mm.observability.metrics import counter_inc, gauge, histogram_observe
from mm.observability.traces import latency_metric, span

__all__ = [
    "counter_inc",
    "gauge",
    "histogram_observe",
    "latency_metric",
    "log_event",
    "span",
]
