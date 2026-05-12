"""Observability: context, Prometheus registry, logs, traces, bus subscribers, runtime setup."""

from mm.observability.context import ObsContext, get_context, metric_labels
from mm.observability.logs import log_event
from mm.observability.metrics import counter_inc, gauge, histogram_observe
from mm.observability.registry import REGISTRY, MetricsRegistry, start_server
from mm.observability.setup import configure_observability
from mm.observability.traces import latency_metric, span

__all__ = [
    "REGISTRY",
    "MetricsRegistry",
    "ObsContext",
    "configure_observability",
    "counter_inc",
    "gauge",
    "get_context",
    "histogram_observe",
    "latency_metric",
    "log_event",
    "metric_labels",
    "span",
    "start_server",
]
