from __future__ import annotations

from typing import Any

from mm.layers.core.ports.observability import (
    DiagnosticsEmit,
    MetricLabelsFn,
    MetricsRegistryPort,
    StructuredLogEmit,
)


def noop_structured_log(event: str, fields: dict[str, Any] | None = None) -> None:
    return


def noop_metric_labels(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(extra or {})


class _NoopMetricsRegistry(MetricsRegistryPort):
    def gauge(self, name: str, value: float, *, labels: dict[str, Any] | None = None, help: str = "") -> None:
        return

    def counter_inc(self, name: str, inc: float = 1.0, *, labels: dict[str, Any] | None = None, help: str = "") -> None:
        return


NOOP_METRICS_REGISTRY: MetricsRegistryPort = _NoopMetricsRegistry()


def noop_diagnostics_emit(frame: dict[str, Any]) -> None:
    return


def resolve_structured_log(runner: object) -> StructuredLogEmit:
    fn = getattr(runner, "structured_log_emit", None)
    if callable(fn):
        return fn  # type: ignore[return-value]
    return noop_structured_log


def resolve_metric_labels(runner: object) -> MetricLabelsFn:
    fn = getattr(runner, "metric_labels", None)
    if callable(fn):
        return fn  # type: ignore[return-value]
    return noop_metric_labels


def resolve_metrics_registry(runner: object) -> MetricsRegistryPort:
    reg = getattr(runner, "metrics_registry", None)
    if reg is not None and hasattr(reg, "gauge") and hasattr(reg, "counter_inc"):
        return reg  # type: ignore[return-value]
    return NOOP_METRICS_REGISTRY


def resolve_diagnostics_emit(runner: object) -> DiagnosticsEmit:
    fn = getattr(runner, "diagnostics_emit", None)
    if callable(fn):
        return fn  # type: ignore[return-value]
    return noop_diagnostics_emit
