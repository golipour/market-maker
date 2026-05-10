from __future__ import annotations

import contextlib
import time
from typing import Any


def gauge(name: str, value: float, *, labels: dict[str, Any] | None = None, help: str = "") -> None:
    try:
        from mm_adapters.observability.metrics import REGISTRY

        REGISTRY.gauge(name, value, labels=labels, help=help)
    except Exception:
        return


def counter_inc(name: str, inc: float = 1.0, *, labels: dict[str, Any] | None = None, help: str = "") -> None:
    try:
        from mm_adapters.observability.metrics import REGISTRY

        REGISTRY.counter_inc(name, inc, labels=labels, help=help)
    except Exception:
        return


def log_event(event: str, fields: dict[str, Any] | None = None) -> None:
    try:
        from mm_adapters.observability.jsonlog import emit

        emit(event, fields)
    except Exception:
        return


@contextlib.contextmanager
def span(name: str, **attributes: Any):
    """Optional OpenTelemetry span with a no-dependency fallback."""

    tracer = _tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            with contextlib.suppress(Exception):
                current.set_attribute(key, value)
        yield current


@contextlib.contextmanager
def latency_metric(name: str, *, labels: dict[str, Any] | None = None, help: str = ""):
    started = time.monotonic_ns()
    try:
        yield
    finally:
        gauge(name, (time.monotonic_ns() - started) / 1_000_000, labels=labels, help=help)


def _tracer():
    try:
        from opentelemetry import trace

        return trace.get_tracer("market-maker")
    except Exception:
        return None
