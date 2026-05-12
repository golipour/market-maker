from __future__ import annotations

import contextlib
import time
from typing import Any

from mm.observability.metrics import gauge


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
