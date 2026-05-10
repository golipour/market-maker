from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
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
    payload = {"ts_ms": int(time.time() * 1000), "event": event}
    if fields:
        payload.update(fields)
    _append_log(payload)
    try:
        from mm_adapters.observability.jsonlog import emit

        emit(event, fields)
    except Exception:
        return


def _append_log(payload: dict[str, Any]) -> None:
    path = os.environ.get("MM_LOG_PATH", "").strip()
    if not path:
        return
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
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
