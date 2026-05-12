from __future__ import annotations

from typing import Any


def gauge(name: str, value: float, *, labels: dict[str, Any] | None = None, help: str = "") -> None:
    try:
        from mm.observability.registry import REGISTRY

        REGISTRY.gauge(name, value, labels=labels, help=help)
    except Exception:
        return


def counter_inc(name: str, inc: float = 1.0, *, labels: dict[str, Any] | None = None, help: str = "") -> None:
    try:
        from mm.observability.registry import REGISTRY

        REGISTRY.counter_inc(name, inc, labels=labels, help=help)
    except Exception:
        return


def histogram_observe(
    name: str,
    value: float,
    *,
    labels: dict[str, Any] | None = None,
    help: str = "",
    buckets: tuple[float, ...] | None = None,
) -> None:
    try:
        from mm.observability.registry import REGISTRY

        REGISTRY.histogram_observe(name, value, labels=labels, help=help, buckets=buckets)
    except Exception:
        return
