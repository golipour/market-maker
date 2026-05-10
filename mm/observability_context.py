from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ObsContext:
    service: str
    env: str
    trace_id: str = ""
    run_id: str = ""


def get_context() -> ObsContext:
    """Shared observability context for logs, metrics, and traces."""

    service = str(os.environ.get("MM_SERVICE", "market_maker")).strip() or "market_maker"
    env = str(os.environ.get("MM_ENV", "dev")).strip() or "dev"
    trace_id = str(os.environ.get("MM_TRACE_ID", "")).strip()
    run_id = str(os.environ.get("MM_RUN_ID", "")).strip()
    return ObsContext(service=service, env=env, trace_id=trace_id, run_id=run_id)


def metric_labels(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Default low-cardinality labels for Prometheus metrics."""

    ctx = get_context()
    out: dict[str, Any] = {"service": ctx.service, "env": ctx.env}
    if ctx.run_id:
        out["run_id"] = ctx.run_id
    v = str(os.environ.get("MM_METRICS_LABEL_TRACE", "")).strip().lower()
    if v in {"1", "true", "yes", "on"} and ctx.trace_id:
        out["trace_id"] = ctx.trace_id
    if extra:
        out.update(extra)
    return out
