from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ObsContext:
    service: str
    env: str
    # Correlation (Phase 7): inject from platform (K8s downward API, sidecar) or leave empty.
    trace_id: str = ""
    run_id: str = ""


def get_context() -> ObsContext:
    """
    Shared observability context.

    - service: stable service name
    - env: environment (dev/prod/etc)
    - trace_id: optional W3C / platform trace id (``MM_TRACE_ID``)
    - run_id: optional process or deployment id (``MM_RUN_ID``)
    """
    service = str(os.environ.get("MM_SERVICE", "market_maker")).strip() or "market_maker"
    env = str(os.environ.get("MM_ENV", "dev")).strip() or "dev"
    trace_id = str(os.environ.get("MM_TRACE_ID", "")).strip()
    run_id = str(os.environ.get("MM_RUN_ID", "")).strip()
    return ObsContext(service=service, env=env, trace_id=trace_id, run_id=run_id)


def metric_labels(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Default Prometheus label set: service, env, and optional low-cardinality ids.

    ``trace_id`` is omitted by default (high cardinality). Set ``MM_METRICS_LABEL_TRACE=1`` to
    add it for short-lived debugging only.
    """
    ctx = get_context()
    out: dict[str, Any] = {
        "service": ctx.service,
        "env": ctx.env,
    }
    if ctx.run_id:
        out["run_id"] = ctx.run_id
    v = str(os.environ.get("MM_METRICS_LABEL_TRACE", "")).strip().lower()
    if v in {"1", "true", "yes", "on"} and ctx.trace_id:
        out["trace_id"] = ctx.trace_id
    if extra:
        out.update(extra)
    return out

