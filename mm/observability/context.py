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
    worker_id: str = ""
    strategy_id: str = ""


def get_context() -> ObsContext:
    """Shared observability context for logs, metrics, and traces."""

    service = str(os.environ.get("MM_SERVICE", "market_maker")).strip() or "market_maker"
    env = str(os.environ.get("MM_ENV", "dev")).strip() or "dev"
    trace_id = str(os.environ.get("MM_TRACE_ID", "")).strip()
    run_id = str(os.environ.get("MM_RUN_ID", "")).strip()
    worker_id = str(os.environ.get("MM_WORKER_ID", "")).strip()
    strategy_id = str(os.environ.get("MM_STRATEGY_ID", "")).strip()
    return ObsContext(
        service=service,
        env=env,
        trace_id=trace_id,
        run_id=run_id,
        worker_id=worker_id,
        strategy_id=strategy_id,
    )


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def metric_labels(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Default low-cardinality labels for Prometheus metrics.

    ``run_id`` only when ``MM_METRICS_LABEL_RUN_ID`` is truthy and ``MM_RUN_ID`` is set.
    ``worker_id`` and ``strategy_id`` follow the same opt-in pattern via
    ``MM_METRICS_LABEL_WORKER_ID`` and ``MM_METRICS_LABEL_STRATEGY_ID``.
    Never put ``trace_id`` on metric labels; use logs/spans/envelopes.
    """

    ctx = get_context()
    out: dict[str, Any] = {"service": ctx.service, "env": ctx.env}
    if _truthy_env("MM_METRICS_LABEL_RUN_ID") and ctx.run_id:
        out["run_id"] = ctx.run_id
    if _truthy_env("MM_METRICS_LABEL_WORKER_ID") and ctx.worker_id:
        out["worker_id"] = ctx.worker_id
    if _truthy_env("MM_METRICS_LABEL_STRATEGY_ID") and ctx.strategy_id:
        out["strategy_id"] = ctx.strategy_id
    if extra:
        out.update(extra)
    return out
