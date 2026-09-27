"""Stage-level timing instrumentation for the MPC optimizer.

The optimizer historically reported only ``sample_draw_sec``, ``train_sec``
and ``total_sec``. Those are seconds, while the persistence layer looks for
``*_ms`` keys, so ``mm_optimizer_facts.optimize_ms`` was always null and every
latency question had to be answered by scanning ``mm_events.payload`` JSON.

This module provides a small accumulator that records wall-clock time per
stage in **milliseconds** and renders a flat mapping that the existing
``_timing_float`` fact extractors can consume directly.

Stages
------
``setup``        book/grid/state construction and warm-start resolution
``sample``       Monte Carlo flow and price-innovation draws
``forward``      differentiable rollout up to the terminal objective
``objective``    terminal objective, validity and viability assembly
``backward``     autograd backward pass
``adam``         Adam update and bound projection (closure cost excluded)
``diagnostics``  scalar extraction and per-epoch metrics construction

``objective`` is nested inside ``forward``: the rollout and objective assembly
are interleaved inside the closure, so ``objective`` is reported as an
informational breakdown and is excluded from the ``optimize_ms`` total.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = [
    "STAGE_KEYS",
    "NESTED_STAGES",
    "OptimizerStageTimer",
    "stage_timing_payload",
]


#: Canonical stage order used for stable diagnostics payloads.
STAGE_KEYS: tuple[str, ...] = (
    "setup",
    "sample",
    "forward",
    "objective",
    "backward",
    "adam",
    "diagnostics",
)

#: Stages that are nested inside another stage and therefore not summed into
#: ``optimize_ms``; they are informational breakdowns only.
NESTED_STAGES: frozenset[str] = frozenset({"objective"})


@dataclass(slots=True)
class OptimizerStageTimer:
    """Accumulate per-stage wall-clock milliseconds for one optimizer run.

    The timer is deliberately allocation-light: ``perf_counter`` is called
    twice per measured region and a single float is accumulated. One instance
    is created per ``optimize_with_diagnostics`` call, so using it from the
    Adam epoch loop adds no measurable overhead.
    """

    totals_ms: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    epochs_run: int = 0

    def record(self, stage: str, elapsed_ms: float) -> None:
        """Add ``elapsed_ms`` to ``stage``'s running total."""
        name = str(stage).strip()
        if not name:
            return
        value = float(elapsed_ms)
        if not value > 0.0:
            # A non-positive interval can only come from a mis-measured
            # region; keep totals monotonic so downstream percentiles stay
            # meaningful.
            return
        self.totals_ms[name] = self.totals_ms.get(name, 0.0) + value
        self.counts[name] = self.counts.get(name, 0) + 1

    def bump_epochs(self, count: int = 1) -> None:
        self.epochs_run = max(0, self.epochs_run + int(count))

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time a region and attribute it to ``stage``."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - started) * 1000.0)

    def total_ms(self) -> float:
        """Sum of non-nested stages."""
        return sum(
            value
            for stage, value in self.totals_ms.items()
            if stage not in NESTED_STAGES
        )

    def as_payload(self) -> dict[str, Any]:
        """Render a flat mapping for ``OptimizerRunDiagnostics.timing``.

        Keys are ``<stage>_ms`` so the persistence extractors pick them up
        with no extra mapping. ``optimize_ms`` carries the top-level total,
        which is what ``mm_optimizer_facts.optimize_ms`` reads.
        """
        payload: dict[str, Any] = {}
        for stage in STAGE_KEYS:
            value = self.totals_ms.get(stage)
            if value is not None:
                payload[f"{stage}_ms"] = float(value)
        for stage, value in self.totals_ms.items():
            if stage not in STAGE_KEYS:
                payload[f"{stage}_ms"] = float(value)
        payload["optimize_ms"] = float(self.total_ms())
        payload["epochs_run"] = int(self.epochs_run)
        return payload


def stage_timing_payload(
    timer: OptimizerStageTimer,
    *,
    legacy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge ``timer`` output with legacy second-based timing keys.

    The legacy ``*_sec`` keys are preserved because existing dashboards and
    alert rules read them directly; new code should read the ``*_ms`` keys.
    """
    payload: dict[str, Any] = dict(legacy or {})
    payload.update(timer.as_payload())
    return payload
