"""Aggregate walk-forward metrics."""

from __future__ import annotations

import math
from collections import Counter

from research.model_lab.core.types import StepScore, WalkForwardSummary


def summarize_scores(
    scores: list[StepScore],
    *,
    model_name: str,
    model_kind: str,
    family: str,
    n_steps: int,
) -> WalkForwardSummary:
    valid_nll = [float(s.nll) for s in scores if s.nll is not None and math.isfinite(s.nll)]
    valid_ll = [
        float(s.log_likelihood)
        for s in scores
        if s.log_likelihood is not None and math.isfinite(s.log_likelihood)
    ]
    skip_reasons = Counter(s.skip_reason for s in scores if s.skip_reason)

    mean_nll = sum(valid_nll) / len(valid_nll) if valid_nll else None
    median_nll = sorted(valid_nll)[len(valid_nll) // 2] if valid_nll else None
    total_nll = sum(valid_nll) if valid_nll else None
    mean_ll = sum(valid_ll) / len(valid_ll) if valid_ll else None

    return WalkForwardSummary(
        model_name=model_name,
        model_kind=model_kind,
        family=family,
        n_steps=n_steps,
        n_scored=len(valid_nll),
        n_skipped=len(scores) - len(valid_nll),
        mean_nll=mean_nll,
        median_nll=median_nll,
        total_nll=total_nll,
        mean_log_likelihood=mean_ll,
        skip_reasons=dict(skip_reasons),
    )
