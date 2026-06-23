"""Walk-forward (prequential) evaluation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from research.model_lab.core.metrics import summarize_scores
from research.model_lab.core.protocol import OnlineModel
from research.model_lab.core.types import IntervalStep, StepScore, WalkForwardSummary


@dataclass
class WalkForwardResult:
    scores: list[StepScore] = field(default_factory=list)
    summary: WalkForwardSummary | None = None


def run_walk_forward(
    model: OnlineModel,
    steps: Iterable[IntervalStep],
    *,
    min_updates_before_score: int = 1,
) -> WalkForwardResult:
    """Evaluate one model on a time-ordered interval sequence.

    Contract (matches production ``score_frame_models`` + update):

    1. For transition ``(previous, current)`` at index ``i >= 1``:
       - If ``i - 1 >= min_updates_before_score``: score ``current`` using params
         trained on steps ``0 .. i-1`` only.
       - ``update(previous, current)`` so params include step ``i``.

    Step 0 is only used as context for the first scored transition.
    """

    seq = list(steps)
    model.reset()
    scores: list[StepScore] = []
    for index in range(1, len(seq)):
        previous = seq[index - 1]
        current = seq[index]
        if index - 1 >= min_updates_before_score:
            score = model.score(previous, current)
            scores.append(
                StepScore(
                    step_index=index,
                    ts=current.ts,
                    model_name=score.model_name,
                    model_kind=score.model_kind,
                    family=score.family,
                    log_likelihood=score.log_likelihood,
                    nll=score.nll,
                    skip_reason=score.skip_reason,
                    standardized_residual=score.standardized_residual,
                    observation=dict(score.observation),
                    params_before=dict(score.params_before),
                )
            )
        model.update(previous, current)

    summary = summarize_scores(
        scores,
        model_name=model.name,
        model_kind=model.kind,
        family=model.family,
        n_steps=len(seq),
    )
    return WalkForwardResult(scores=scores, summary=summary)
