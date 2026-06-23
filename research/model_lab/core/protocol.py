"""Online model protocol for walk-forward evaluation."""

from __future__ import annotations

from typing import Any, Protocol

from research.model_lab.core.types import IntervalStep, StepScore


class OnlineModel(Protocol):
    """Incremental model: score the current step, then update from the transition."""

    name: str
    kind: str
    family: str  # "flow" | "price"

    def reset(self) -> None:
        """Return model to initial (cold-start) state."""

    def get_params(self) -> dict[str, Any]:
        """Current parameters (after all updates so far)."""

    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore:
        """NLL of ``current`` using parameters trained only on steps before ``current``."""

    def update(self, previous: IntervalStep, current: IntervalStep) -> bool:
        """Incorporate transition ``previous`` → ``current``. Return True if state changed."""
