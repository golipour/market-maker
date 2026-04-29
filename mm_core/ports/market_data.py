from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StepperPort(Protocol):
    """
    Port for the stepper-like component that produces windowed steps.

    This is intentionally minimal; we will refine types once we migrate away from
    tuple-based Step.
    """

    T_ms: int
    steps: list[tuple]

    def advance(self, t_end_ms: int | None = None) -> tuple | None:
        ...

