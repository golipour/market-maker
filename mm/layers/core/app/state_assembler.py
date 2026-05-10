"""
Phase 3 — assemble observed market/step data into domain contracts.

``StateAssembler`` is the single entry for ParsedStep → ``StepSnapshot`` and for the
legacy tuple → controller ``State``. Eventually tuple plumbing disappears upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mm.layers.core.app.assemblers import step_snapshot_from_step
from mm.layers.core.app.parsed_step import ParsedStep
from mm.layers.core.domain.snapshots import StepSnapshot
from mm.layers.core.domain.types import InstrumentId
from mm.layers.core.units.spec import InstrumentSpec

if TYPE_CHECKING:
    from mm.core.orderflow_state import State


class StateAssembler:
    """Observed step → ``StepSnapshot``; tuple → ``State`` for MPC/controller."""

    @staticmethod
    def step_snapshot(
        parsed: ParsedStep,
        *,
        instrument: InstrumentId,
        spec: InstrumentSpec,
    ) -> StepSnapshot:
        return step_snapshot_from_step(parsed, instrument=instrument, spec=spec)

    @staticmethod
    def controller_state(step: tuple, *, spread_max: float | None = None) -> State:
        from mm.core.orderflow_state import state_from_step

        return state_from_step(step, spread_max=spread_max)
