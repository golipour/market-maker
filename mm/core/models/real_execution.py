"""Live/replay execution model for the ``mm`` package (float domain).

Delegates to ``ObservedStepExecutionModel`` from ``market_making_controller`` so
MPC and reward code keep identical behavior; this type exists only to give the
new stack a clear name distinct from "simulated" OB+flow execution.
"""

from __future__ import annotations

from market_making_controller.models.execution import ObservedStepExecutionModel


class RealExecutionModel(ObservedStepExecutionModel):
    """Exchange-observed inventory, cash, and refs (no synthetic fills from the book)."""
