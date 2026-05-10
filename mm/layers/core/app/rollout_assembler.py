"""
Phase 3 — bridge from runner runtime state to MPC / tensor batches.

The optimizer (``mm.mpc.mpc_tensor_batched``) reads duck-typed
``runner`` and tuple steps. This module defines a small **read-only feature bundle**
we can grow into a real tensor assembler without changing trading behavior yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RolloutFeatures:
    """Minimal scalar bundle observable from ``OrderFlowRunner`` for rollout packing."""

    step_counter: int
    inventory_lots: int
    cash_tick_lot: float
    spread_max: float | None


class RolloutAssembler:
    """Collect features for future batched rollout construction (MPC tensors)."""

    @staticmethod
    def features_from_runner(runner: Any) -> RolloutFeatures:
        return RolloutFeatures(
            step_counter=int(getattr(runner, "_step_counter", 0)),
            inventory_lots=int(getattr(runner, "inventory", 0)),
            cash_tick_lot=float(getattr(runner, "cash", 0.0)),
            spread_max=getattr(runner, "spread_max", None),
        )
