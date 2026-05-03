"""
Single place for market-order-flow model + tester updates (tuple step boundary).

When MOF accepts canonical inputs, swap internals here without touching the runner loop.
"""

from __future__ import annotations

from typing import Any


def update_flow_models_after_step(
    runner: Any,
    prev_step: tuple | None,
    curr_step: tuple,
    params: dict,
) -> None:
    """Apply one step of MOF + flow tester learning from consecutive tuple steps."""
    runner.tester.update(prev_step, curr_step, params)
    runner.mof_model.update(prev_step, curr_step)
