"""
Single place for market-order-flow model + tester updates (step boundary).

Also passes **parsed** step views (``ParsedStep``) for models that want canonical fields
without re-parsing.
"""

from __future__ import annotations

from typing import Any

from mm.layers.core.app.parsed_step import ParsedStep, parse_step


def update_flow_models_after_step(
    runner: Any,
    prev_step: Any | None,
    curr_step: Any,
    params: dict,
) -> None:
    """Apply one step of MOF + flow tester learning from consecutive steps."""
    parsed_prev: ParsedStep | None = parse_step(prev_step) if prev_step is not None else None
    parsed_curr = parse_step(curr_step)
    runner.tester.update(
        prev_step, curr_step, params, parsed_prev=parsed_prev, parsed_curr=parsed_curr
    )
    runner.mof_model.update(
        prev_step, curr_step, parsed_prev=parsed_prev, parsed_curr=parsed_curr
    )
