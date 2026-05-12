from __future__ import annotations

from typing import Any

from mm.core.models.strategy import StrategyInputFrame


def market_flow_step_from_frame(frame: StrategyInputFrame) -> tuple[Any, ...]:
    """Temporary adapter from canonical strategy frames to legacy MPC step shape."""

    return (
        frame.primary.t_end_ms,
        None,
        frame.primary.buy_volume,
        frame.primary.sell_volume,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        None,
        None,
        None,
        None,
        0.0,
        0.0,
    )
