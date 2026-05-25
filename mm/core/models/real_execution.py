"""Live/replay execution configuration for the ``mm`` package.

``RealExecutionModel`` holds fee rates consumed by MPC rollouts. Portfolio state
in the ``mm`` runtime comes from private adapters and market frames, not from
this class.
"""

from __future__ import annotations


class RealExecutionModel:
    """Fee parameters for MPC; independent of ``market_making_controller``."""

    def __init__(
        self,
        maker_fee: float = 0.0,
    ) -> None:
        self.maker_fee = max(0.0, min(1.0 - 1e-9, float(maker_fee)))

    def __repr__(self) -> str:
        return f"RealExecutionModel(maker_fee={self.maker_fee!r})"
