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
        local_maker_fee: float | None = None,
        global_maker_fee: float | None = None,
    ) -> None:
        self.maker_fee = self._clamp_fee(maker_fee)
        self.local_maker_fee = self._clamp_fee(
            self.maker_fee if local_maker_fee is None else local_maker_fee
        )
        self.global_maker_fee = self._clamp_fee(
            self.maker_fee if global_maker_fee is None else global_maker_fee
        )

    @staticmethod
    def _clamp_fee(value: float) -> float:
        return max(0.0, min(1.0 - 1e-9, float(value)))

    def __repr__(self) -> str:
        return (
            "RealExecutionModel("
            f"maker_fee={self.maker_fee!r}, "
            f"local_maker_fee={self.local_maker_fee!r}, "
            f"global_maker_fee={self.global_maker_fee!r})"
        )
