"""Live/replay execution configuration for the ``mm`` package (no legacy imports).

``RealExecutionModel`` holds fee rates and related knobs consumed by MPC rollouts
(e.g. ``MonteCarloMPCTensorBatched`` reads ``maker_fee`` and optional ``hedge_fee``
from the execution object). Portfolio state in the ``mm`` runtime comes from
private adapters and market frames, not from this class.
"""

from __future__ import annotations

from typing import Any


class RealExecutionModel:
    """Fee parameters for MPC; independent of ``market_making_controller``."""

    def __init__(
        self,
        maker_fee: float = 0.0,
        hedge_fee: float | None = None,
        global_fee: float | None = None,
        **_ignored: Any,
    ) -> None:
        self.maker_fee = max(0.0, min(1.0 - 1e-9, float(maker_fee)))
        fee = global_fee if global_fee is not None else hedge_fee
        if fee is not None:
            self.global_fee = max(0.0, min(1.0 - 1e-9, float(fee)))

    def __repr__(self) -> str:
        parts = [f"maker_fee={self.maker_fee!r}"]
        if hasattr(self, "global_fee"):
            parts.append(f"global_fee={self.global_fee!r}")
        return f"RealExecutionModel({', '.join(parts)})"
