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
        **_ignored: Any,
    ) -> None:
        # Same clamp semantics as legacy observed-step model; MPC reads these attributes.
        self.maker_fee = max(0.0, min(1.0 - 1e-9, float(maker_fee)))
        if hedge_fee is not None:
            self.global__fee = max(0.0, min(1.0 - 1e-9, float(hedge_fee)))

    def __repr__(self) -> str:
        parts = [f"maker_fee={self.maker_fee!r}"]
        if hasattr(self, "hedge_fee"):
            parts.append(f"hedge_fee={self.global__fee!r}")
        return f"RealExecutionModel({', '.join(parts)})"
