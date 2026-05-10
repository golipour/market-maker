"""VWAP reference pricing for batched MPC rollouts (``mm`` naming layer).

The numeric implementation lives in ``mm_core.pricing.simulated_rep_prices``;
this module exposes VWAP-oriented names for the new stack.
"""

from __future__ import annotations

from mm_core.pricing.simulated_rep_prices import representative_prices_batched as _core_representative_prices_batched


def vwap_reference_prices_batched(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    center_mid: torch.Tensor,
    spread_grid: torch.Tensor,
    *,
    vwap_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VWAP buy/sell reference prices up to ``vwap_depth`` cumulative base quantity.

    ``vwap_depth`` matches the semantics of ``target_depth`` in the core helper:
    cap on cumulative volume (per side) used for the reference VWAP.
    """
    return _core_representative_prices_batched(
        ob_bids,
        ob_asks,
        center_mid,
        spread_grid,
        target_depth=float(vwap_depth),
    )
