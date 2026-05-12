"""Target depth (lots) for VWAP pricing."""

from __future__ import annotations

from mm.pricing.constants import DEFAULT_VWAP_DEPTH_LOTS


def target_depth_lots_from_price_model(price_model: object | None) -> float:
    """Read ``_target_depth_lots`` from a torch price model; fall back to the global default."""

    if price_model is None:
        return float(DEFAULT_VWAP_DEPTH_LOTS)
    return float(getattr(price_model, "_target_depth_lots", DEFAULT_VWAP_DEPTH_LOTS))
