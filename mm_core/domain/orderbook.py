from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CanonicalLevel:
    """One order book level in quote/base and base quantity (no tick/lot)."""

    price_quote: float
    qty_base: float


@dataclass(frozen=True, slots=True)
class CanonicalOrderbook:
    """
    Order book snapshot in float units; suitable for models and UI.

    ``mid_quote`` is quote per base, consistent with ``InstrumentSpec.tick_size`` grid.
    """

    t_ms: int
    mid_quote: float
    bids: tuple[CanonicalLevel, ...]
    asks: tuple[CanonicalLevel, ...]
