"""
Rounding helpers for venue grids (Phase 2 — use at adapter / execution edges).

Core numeric policies should consume already-rounded floats from here where tick/lot
rules apply.
"""

from __future__ import annotations

import math


def round_price_to_tick(price_quote: float, *, tick_size: float, mode: str = "nearest") -> float:
    """Map a quote/base price to the exchange tick grid."""
    if tick_size <= 0.0:
        return float(price_quote)
    u = float(price_quote) / float(tick_size)
    if mode == "down":
        return math.floor(u) * tick_size
    if mode == "up":
        return math.ceil(u) * tick_size
    return round(u) * tick_size


def round_qty_to_lot(qty_base: float, *, lot_size: float, mode: str = "down") -> float:
    """Map a base quantity to a multiple of lot size (conservative default: down)."""
    if lot_size <= 0.0:
        return float(qty_base)
    u = float(qty_base) / float(lot_size)
    if mode == "up":
        return math.ceil(u) * lot_size
    if mode == "nearest":
        return round(u) * lot_size
    return math.floor(u) * lot_size
