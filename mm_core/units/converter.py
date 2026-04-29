from __future__ import annotations

from dataclasses import dataclass

from .spec import InstrumentSpec


@dataclass(frozen=True, slots=True)
class VenueOrder:
    """
    Venue-ready order in the legacy tick/lot representation.
    """

    side: str  # "buy" | "sell"
    price_ticks: int
    qty_lots: int


def price_from_ticks(price_ticks: int, *, tick_size: float) -> float:
    """Convert price ticks to real quote/base price."""
    return float(price_ticks) * float(tick_size)


def qty_from_lots(qty_lots: int, *, lot_size: float) -> float:
    """Convert quantity lots to real base quantity."""
    return float(qty_lots) * float(lot_size)


def notional_from_tick_lot(notional_tick_lot: float, *, tick_size: float, lot_size: float) -> float:
    """Convert tick*lot notional units to real quote notional."""
    return float(notional_tick_lot) * float(tick_size) * float(lot_size)


def instrument_spec_from_trade_stats(exchange: str, symbol: str) -> InstrumentSpec:
    """
    Build InstrumentSpec using current trade_stats helpers.

    Centralizes these calls so live execution doesn't scatter microstructure queries.
    """
    from trade_stats import (
        get_lot_size,
        get_min_order_amount_tick_lot,
        get_min_order_qty_lots,
        get_tick_size,
    )

    ex = str(exchange or "").strip()
    sym = str(symbol or "").strip()
    tick_size = float(get_tick_size(sym, ex))
    lot_size = float(get_lot_size(sym, ex))
    min_qty_lots = max(0.0, float(get_min_order_qty_lots(sym, ex)))
    min_notional_tick_lot = max(0.0, float(get_min_order_amount_tick_lot(sym, ex)))
    return InstrumentSpec(
        exchange=ex,
        symbol=sym,
        tick_size=tick_size,
        lot_size=lot_size,
        min_qty_lots=min_qty_lots,
        min_notional_tick_lot=min_notional_tick_lot,
    )


def ladder_level_to_venue_order(
    *,
    side: str,
    mid_ticks: float,
    spread_ticks: float,
    qty_lots: int,
) -> VenueOrder | None:
    """
    Convert one ladder level (spread, qty_lots) into a venue order (price_ticks, qty_lots).
    """
    q = int(qty_lots)
    if q <= 0:
        return None
    if str(side).lower().startswith("buy"):
        px = int(round(float(mid_ticks) - float(spread_ticks)))
        side_norm = "buy"
    else:
        px = int(round(float(mid_ticks) + float(spread_ticks)))
        side_norm = "sell"
    if px <= 0:
        return None
    return VenueOrder(side=side_norm, price_ticks=px, qty_lots=q)

