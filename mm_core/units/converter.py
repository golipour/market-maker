from __future__ import annotations

from dataclasses import dataclass

from .spec import InstrumentSpec


@dataclass(frozen=True, slots=True)
class VenueOrder:
    """Venue-ready limit order (integer ticks × lots at the exchange edge only)."""

    side: str  # "buy" | "sell"
    price_ticks: int
    qty_lots: int


def price_from_ticks(price_ticks: int | float, *, tick_size: float) -> float:
    """Convert price ticks (int or float) to real quote/base price."""
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
    from mm.integrations.trade_stats import (
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


@dataclass(frozen=True, slots=True)
class UnitConverter:
    """
    Single entry point for tick/lot ↔ float conversions for one instrument.

    Use **inside adapters** and **at execution boundaries**; keep trading logic in floats
    where possible by converting early from tuple/step payloads.
    """

    spec: InstrumentSpec

    def price_from_ticks(self, price_ticks: int | float) -> float:
        return price_from_ticks(price_ticks, tick_size=self.spec.tick_size)

    def qty_from_lots(self, qty_lots: int | float) -> float:
        return qty_from_lots(int(qty_lots), lot_size=self.spec.lot_size)

    def notional_from_tick_lot(self, notional_tick_lot: float) -> float:
        return notional_from_tick_lot(
            float(notional_tick_lot),
            tick_size=self.spec.tick_size,
            lot_size=self.spec.lot_size,
        )

    def price_ticks_from_quote(self, price_quote: float) -> int:
        from mm_core.units.rounding import round_price_to_tick

        ts = float(self.spec.tick_size)
        if ts <= 0:
            return int(round(price_quote))
        pq = round_price_to_tick(float(price_quote), tick_size=ts)
        return int(round(pq / ts))

    def qty_lots_from_base(self, qty_base: float) -> int:
        from mm_core.units.rounding import round_qty_to_lot

        ls = float(self.spec.lot_size)
        if ls <= 0:
            return max(0, int(qty_base))
        qb = round_qty_to_lot(float(qty_base), lot_size=ls, mode="down")
        return max(0, int(round(qb / ls)))

    def ladder_to_venue_order(
        self,
        *,
        side: str,
        mid_ticks: float,
        spread_ticks: float,
        qty_lots: int,
    ) -> VenueOrder | None:
        return ladder_level_to_venue_order(
            side=side,
            mid_ticks=mid_ticks,
            spread_ticks=spread_ticks,
            qty_lots=qty_lots,
        )


def unit_converter_for(exchange: str, symbol: str) -> UnitConverter:
    """Build a converter from live ``trade_stats`` microstructure (edge lookup)."""
    return UnitConverter(instrument_spec_from_trade_stats(exchange, symbol))

