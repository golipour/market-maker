"""
Portfolio wealth from cash + inventory + hedge leg — **one formula** for ticks and quote/base.

Tick-space uses the same convention as ``State`` (cash in tick×lot notional, refs in ticks).
Quote-space uses floats from ``InstrumentSnapshot`` / assemblers.
"""

from __future__ import annotations

from mm.layers.core.domain.types import ValuationPolicy


def portfolio_wealth_ticks(
    *,
    cash_tick_lot: float,
    inventory_lots: int,
    main_sell_ref_ticks: float,
    main_mid_ticks: float,
    hedge_inventory_lots: int,
    hedge_cash_tick_lot: float,
    hedge_sell_ref_ticks: float | None,
    policy: ValuationPolicy = ValuationPolicy.sell_ref,
) -> float:
    """
    Total wealth in **tick×lot** scalar units (compatible with legacy ``State`` bookkeeping).

    ``policy``:
    - ``sell_ref``: mark inventory at main sell ref; hedge inventory at hedge sell ref or main sell ref.
    - ``mid``: mark both legs at ``main_mid_ticks`` unless hedge provides its own sell ref.
    """
    cash = float(cash_tick_lot)
    inv = float(inventory_lots)
    ai = float(hedge_inventory_lots)
    ac = float(hedge_cash_tick_lot)
    if policy == ValuationPolicy.mid:
        m = float(main_mid_ticks)
        hedge_mark = float(hedge_sell_ref_ticks) if hedge_sell_ref_ticks is not None else m
        return cash + inv * m + ac + ai * hedge_mark
    main_mark = float(main_sell_ref_ticks)
    hedge_mark = float(hedge_sell_ref_ticks) if hedge_sell_ref_ticks is not None else main_mark
    return cash + inv * main_mark + ac + ai * hedge_mark


def portfolio_wealth_quote(
    *,
    cash_quote: float,
    inventory_base: float,
    main_sell_ref_quote: float,
    main_mid_quote: float,
    hedge_cash_quote: float,
    hedge_inventory_base: float,
    hedge_sell_ref_quote: float | None,
    policy: ValuationPolicy = ValuationPolicy.sell_ref,
) -> float:
    """Total wealth in **quote currency** (canonical floats from ``UnitConverter`` / snapshots)."""
    cash = float(cash_quote)
    inv = float(inventory_base)
    ai = float(hedge_inventory_base)
    ac = float(hedge_cash_quote)
    if policy == ValuationPolicy.mid:
        m = float(main_mid_quote)
        hedge_mark = float(hedge_sell_ref_quote) if hedge_sell_ref_quote is not None else m
        return cash + inv * m + ac + ai * hedge_mark
    main_mark = float(main_sell_ref_quote)
    hedge_mark = float(hedge_sell_ref_quote) if hedge_sell_ref_quote is not None else main_mark
    return cash + inv * main_mark + ac + ai * hedge_mark


def main_inventory_value_ticks(
    *,
    inventory_lots: int,
    main_sell_ref_ticks: float,
    main_mid_ticks: float,
    policy: ValuationPolicy = ValuationPolicy.sell_ref,
) -> float:
    """Mark-to-market value of **main** inventory only (tick×lot); used for plotting ``inv_value``."""
    inv = float(inventory_lots)
    if policy == ValuationPolicy.mid:
        return inv * float(main_mid_ticks)
    return inv * float(main_sell_ref_ticks)
