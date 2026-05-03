"""
Portfolio wealth from cash + inventory + anchor leg — **one formula** for ticks and quote/base.

Tick-space uses the same convention as ``State`` (cash in tick×lot notional, refs in ticks).
Quote-space uses floats from ``InstrumentSnapshot`` / assemblers.
"""

from __future__ import annotations

from mm_core.domain.types import ValuationPolicy


def portfolio_wealth_ticks(
    *,
    cash_tick_lot: float,
    inventory_lots: int,
    main_sell_ref_ticks: float,
    main_mid_ticks: float,
    anchor_inventory_lots: int,
    anchor_cash_tick_lot: float,
    anchor_sell_ref_ticks: float | None,
    policy: ValuationPolicy = ValuationPolicy.sell_ref,
) -> float:
    """
    Total wealth in **tick×lot** scalar units (compatible with legacy ``State`` bookkeeping).

    ``policy``:
    - ``sell_ref``: mark inventory at main sell ref; anchor inventory at anchor sell ref or main sell ref.
    - ``mid``: mark both legs at ``main_mid_ticks`` unless anchor provides its own sell ref.
    """
    cash = float(cash_tick_lot)
    inv = float(inventory_lots)
    ai = float(anchor_inventory_lots)
    ac = float(anchor_cash_tick_lot)
    if policy == ValuationPolicy.mid:
        m = float(main_mid_ticks)
        anchor_mark = float(anchor_sell_ref_ticks) if anchor_sell_ref_ticks is not None else m
        return cash + inv * m + ac + ai * anchor_mark
    main_mark = float(main_sell_ref_ticks)
    anchor_mark = float(anchor_sell_ref_ticks) if anchor_sell_ref_ticks is not None else main_mark
    return cash + inv * main_mark + ac + ai * anchor_mark


def portfolio_wealth_quote(
    *,
    cash_quote: float,
    inventory_base: float,
    main_sell_ref_quote: float,
    main_mid_quote: float,
    anchor_cash_quote: float,
    anchor_inventory_base: float,
    anchor_sell_ref_quote: float | None,
    policy: ValuationPolicy = ValuationPolicy.sell_ref,
) -> float:
    """Total wealth in **quote currency** (canonical floats from ``UnitConverter`` / snapshots)."""
    cash = float(cash_quote)
    inv = float(inventory_base)
    ai = float(anchor_inventory_base)
    ac = float(anchor_cash_quote)
    if policy == ValuationPolicy.mid:
        m = float(main_mid_quote)
        anchor_mark = float(anchor_sell_ref_quote) if anchor_sell_ref_quote is not None else m
        return cash + inv * m + ac + ai * anchor_mark
    main_mark = float(main_sell_ref_quote)
    anchor_mark = float(anchor_sell_ref_quote) if anchor_sell_ref_quote is not None else main_mark
    return cash + inv * main_mark + ac + ai * anchor_mark
