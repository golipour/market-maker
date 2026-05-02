from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mm_core.domain.snapshots import InstrumentSnapshot, StepSnapshot
from mm_core.domain.types import InstrumentId
from mm_core.app.parsed_step import ParsedStep
from mm_core.units.converter import (
    notional_from_tick_lot,
    price_from_ticks,
    qty_from_lots,
)
from mm_core.units.spec import InstrumentSpec


def instrument_snapshot_from_step(
    step: ParsedStep,
    *,
    instrument: InstrumentId,
    spec: InstrumentSpec,
) -> InstrumentSnapshot:
    """
    Convert tick/lot step fields into canonical float units.
    """
    # Keep tick precision: `step.ob.mid` and ref ticks may be float.
    mid_quote = price_from_ticks(float(step.ob.mid), tick_size=spec.tick_size)
    buy_ref_quote = price_from_ticks(float(step.main_buy_ref), tick_size=spec.tick_size)
    sell_ref_quote = price_from_ticks(float(step.main_sell_ref), tick_size=spec.tick_size)
    inv_base = qty_from_lots(int(step.inventory), lot_size=spec.lot_size)
    cash_quote = notional_from_tick_lot(float(step.cash), tick_size=spec.tick_size, lot_size=spec.lot_size)
    return InstrumentSnapshot(
        instrument=instrument,
        t_ms=int(step.t_end_ms),
        mid=float(mid_quote),
        buy_ref=float(buy_ref_quote),
        sell_ref=float(sell_ref_quote),
        inventory=float(inv_base),
        cash=float(cash_quote),
    )


def step_snapshot_from_step(
    step: ParsedStep,
    *,
    instrument: InstrumentId,
    spec: InstrumentSpec,
) -> StepSnapshot:
    """
    StepSnapshot for debugging/observability: raw tick/lot fields plus canonical floats.
    """
    main = instrument_snapshot_from_step(step, instrument=instrument, spec=spec)
    anchor = {
        "buy_ref_quote": (
            price_from_ticks(float(step.anchor_buy_ref), tick_size=spec.tick_size)
            if step.anchor_buy_ref is not None
            else None
        ),
        "sell_ref_quote": (
            price_from_ticks(float(step.anchor_sell_ref), tick_size=spec.tick_size)
            if step.anchor_sell_ref is not None
            else None
        ),
        "inventory_base": qty_from_lots(int(step.anchor_inventory), lot_size=spec.lot_size),
        "cash_quote": notional_from_tick_lot(float(step.anchor_cash), tick_size=spec.tick_size, lot_size=spec.lot_size),
    }
    data: dict[str, Any] = {
        "raw": {
            "t_end_ms": int(step.t_end_ms),
            "buy_lots": int(step.buy_lots),
            "sell_lots": int(step.sell_lots),
            "inventory_lots": int(step.inventory),
            "cash_tick_lot": float(step.cash),
            "main_buy_ref_ticks": float(step.main_buy_ref),
            "main_sell_ref_ticks": float(step.main_sell_ref),
        },
        "main": asdict(main),
        "anchor": anchor,
    }
    return StepSnapshot(instrument=instrument, t_end_ms=int(step.t_end_ms), data=data)

