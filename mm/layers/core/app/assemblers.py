from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mm.layers.core.domain.snapshots import InstrumentSnapshot, StepSnapshot
from mm.layers.core.domain.types import InstrumentId
from mm.layers.core.app.parsed_step import ParsedStep
from mm.layers.core.units.converter import UnitConverter
from mm.layers.core.units.spec import InstrumentSpec


def instrument_snapshot_from_step(
    step: ParsedStep,
    *,
    instrument: InstrumentId,
    spec: InstrumentSpec,
    unit_converter: UnitConverter | None = None,
) -> InstrumentSnapshot:
    """
    Convert tick/lot step fields into canonical float units.
    """
    uc = unit_converter or UnitConverter(spec)
    # Keep tick precision: `step.ob.mid` and ref ticks may be float.
    mid_quote = uc.price_from_ticks(float(step.ob.mid))
    buy_ref_quote = uc.price_from_ticks(float(step.main_buy_ref))
    sell_ref_quote = uc.price_from_ticks(float(step.main_sell_ref))
    inv_base = uc.qty_from_lots(int(step.inventory))
    cash_quote = uc.notional_from_tick_lot(float(step.cash))
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
    uc = UnitConverter(spec)
    main = instrument_snapshot_from_step(
        step, instrument=instrument, spec=spec, unit_converter=uc
    )
    hedge_leg = {
        "buy_ref_quote": (
            uc.price_from_ticks(float(step.hedge_buy_ref)) if step.hedge_buy_ref is not None else None
        ),
        "sell_ref_quote": (
            uc.price_from_ticks(float(step.hedge_sell_ref)) if step.hedge_sell_ref is not None else None
        ),
        "inventory_base": uc.qty_from_lots(int(step.hedge_inventory)),
        "cash_quote": uc.notional_from_tick_lot(float(step.hedge_cash)),
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
        "hedge": hedge_leg,
    }
    return StepSnapshot(
        instrument=instrument,
        t_end_ms=int(step.t_end_ms),
        data=data,
        main_instrument=main,
    )

