from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """
    Exchange-specific microstructure spec.

    This is a first scaffold. Today live execution uses tick/lot units directly,
    so we expose both float and tick/lot derived quantities for centralization.
    """

    exchange: str
    symbol: str

    tick_size: float
    lot_size: float

    min_qty_lots: float
    min_notional_tick_lot: float

