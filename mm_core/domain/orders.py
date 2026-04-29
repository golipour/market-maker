from __future__ import annotations

from dataclasses import dataclass

from .types import InstrumentId, Side


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """
    Core order intent in canonical float units.

    - price: quote per base (e.g. USDT per BTC)
    - qty: base quantity (e.g. BTC)
    """

    instrument: InstrumentId
    side: Side
    price: float
    qty: float


@dataclass(frozen=True, slots=True)
class CancelIntent:
    instrument: InstrumentId
    order_id: str


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    instrument: InstrumentId
    order_id: str
    status: str
    filled_qty: float = 0.0
    avg_price: float | None = None
    fee: float | None = None
    fee_currency: str | None = None

