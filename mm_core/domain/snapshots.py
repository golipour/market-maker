from __future__ import annotations

from dataclasses import dataclass

from .types import InstrumentId


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    """
    Windowed snapshot produced by the data layer for a single instrument.

    Canonical float units:
    - prices are quote/base floats
    - volumes are base quantity floats
    - cash is quote floats
    """

    instrument: InstrumentId
    t_end_ms: int
    # Minimal skeleton; will be expanded as we migrate away from tuple Step.
    data: dict


@dataclass(frozen=True, slots=True)
class InstrumentSnapshot:
    instrument: InstrumentId
    t_ms: int
    mid: float
    buy_ref: float
    sell_ref: float
    inventory: float
    cash: float


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    t_ms: int
    instruments: tuple[InstrumentSnapshot, ...]

