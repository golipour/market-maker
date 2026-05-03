from __future__ import annotations

from dataclasses import dataclass

from .types import InstrumentId


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
class StepSnapshot:
    """
    Windowed snapshot for one instrument after assembly from the observed step.

    ``main_instrument`` is the typed canonical view; ``data`` retains ``raw`` / ``main`` /
    ``anchor`` dicts for backward compatibility and JSON logs.
    """

    instrument: InstrumentId
    t_end_ms: int
    data: dict
    main_instrument: InstrumentSnapshot | None = None


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    t_ms: int
    instruments: tuple[InstrumentSnapshot, ...]

