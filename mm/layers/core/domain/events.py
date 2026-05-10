from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import DataSource, InstrumentId


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """
    Canonical market event in float units.

    This is a skeleton contract used for redesign planning; we will expand it
    gradually as we migrate adapters.
    """

    instrument: InstrumentId
    source: DataSource
    event_time_ms: int
    ingest_time_ms: int
    kind: Literal["orderbook", "trade", "balance", "execution", "status"]
    payload: dict

