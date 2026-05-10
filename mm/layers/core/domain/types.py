from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class InstrumentId:
    """
    Canonical instrument identifier.

    Examples:
      - exchange="nobitex", symbol="BTC_USDT"
      - exchange="wallex",  symbol="BTC_USDT"  (hedge)
    """

    exchange: str
    symbol: str

    def __str__(self) -> str:
        return f"{self.exchange}:{self.symbol}"


class Side(str, Enum):
    buy = "buy"
    sell = "sell"


class DataSource(str, Enum):
    websocket = "websocket"
    rest = "rest"
    synthetic = "synthetic"


class ValuationPolicy(str, Enum):
    """
    Inventory marking convention.

    Current project choice (Option B): mark-to-bid / sell_ref.
    """

    sell_ref = "sell_ref"
    mid = "mid"

