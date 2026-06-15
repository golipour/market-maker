from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import msgspec
except ModuleNotFoundError:  # pragma: no cover - keeps local tests usable until dependency install.
    msgspec = None  # type: ignore[assignment]


if msgspec is not None:

    class RawTrade(msgspec.Struct, frozen=True):
        trade_id: Optional[str]
        timestamp_ms: int
        price: float
        qty: float
        side: str
        exchange: str
        symbol: str


    class PriceLevel(msgspec.Struct, frozen=True):
        price: float
        qty: float


    class RawOrderbook(msgspec.Struct, frozen=True):
        exchange: str
        symbol: str
        timestamp_ms: int
        bids: Tuple[PriceLevel, ...]
        asks: Tuple[PriceLevel, ...]
        is_snapshot: bool


    class MarketEvent(msgspec.Struct, frozen=True):
        exchange: str
        symbol: str
        interval_ms: int
        t_start_ms: int
        t_end_ms: int
        orderbook: RawOrderbook
        trades: Tuple[RawTrade, ...]
        buy_volume: float
        sell_volume: float
        vwap: float
        imbalance: float
        microprice: float
        book_buy_vwap: float = 0.0
        book_sell_vwap: float = 0.0
        book_vwap_depth: float = 0.0

else:

    @dataclass(frozen=True, slots=True)
    class RawTrade:
        trade_id: Optional[str]
        timestamp_ms: int
        price: float
        qty: float
        side: str
        exchange: str
        symbol: str


    @dataclass(frozen=True, slots=True)
    class PriceLevel:
        price: float
        qty: float


    @dataclass(frozen=True, slots=True)
    class RawOrderbook:
        exchange: str
        symbol: str
        timestamp_ms: int
        bids: Tuple[PriceLevel, ...]
        asks: Tuple[PriceLevel, ...]
        is_snapshot: bool


    @dataclass(frozen=True, slots=True)
    class MarketEvent:
        exchange: str
        symbol: str
        interval_ms: int
        t_start_ms: int
        t_end_ms: int
        orderbook: RawOrderbook
        trades: Tuple[RawTrade, ...]
        buy_volume: float
        sell_volume: float
        vwap: float
        imbalance: float
        microprice: float
        book_buy_vwap: float = 0.0
        book_sell_vwap: float = 0.0
        book_vwap_depth: float = 0.0

