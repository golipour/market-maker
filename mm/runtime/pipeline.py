from __future__ import annotations

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.ingestion.orderbook import OrderbookManager
from mm.ingestion.symbol_engine import SymbolEngine
from mm.ingestion.trade_buffer import RollingTradeBuffer


def build_symbol_engine(
    *,
    exchange: str,
    symbol: str,
    adapter: AbstractMarketDataAdapter,
    bus: AbstractEventBus,
    max_trade_buffer_ms: int = 10 * 60 * 1000,
    orderbook_history_size: int = 1024,
) -> SymbolEngine:
    return SymbolEngine(
        exchange=exchange,
        symbol=symbol,
        adapter=adapter,
        trade_buffer=RollingTradeBuffer(max_buffer_ms=max_trade_buffer_ms),
        orderbook_manager=OrderbookManager(history_size=orderbook_history_size),
        bus=bus,
    )

