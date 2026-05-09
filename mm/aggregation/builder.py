from __future__ import annotations

from mm.aggregation.features import microprice, top_imbalance, volume_by_side, vwap
from mm.core.models.market import MarketEvent
from mm.ingestion.orderbook import OrderbookManager
from mm.ingestion.trade_buffer import RollingTradeBuffer


class MarketEventBuilder:
    def __init__(
        self,
        exchange: str,
        symbol: str,
        trade_buffer: RollingTradeBuffer,
        orderbook_manager: OrderbookManager,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.trade_buffer = trade_buffer
        self.orderbook_manager = orderbook_manager

    def build(self, interval_ms: int, t_start_ms: int, t_end_ms: int) -> MarketEvent:
        book = self.orderbook_manager.snapshot_at(t_end_ms)
        trades = self.trade_buffer.slice(t_start_ms, t_end_ms)
        buy_volume, sell_volume = volume_by_side(trades)
        return MarketEvent(
            exchange=self.exchange,
            symbol=self.symbol,
            interval_ms=int(interval_ms),
            t_start_ms=int(t_start_ms),
            t_end_ms=int(t_end_ms),
            orderbook=book,
            trades=trades,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            vwap=vwap(trades),
            imbalance=top_imbalance(book),
            microprice=microprice(book),
        )

