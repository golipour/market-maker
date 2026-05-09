from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from mm.core.models.market import RawTrade


class TradeDeduplicator:
    def __init__(self, max_keys: int = 50_000):
        self._max_keys = max(1, int(max_keys))
        self._seen: OrderedDict[tuple, None] = OrderedDict()

    @staticmethod
    def key(trade: RawTrade) -> tuple:
        if trade.trade_id:
            return (trade.exchange, trade.symbol, trade.trade_id)
        return (
            trade.exchange,
            trade.symbol,
            trade.timestamp_ms,
            trade.price,
            trade.qty,
            trade.side,
        )

    def process(self, trades: Iterable[RawTrade]) -> list[RawTrade]:
        out = []
        for trade in trades:
            key = self.key(trade)
            if key in self._seen:
                self._seen.move_to_end(key)
                continue
            self._seen[key] = None
            out.append(trade)
            while len(self._seen) > self._max_keys:
                self._seen.popitem(last=False)
        return out

