from __future__ import annotations

import bisect

from mm.core.models.market import PriceLevel, RawOrderbook


class OrderbookManager:
    def __init__(self, history_size: int = 1024):
        self._history_size = max(1, int(history_size))
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._times: list[int] = []
        self._snapshots: list[RawOrderbook] = []

    def apply_snapshot(self, book: RawOrderbook) -> None:
        self._bids = {level.price: level.qty for level in book.bids if level.qty > 0}
        self._asks = {level.price: level.qty for level in book.asks if level.qty > 0}
        self._record(book.exchange, book.symbol, book.timestamp_ms)

    def apply_delta(self, delta: RawOrderbook) -> None:
        for level in delta.bids:
            self._apply_level(self._bids, level)
        for level in delta.asks:
            self._apply_level(self._asks, level)
        self._record(delta.exchange, delta.symbol, delta.timestamp_ms)

    def snapshot_at(self, t_ms: int) -> RawOrderbook:
        if not self._snapshots:
            raise LookupError("orderbook manager has no snapshots")
        idx = bisect.bisect_right(self._times, int(t_ms)) - 1
        if idx < 0:
            return self._snapshots[0]
        return self._snapshots[idx]

    @staticmethod
    def _apply_level(side: dict[float, float], level: PriceLevel) -> None:
        if level.qty <= 0:
            side.pop(level.price, None)
        else:
            side[level.price] = level.qty

    def _record(self, exchange: str, symbol: str, timestamp_ms: int) -> None:
        bids = tuple(PriceLevel(p, q) for p, q in sorted(self._bids.items(), reverse=True))
        asks = tuple(PriceLevel(p, q) for p, q in sorted(self._asks.items()))
        snap = RawOrderbook(
            exchange=exchange,
            symbol=symbol,
            timestamp_ms=int(timestamp_ms),
            bids=bids,
            asks=asks,
            is_snapshot=True,
        )
        idx = bisect.bisect_right(self._times, snap.timestamp_ms)
        self._times.insert(idx, snap.timestamp_ms)
        self._snapshots.insert(idx, snap)
        while len(self._snapshots) > self._history_size:
            self._times.pop(0)
            self._snapshots.pop(0)

