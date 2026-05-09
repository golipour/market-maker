from __future__ import annotations

import bisect
import math

from mm.core.models.market import PriceLevel, RawOrderbook


class OrderbookManager:
    def __init__(self, history_size: int = 1024):
        self._history_size = max(1, int(history_size))
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._times: list[int] = []
        self._snapshots: list[RawOrderbook] = []
        self._has_snapshot = False

    def apply_snapshot(self, book: RawOrderbook) -> None:
        self._bids = self._levels_to_side(book.bids, keep_deletions=False)
        self._asks = self._levels_to_side(book.asks, keep_deletions=False)
        self._has_snapshot = True
        self._record(book.exchange, book.symbol, book.timestamp_ms)

    def apply_delta(self, delta: RawOrderbook) -> None:
        if not self._has_snapshot:
            raise LookupError("orderbook delta received before initial snapshot")
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
        price = float(level.price)
        qty = float(level.qty)
        if not math.isfinite(price) or price <= 0:
            return
        if not math.isfinite(qty) or qty <= 0:
            side.pop(price, None)
        else:
            side[price] = qty

    @staticmethod
    def _levels_to_side(levels: tuple[PriceLevel, ...], *, keep_deletions: bool) -> dict[float, float]:
        side: dict[float, float] = {}
        for level in levels:
            price = float(level.price)
            qty = float(level.qty)
            if not math.isfinite(price) or price <= 0:
                continue
            if not math.isfinite(qty) or qty <= 0:
                if keep_deletions:
                    side[price] = 0.0
                continue
            side[price] = qty
        return side

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

