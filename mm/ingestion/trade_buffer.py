from __future__ import annotations

import bisect
from collections import deque

from mm.core.models.market import RawTrade
from mm.ingestion.deduplicator import TradeDeduplicator


class RollingTradeBuffer:
    def __init__(self, max_buffer_ms: int = 10 * 60 * 1000, dedup: TradeDeduplicator | None = None):
        self._max_buffer_ms = max(1, int(max_buffer_ms))
        self._dedup = dedup or TradeDeduplicator()
        self._trades: deque[RawTrade] = deque()
        self._timestamps: list[int] = []

    def add(self, trade: RawTrade) -> bool:
        deduped = self._dedup.process([trade])
        if not deduped:
            return False
        item = deduped[0]
        idx = bisect.bisect_right(self._timestamps, item.timestamp_ms)
        self._timestamps.insert(idx, item.timestamp_ms)
        self._trades.insert(idx, item)
        self._prune(item.timestamp_ms)
        return True

    def slice(self, t_start_ms: int, t_end_ms: int) -> tuple[RawTrade, ...]:
        start = bisect.bisect_left(self._timestamps, int(t_start_ms))
        end = bisect.bisect_left(self._timestamps, int(t_end_ms))
        return tuple(list(self._trades)[start:end])

    def _prune(self, now_ms: int) -> None:
        min_ts = int(now_ms) - self._max_buffer_ms
        while self._timestamps and self._timestamps[0] < min_ts:
            self._timestamps.pop(0)
            self._trades.popleft()

