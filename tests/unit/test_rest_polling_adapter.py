from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import pytest

from mm.core.models.market import PriceLevel, RawOrderbook, RawTrade
from mm.exchanges.shared.provider_adapter import RestPollingMarketDataAdapter


class DummyRestPollingAdapter(RestPollingMarketDataAdapter):
    def __init__(self, exchange: str = "test_ex", symbol: str = "BTC_USDT", poll_interval_ms: int = 50):
        super().__init__(exchange, symbol, poll_interval_ms=poll_interval_ms)
        self.orderbook_fetch_times: list[float] = []
        self.trade_fetch_times: list[float] = []
        self._book_counter = 0

    async def fetch_orderbook(self) -> RawOrderbook:
        self.orderbook_fetch_times.append(time.monotonic())
        self._book_counter += 1
        return RawOrderbook(
            exchange=self.exchange,
            symbol=self.symbol,
            timestamp_ms=int(time.time() * 1000) + self._book_counter,
            bids=(PriceLevel(100.0, 1.0),),
            asks=(PriceLevel(101.0, 1.0),),
            is_snapshot=True,
        )

    async def fetch_trades(self) -> list[RawTrade]:
        self.trade_fetch_times.append(time.monotonic())
        return [
            RawTrade(
                trade_id=f"t_{len(self.trade_fetch_times)}",
                timestamp_ms=int(time.time() * 1000),
                price=100.0,
                qty=0.5,
                side="buy",
                exchange=self.exchange,
                symbol=self.symbol,
            )
        ]


@pytest.mark.asyncio
async def test_orderbook_stream_cadence_matches_poll_interval() -> None:
    poll_ms = 40
    adapter = DummyRestPollingAdapter(poll_interval_ms=poll_ms)
    await adapter.connect()

    books: list[RawOrderbook] = []
    async def collect() -> None:
        async for book in adapter.orderbook_stream():
            books.append(book)
            if len(books) >= 4:
                break

    await asyncio.wait_for(collect(), timeout=1.0)
    await adapter.disconnect()

    assert len(adapter.orderbook_fetch_times) >= 4
    intervals = [
        (adapter.orderbook_fetch_times[i] - adapter.orderbook_fetch_times[i - 1]) * 1000.0
        for i in range(1, len(adapter.orderbook_fetch_times))
    ]
    # Each interval should be close to poll_ms (40ms), definitely well below 2x (80ms)
    for delta in intervals:
        assert 30.0 <= delta <= 70.0, f"Expected ~40ms interval, got {delta:.1f}ms"


@pytest.mark.asyncio
async def test_independent_orderbook_and_trade_streams_do_not_double_wait() -> None:
    poll_ms = 50
    adapter = DummyRestPollingAdapter(poll_interval_ms=poll_ms)
    await adapter.connect()

    books: list[RawOrderbook] = []
    trades: list[RawTrade] = []

    async def collect_books() -> None:
        async for book in adapter.orderbook_stream():
            books.append(book)
            if len(books) >= 4:
                break

    async def collect_trades() -> None:
        async for trade in adapter.trade_stream():
            trades.append(trade)
            if len(trades) >= 4:
                break

    t0 = time.monotonic()
    await asyncio.wait_for(asyncio.gather(collect_books(), collect_trades()), timeout=1.5)
    total_time = (time.monotonic() - t0) * 1000.0
    await adapter.disconnect()

    # 4 polls at 50ms should finish in ~150-250ms total.
    # With the old bug (double wait across streams + trailing sleep), 4 polls took > 400-600ms.
    assert total_time < 350.0, f"Concurrent streams took too long: {total_time:.1f}ms"

    # Both streams should have run independently around 50ms gaps
    ob_intervals = [
        (adapter.orderbook_fetch_times[i] - adapter.orderbook_fetch_times[i - 1]) * 1000.0
        for i in range(1, len(adapter.orderbook_fetch_times))
    ]
    trade_intervals = [
        (adapter.trade_fetch_times[i] - adapter.trade_fetch_times[i - 1]) * 1000.0
        for i in range(1, len(adapter.trade_fetch_times))
    ]
    for d in ob_intervals:
        assert 35.0 <= d <= 85.0, f"Orderbook interval was {d:.1f}ms, expected ~50ms"
    for d in trade_intervals:
        assert 35.0 <= d <= 85.0, f"Trade interval was {d:.1f}ms, expected ~50ms"


@pytest.mark.asyncio
async def test_independent_backoff_per_source() -> None:
    adapter = DummyRestPollingAdapter(poll_interval_ms=100)
    await adapter.connect()

    # Trigger error on trade stream
    await adapter._sleep_after_poll_error("trade_stream", ValueError("transient trade error"))
    assert adapter._transient_backoffs["trade_stream"] == 0.2
    assert adapter._transient_backoffs["orderbook_stream"] == 0.1

    # Reset trade stream backoff
    adapter._reset_poll_backoff("trade_stream")
    assert adapter._transient_backoffs["trade_stream"] == 0.1
    assert adapter._transient_backoffs["orderbook_stream"] == 0.1
