from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook, RawTrade
from mm.core.payload_schema import market_event_event
from mm.ingestion.orderbook import OrderbookManager
from mm.runtime.pipeline import build_symbol_engine


class FakeMarketDataAdapter(AbstractMarketDataAdapter):
    def __init__(self):
        self._connected = False
        self._trades: asyncio.Queue[RawTrade | None] = asyncio.Queue()
        self._books: asyncio.Queue[RawOrderbook | None] = asyncio.Queue()

    async def connect(self) -> None:
        self._connected = True
        await self._books.put(
            RawOrderbook(
                exchange="nobitex",
                symbol="BTC_IRT",
                timestamp_ms=1_000,
                bids=(PriceLevel(100.0, 2.0), PriceLevel(99.0, 3.0)),
                asks=(PriceLevel(101.0, 4.0), PriceLevel(102.0, 5.0)),
                is_snapshot=True,
            )
        )
        await self._trades.put(
            RawTrade(
                trade_id="t1",
                timestamp_ms=1_100,
                price=101.0,
                qty=0.5,
                side="buy",
                exchange="nobitex",
                symbol="BTC_IRT",
            )
        )
        await self._trades.put(
            RawTrade(
                trade_id="t2",
                timestamp_ms=1_200,
                price=100.0,
                qty=0.25,
                side="sell",
                exchange="nobitex",
                symbol="BTC_IRT",
            )
        )

    async def disconnect(self) -> None:
        self._connected = False
        await self._trades.put(None)
        await self._books.put(None)

    async def trade_stream(self) -> AsyncIterator[RawTrade]:
        while self._connected:
            item = await self._trades.get()
            if item is None:
                break
            yield item

    async def orderbook_stream(self) -> AsyncIterator[RawOrderbook]:
        while self._connected:
            item = await self._books.get()
            if item is None:
                break
            yield item


@pytest.mark.asyncio
async def test_symbol_engine_publishes_raw_and_market_events():
    bus = InMemoryBus()
    engine = build_symbol_engine(
        exchange="nobitex",
        symbol="BTC_IRT",
        adapter=FakeMarketDataAdapter(),
        bus=bus,
    )
    raw_trade_queue = bus.subscribe_queue("market:raw:nobitex:BTC_IRT:trades")
    raw_book_queue = bus.subscribe_queue("market:raw:nobitex:BTC_IRT:orderbook")
    event_queue = bus.subscribe_queue("market:events:nobitex:BTC_IRT:1000")

    try:
        await engine.start()
        raw_book = await asyncio.wait_for(raw_book_queue.get(), timeout=1.0)
        trade_1 = await asyncio.wait_for(raw_trade_queue.get(), timeout=1.0)
        trade_2 = await asyncio.wait_for(raw_trade_queue.get(), timeout=1.0)

        assert raw_book.bids[0] == PriceLevel(100.0, 2.0)
        assert [trade_1.trade_id, trade_2.trade_id] == ["t1", "t2"]

        event = engine.market_event_builder().build(
            interval_ms=1000,
            t_start_ms=1000,
            t_end_ms=2000,
        )
        await bus.publish("market:events:nobitex:BTC_IRT:1000", market_event_event(event))
        seen = await asyncio.wait_for(event_queue.get(), timeout=1.0)

        from mm.core.payload_schema import market_event_from_payload

        parsed = market_event_from_payload(seen)
        assert parsed is not None
        assert parsed.exchange == "nobitex"
        assert parsed.symbol == "BTC_IRT"
        assert parsed.buy_volume == 0.5
        assert parsed.sell_volume == 0.25
        assert parsed.vwap == pytest.approx((101.0 * 0.5 + 100.0 * 0.25) / 0.75)
        assert parsed.imbalance == pytest.approx((2.0 - 4.0) / 6.0)
        assert parsed.microprice == pytest.approx((100.0 * 4.0 + 101.0 * 2.0) / 6.0)
    finally:
        await engine.stop()


def test_orderbook_manager_materializes_websocket_deltas_for_market_events():
    manager = OrderbookManager()
    manager.apply_snapshot(
        RawOrderbook(
            exchange="nobitex",
            symbol="BTC_IRT",
            timestamp_ms=1_000,
            bids=(PriceLevel(100.0, 2.0), PriceLevel(99.0, 3.0)),
            asks=(PriceLevel(101.0, 4.0), PriceLevel(102.0, 5.0)),
            is_snapshot=True,
        )
    )

    manager.apply_delta(
        RawOrderbook(
            exchange="nobitex",
            symbol="BTC_IRT",
            timestamp_ms=1_500,
            bids=(PriceLevel(100.0, 1.25), PriceLevel(98.0, 2.5)),
            asks=(PriceLevel(101.0, 0.0), PriceLevel(100.5, 1.0)),
            is_snapshot=False,
        )
    )

    snapshot = manager.snapshot_at(2_000)

    assert snapshot.is_snapshot is True
    assert snapshot.timestamp_ms == 1_500
    assert snapshot.bids[:3] == (
        PriceLevel(100.0, 1.25),
        PriceLevel(99.0, 3.0),
        PriceLevel(98.0, 2.5),
    )
    assert snapshot.asks[:2] == (
        PriceLevel(100.5, 1.0),
        PriceLevel(102.0, 5.0),
    )


def test_orderbook_manager_requires_snapshot_before_delta():
    manager = OrderbookManager()

    with pytest.raises(LookupError, match="delta received before initial snapshot"):
        manager.apply_delta(
            RawOrderbook(
                exchange="nobitex",
                symbol="BTC_IRT",
                timestamp_ms=1_000,
                bids=(PriceLevel(100.0, 2.0),),
                asks=(),
                is_snapshot=False,
            )
        )
