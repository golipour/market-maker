from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from mm.bus.redis_streams import RedisStreamsBus
from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.core.models.market import PriceLevel, RawOrderbook, RawTrade
from mm.core.payload_schema import market_event_from_payload
from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.config import BusRuntimeConfig, ControllerRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig
from mm.runtime.engine import build_runtime_components, resolve_build_options
from mm.runtime.ingestor import build_ingestor_components
from mm.runtime.ingestor_config import ingestor_config_from_mapping
from tests.unit.test_redis_streams_bus import SharedFakeRedis


class _FakeMarketDataAdapter(AbstractMarketDataAdapter):
    def __init__(self, exchange: str, symbol: str) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self._connected = False
        self._trades: asyncio.Queue[RawTrade | None] = asyncio.Queue()
        self._books: asyncio.Queue[RawOrderbook | None] = asyncio.Queue()

    async def connect(self) -> None:
        self._connected = True
        await self._books.put(
            RawOrderbook(
                exchange=self.exchange,
                symbol=self.symbol,
                timestamp_ms=1_000,
                bids=(PriceLevel(100.0, 2.0),),
                asks=(PriceLevel(101.0, 4.0),),
                is_snapshot=True,
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
async def test_build_ingestor_components_requires_redis_bus() -> None:
    with pytest.raises(ValueError, match="redis_streams"):
        await build_ingestor_components(
            ingestor_config_from_mapping(
                {
                    "feeds": [{"exchange": "nobitex", "symbol": "BTC_IRT", "interval_ms": 1000}],
                    "bus": {"kind": "memory"},
                }
            )
        )


@pytest.mark.asyncio
async def test_ingestor_publishes_market_events_to_redis_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mm.runtime.ingestor.select_public_market_data_adapter",
        lambda exchange, symbol, **kwargs: type(
            "Selection",
            (),
            {"adapter": _FakeMarketDataAdapter(exchange, symbol), "transport": "fake"},
        )(),
    )
    redis = SharedFakeRedis()
    bus = RedisStreamsBus(stream_prefix="test", read_block_ms=10)
    bus._redis = redis
    config = ingestor_config_from_mapping(
        {
            "feeds": [{"exchange": "nobitex", "symbol": "BTC_IRT", "interval_ms": 1000}],
            "bus": {"kind": "redis_streams", "stream_prefix": "test"},
            "observability": {"metrics_enabled": False},
        }
    )
    ingestor = await build_ingestor_components(config, bus=bus, enable_observability=False)
    topic = config.feeds[0].market_event_topic()
    queue = bus.subscribe_queue(topic)
    await asyncio.wait_for(redis.xread_started.wait(), timeout=1.0)

    await ingestor.start()
    try:
        engine = ingestor.engines[config.feeds[0].poll_source_key()]
        await asyncio.sleep(0.05)
        from mm.core.payload_schema import market_event_event

        event = engine.market_event_builder().build(1000, 1000, 2000)
        await bus.publish(topic, market_event_event(event))
        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        event = market_event_from_payload(payload)
        assert event is not None
        assert event.exchange == "nobitex"
        assert event.symbol == "BTC_IRT"
        assert event.interval_ms == 1000
    finally:
        await ingestor.stop()


@pytest.mark.asyncio
async def test_consumer_runtime_skips_local_engines_for_bus_sources() -> None:
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(
            role="local",
            exchange="nobitex",
            symbol="BTC_IRT",
            market_data_source="bus",
        ),
        bus=BusRuntimeConfig(kind="redis_streams"),
        controller=ControllerRuntimeConfig(enabled=False, actor_id="test"),
    )
    assert resolve_build_options(config).mode == "consumer_only"
    runtime = await build_runtime_components(
        config,
        build_options=RuntimeBuildOptions.for_mode("consumer_only"),
    )
    assert runtime.engines == {}
    assert runtime.frame_assembler is not None


@pytest.mark.asyncio
async def test_consumer_runtime_receives_bus_market_event() -> None:
    redis = SharedFakeRedis()
    publisher = RedisStreamsBus(stream_prefix="test", read_block_ms=10)
    consumer = RedisStreamsBus(stream_prefix="test", read_block_ms=10)
    publisher._redis = redis
    consumer._redis = redis

    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(
            role="local",
            exchange="nobitex",
            symbol="BTC_IRT",
            market_data_source="bus",
        ),
        interval_ms=1000,
        bus=BusRuntimeConfig(kind="redis_streams", stream_prefix="test"),
        controller=ControllerRuntimeConfig(enabled=False, actor_id="test"),
    )
    runtime = await build_runtime_components(
        config,
        bus=consumer,
        build_options=RuntimeBuildOptions.for_mode("consumer_only"),
    )
    topic = config.local.market_event_topic(config.interval_ms)
    queue = consumer.subscribe_queue(topic)
    await asyncio.wait_for(redis.xread_started.wait(), timeout=1.0)

    from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook

    book = RawOrderbook(
        exchange="nobitex",
        symbol="BTC_IRT",
        timestamp_ms=2_000,
        bids=(PriceLevel(100.0, 1.0),),
        asks=(PriceLevel(101.0, 1.0),),
        is_snapshot=True,
    )
    event = MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=1_000,
        t_end_ms=2_000,
        orderbook=book,
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=100.5,
    )

    await publisher.publish(topic, event)
    payload = await asyncio.wait_for(queue.get(), timeout=1.0)
    parsed = market_event_from_payload(payload)
    assert parsed is not None
    assert parsed.t_end_ms == 2_000
    assert runtime.engines == {}

    await publisher.close()
    await consumer.close()
