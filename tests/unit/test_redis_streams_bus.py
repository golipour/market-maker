from __future__ import annotations

import asyncio

import pytest

from mm.bus.redis_streams import RedisStreamsBus
from mm.core.models.private import Balance, PrivateStateSnapshot
from mm.observability.registry import REGISTRY


class SharedFakeRedis:
    def __init__(self) -> None:
        self.streams = {}
        self._counter = 0
        self._condition = asyncio.Condition()
        self.xread_started = asyncio.Event()

    async def xadd(self, stream, fields, **kwargs):
        async with self._condition:
            self._counter += 1
            message_id = f"{self._counter}-0"
            self.streams.setdefault(stream, []).append((message_id, fields))
            self._condition.notify_all()
            return message_id

    async def xread(self, streams, *, block=None, count=None):
        async with self._condition:
            start_indexes = {
                stream: self._start_index(stream, last_id)
                for stream, last_id in streams.items()
            }
            self.xread_started.set()
            while True:
                for stream, start_index in start_indexes.items():
                    messages = self.streams.get(stream, [])
                    if len(messages) > start_index:
                        limit = count or len(messages)
                        return [(stream, messages[start_index : start_index + limit])]
                timeout = None if block is None else max(0.001, float(block) / 1000.0)
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    return []

    async def aclose(self):
        pass

    def _start_index(self, stream, last_id):
        messages = self.streams.get(stream, [])
        if last_id == "$":
            return len(messages)
        for index, (message_id, _) in enumerate(messages):
            if message_id == last_id:
                return index + 1
        return 0


@pytest.mark.asyncio
async def test_redis_streams_bus_preserves_typed_local_delivery(monkeypatch) -> None:
    writes = []

    class FakeRedis:
        async def xadd(self, stream, fields, **kwargs):
            writes.append((stream, fields, kwargs))

        async def aclose(self):
            pass

    bus = RedisStreamsBus(stream_prefix="test")
    bus._redis = FakeRedis()
    queue = bus.subscribe_queue("private:state:wallex:snapshot")
    snapshot = PrivateStateSnapshot(
        exchange="wallex",
        timestamp_ms=1,
        balances={"BTC": Balance("wallex", "BTC", 1.0, 0.0, 1.0, 1)},
        open_orders={},
    )

    await bus.publish("private:state:wallex:snapshot", snapshot)

    assert await queue.get() is snapshot
    assert writes
    assert writes[0][0] == "test.private.state.wallex.snapshot"
    await bus.close()


@pytest.mark.asyncio
async def test_redis_streams_bus_records_stream_envelope_size() -> None:
    writes = []

    class FakeRedis:
        async def xadd(self, stream, fields, **kwargs):
            writes.append((stream, fields, kwargs))

        async def aclose(self):
            pass

    bus = RedisStreamsBus(stream_prefix="test")
    bus._redis = FakeRedis()
    event = {"type": "RedisPayloadSizeMetric", "exchange": "wallex", "payload": "x" * 512}

    await bus.publish("unit:payload-size:redis", event)

    assert writes
    text = REGISTRY.render()
    assert 'mm_event_payloads_total{event_type="RedisPayloadSizeMetric",stage="bus_publish",topic="unit:payload-size:redis"}' in text
    assert 'mm_event_envelope_bytes_total{event_type="RedisPayloadSizeMetric",stage="redis_stream",topic="unit:payload-size:redis"}' in text
    await bus.close()


@pytest.mark.asyncio
async def test_redis_streams_bus_compacts_dict_private_snapshot_payload() -> None:
    writes = []

    class FakeRedis:
        async def xadd(self, stream, fields, **kwargs):
            writes.append((stream, fields, kwargs))

        async def aclose(self):
            pass

    bus = RedisStreamsBus(stream_prefix="test")
    bus._redis = FakeRedis()
    queue = bus.subscribe_queue("private:state:nobitex:snapshot")
    snapshot = PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=1,
        balances={"USDT": Balance("nobitex", "USDT", 100.0, 0.0, 100.0, 1)},
        open_orders={},
    )
    payload = snapshot.as_payload()

    await bus.publish("private:state:nobitex:snapshot", payload)

    delivered = await queue.get()
    assert delivered is payload
    assert writes
    assert writes[0][0] == "test.private.state.nobitex.snapshot"
    await bus.close()


@pytest.mark.asyncio
async def test_redis_streams_bus_skips_zero_balance_stream_writes() -> None:
    writes = []

    class FakeRedis:
        async def xadd(self, stream, fields, **kwargs):
            writes.append((stream, fields, kwargs))

        async def aclose(self):
            pass

    bus = RedisStreamsBus(stream_prefix="test")
    bus._redis = FakeRedis()
    queue = bus.subscribe_queue("private:state:wallex:balances")
    balance = Balance("wallex", "ZERO", 0.0, 0.0, 0.0, 1)

    await bus.publish("private:state:wallex:balances", balance)

    assert await queue.get() is balance
    assert writes == []
    await bus.close()


@pytest.mark.asyncio
async def test_redis_streams_bus_publish_keeps_local_delivery_when_stream_write_fails() -> None:
    class FailingRedis:
        async def xadd(self, stream, fields, **kwargs):
            raise ConnectionError("redis unavailable")

        async def aclose(self):
            pass

    bus = RedisStreamsBus(stream_prefix="test")
    bus._redis = FailingRedis()
    queue = bus.subscribe_queue("strategy:decisions")
    event = {"type": "StrategyDecision", "actor_id": "a1"}

    await bus.publish("strategy:decisions", event)

    assert await asyncio.wait_for(queue.get(), timeout=1.0) is event
    await bus.close()


@pytest.mark.asyncio
async def test_redis_streams_bus_subscribe_receives_cross_process_stream_events() -> None:
    redis = SharedFakeRedis()
    publisher = RedisStreamsBus(stream_prefix="test", read_block_ms=10)
    subscriber = RedisStreamsBus(stream_prefix="test", read_block_ms=10)
    publisher._redis = redis
    subscriber._redis = redis

    async def next_event():
        async for event in subscriber.subscribe("admin:commands"):
            return event

    task = asyncio.create_task(next_event())
    await asyncio.wait_for(redis.xread_started.wait(), timeout=1.0)
    await publisher.publish(
        "admin:commands",
        {"type": "AdminCommand", "command": "kill_switch", "enabled": True},
    )

    event = await asyncio.wait_for(task, timeout=1.0)

    assert event["command"] == "kill_switch"
    assert event["enabled"] is True

    await publisher.close()
    await subscriber.close()


@pytest.mark.asyncio
async def test_redis_streams_bus_market_event_cross_process_round_trip() -> None:
    from mm.application.frame_assembler import _require_market_event
    from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
    from mm.core.payload_schema import market_event_event

    redis = SharedFakeRedis()
    publisher = RedisStreamsBus(stream_prefix="test", read_block_ms=10)
    subscriber = RedisStreamsBus(stream_prefix="test", read_block_ms=10)
    publisher._redis = redis
    subscriber._redis = redis

    book = RawOrderbook(
        exchange="wallex",
        symbol="BTC_IRT",
        timestamp_ms=5_000,
        bids=(PriceLevel(100.0, 5.0),),
        asks=(PriceLevel(101.0, 4.0),),
        is_snapshot=True,
    )
    original = MarketEvent(
        exchange="wallex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=4_000,
        t_end_ms=5_000,
        orderbook=book,
        trades=(),
        buy_volume=1.25,
        sell_volume=0.75,
        vwap=100.5,
        imbalance=0.1,
        microprice=100.55,
    )
    topic = "market:events:wallex:BTC_IRT:1000"
    queue = subscriber.subscribe_queue(topic)
    await asyncio.wait_for(redis.xread_started.wait(), timeout=1.0)

    await publisher.publish(topic, original)
    delivered = await asyncio.wait_for(queue.get(), timeout=1.0)
    parsed = _require_market_event(delivered)

    assert parsed.exchange == original.exchange
    assert parsed.symbol == original.symbol
    assert parsed.interval_ms == original.interval_ms
    assert parsed.t_start_ms == original.t_start_ms
    assert parsed.t_end_ms == original.t_end_ms
    assert parsed.buy_volume == original.buy_volume
    assert parsed.sell_volume == original.sell_volume
    assert parsed.vwap == original.vwap
    assert parsed.imbalance == original.imbalance
    assert parsed.microprice == original.microprice
    assert parsed.orderbook.bids[0].price == original.orderbook.bids[0].price
    assert parsed.orderbook.asks[0].qty == original.orderbook.asks[0].qty

    payload = market_event_event(parsed)
    assert payload["payload_schema"] == "mm.market_event"
    assert payload["payload_version"] == 1

    await publisher.close()
    await subscriber.close()


@pytest.mark.asyncio
async def test_redis_streams_bus_joint_strategy_frame_cross_process_round_trip() -> None:
    from mm.core.models.joint_strategy import JointStrategyFrame
    from mm.core.payload_schema import joint_strategy_frame_event
    from tests.unit.conftest import joint_frame_for_input
    from tests.unit.test_strategy_pipeline import _event_for

    redis = SharedFakeRedis()
    publisher = RedisStreamsBus(stream_prefix="test", read_block_ms=10)
    subscriber = RedisStreamsBus(stream_prefix="test", read_block_ms=10)
    publisher._redis = redis
    subscriber._redis = redis

    local = _event_for("nobitex", "USDT_IRT", t_start_ms=9_000, t_end_ms=10_000)
    from mm.core.models.strategy import StrategyInputFrame

    input_frame = StrategyInputFrame(local=local, created_at_ms=10_000)
    original = joint_frame_for_input(input_frame)
    assert isinstance(original, JointStrategyFrame)
    topic = "strategy:joint:frames:nobitex:USDT_IRT:10000"
    queue = subscriber.subscribe_queue(topic)
    await asyncio.wait_for(redis.xread_started.wait(), timeout=1.0)

    await publisher.publish(topic, original)
    delivered = await asyncio.wait_for(queue.get(), timeout=1.0)

    assert isinstance(delivered, JointStrategyFrame)
    assert delivered.local_market.exchange == original.local_market.exchange
    assert delivered.local_market.symbol == original.local_market.symbol
    assert delivered.input_frame is not None
    assert delivered.ledger is not None

    payload = joint_strategy_frame_event(original)
    assert payload["payload_schema"] == "mm.joint_strategy_frame"
    assert isinstance(payload["input_frame"], dict)

    await publisher.close()
    await subscriber.close()
