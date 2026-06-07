from __future__ import annotations

import asyncio

import pytest

from mm.bus.redis_streams import RedisStreamsBus
from mm.core.models.private import Balance, PrivateStateSnapshot


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
