from __future__ import annotations

import pytest

from mm.bus.redis_streams import RedisStreamsBus
from mm.core.models.private import Balance, PrivateStateSnapshot


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
        balances={"BTC": Balance("wallex", "BTC", 1.0, 0.0, 1)},
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
    balance = Balance("wallex", "ZERO", 0.0, 0.0, 1)

    await bus.publish("private:state:wallex:balances", balance)

    assert await queue.get() is balance
    assert writes == []
