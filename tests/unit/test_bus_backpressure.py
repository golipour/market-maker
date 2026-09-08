from __future__ import annotations

import asyncio

import pytest

from mm.bus.memory import InMemoryBus
from mm.observability.registry import REGISTRY
from mm.observability.schema import (
    MM_BUS_EVENTS_DROPPED_TOTAL,
    MM_BUS_SUBSCRIBER_LAG_MS,
    MM_BUS_SUBSCRIBER_QUEUE_DEPTH,
)


@pytest.mark.asyncio
async def test_bounded_bus_blocks_publish_until_subscriber_drains() -> None:
    bus = InMemoryBus(max_queue_size=1, backpressure_policy="block")
    queue = bus.subscribe_queue("topic:a")

    await bus.publish("topic:a", {"n": 1})
    assert queue.qsize() == 1

    publish_task = asyncio.create_task(bus.publish("topic:a", {"n": 2}))
    await asyncio.sleep(0.01)
    assert not publish_task.done()

    assert queue.get_nowait() == {"n": 1}
    await asyncio.wait_for(publish_task, timeout=1.0)
    assert queue.get_nowait() == {"n": 2}


@pytest.mark.asyncio
async def test_drop_newest_policy_skips_events_when_queue_is_full() -> None:
    bus = InMemoryBus(max_queue_size=1, backpressure_policy="drop_newest")
    queue = bus.subscribe_queue("topic:a")

    await bus.publish("topic:a", {"n": 1})
    await bus.publish("topic:a", {"n": 2})

    assert queue.qsize() == 1
    assert queue.get_nowait() == {"n": 1}


@pytest.mark.asyncio
async def test_drop_oldest_policy_replaces_queued_event() -> None:
    bus = InMemoryBus(max_queue_size=1, backpressure_policy="drop_oldest")
    queue = bus.subscribe_queue("topic:a")

    await bus.publish("topic:a", {"n": 1})
    await bus.publish("topic:a", {"n": 2})

    assert queue.qsize() == 1
    assert queue.get_nowait() == {"n": 2}


@pytest.mark.asyncio
async def test_bus_records_subscriber_queue_metrics() -> None:
    bus = InMemoryBus(max_queue_size=2, backpressure_policy="block")
    bus.subscribe_queue("topic:a")

    await bus.publish("topic:a", {"n": 1})

    text = REGISTRY.render()
    assert MM_BUS_SUBSCRIBER_QUEUE_DEPTH in text
    assert MM_BUS_SUBSCRIBER_LAG_MS in text


@pytest.mark.asyncio
async def test_drop_newest_increments_dropped_metric() -> None:
    bus = InMemoryBus(max_queue_size=1, backpressure_policy="drop_newest")
    bus.subscribe_queue("topic:a")

    await bus.publish("topic:a", {"n": 1})
    await bus.publish("topic:a", {"n": 2})

    assert MM_BUS_EVENTS_DROPPED_TOTAL in REGISTRY.render()


@pytest.mark.asyncio
async def test_unsubscribe_queue_removes_subscriber() -> None:
    bus = InMemoryBus(max_queue_size=1, backpressure_policy="block")
    queue = bus.subscribe_queue("topic:a")

    await bus.unsubscribe_queue(queue)
    await bus.publish("topic:a", {"n": 1})

    assert queue.empty()
    assert bus._subscribers.get("topic:a") in (None, [])


@pytest.mark.asyncio
async def test_full_subscriber_does_not_block_other_subscribers() -> None:
    bus = InMemoryBus(max_queue_size=1, backpressure_policy="block")
    stalled = bus.subscribe_queue("topic:a")
    live = bus.subscribe_queue("topic:a")

    await bus.publish("topic:a", {"n": 1})
    assert stalled.qsize() == 1
    assert live.get_nowait() == {"n": 1}

    await asyncio.wait_for(bus.publish("topic:a", {"n": 2}), timeout=1.0)
    assert live.get_nowait() == {"n": 2}
    assert stalled.get_nowait() == {"n": 2}
