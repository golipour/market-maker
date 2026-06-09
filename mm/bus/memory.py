from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.bus.delivery import (
    BusBackpressurePolicy,
    TrackedSubscriberQueue,
    create_subscriber_queue,
    deliver_to_subscriber,
    record_subscriber_queue_metrics,
)
from mm.observability.event_payloads import observe_runtime_event


class InMemoryBus(AbstractEventBus):
    """Development event bus: one tracked asyncio.Queue per subscriber."""

    def __init__(
        self,
        max_queue_size: int = 1024,
        *,
        backpressure_policy: BusBackpressurePolicy = "block",
    ):
        self._max_queue_size = int(max_queue_size)
        self._backpressure_policy = backpressure_policy
        self._subscribers: dict[str, list[TrackedSubscriberQueue]] = defaultdict(list)
        self._subscriber_ids = itertools.count(1)
        self._closed = False

    async def publish(self, topic: str, event: Any) -> None:
        if self._closed:
            return
        observe_runtime_event(topic, event, stage="bus_publish")
        subscribers = list(self._subscribers.get(topic, ()))
        for queue in subscribers:
            await deliver_to_subscriber(
                queue,
                event,
                max_queue_size=self._max_queue_size,
                policy=self._backpressure_policy,
            )
        if subscribers:
            record_subscriber_queue_metrics(subscribers)

    async def collect(self, topic: str, limit: int) -> list[Any]:
        out = []
        async for event in self.subscribe(topic):
            out.append(event)
            if len(out) >= limit:
                break
        return out

    def subscribe_queue(self, topic: str) -> TrackedSubscriberQueue:
        """Register a subscriber queue explicitly for tests and in-process actors."""
        queue = create_subscriber_queue(
            topic,
            f"subscriber-{next(self._subscriber_ids)}",
            max_queue_size=self._max_queue_size,
        )
        self._subscribers[topic].append(queue)
        return queue

    async def _unsubscribe(self, topic: str, queue: TrackedSubscriberQueue) -> None:
        subscribers = self._subscribers.get(topic)
        if subscribers and queue in subscribers:
            subscribers.remove(queue)

    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        queue = self.subscribe_queue(topic)
        try:
            while not self._closed:
                yield await queue.get()
        finally:
            await self._unsubscribe(topic, queue)

    async def close(self) -> None:
        self._closed = True
        self._subscribers.clear()
