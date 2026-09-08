from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

BusBackpressurePolicy = Literal["block", "drop_oldest", "drop_newest"]


class TrackedSubscriberQueue(asyncio.Queue[Any]):
    """Subscriber queue that tracks depth and oldest-item lag for observability."""

    __slots__ = ("topic", "subscriber_id", "_oldest_enqueued_ms")

    def __init__(
        self,
        topic: str,
        subscriber_id: str,
        *,
        maxsize: int = 0,
    ) -> None:
        super().__init__(maxsize=maxsize)
        self.topic = topic
        self.subscriber_id = subscriber_id
        self._oldest_enqueued_ms: int | None = None

    def lag_ms(self, now_ms: int | None = None) -> int:
        if self._oldest_enqueued_ms is None or self.empty():
            return 0
        current = int(time.time() * 1000) if now_ms is None else int(now_ms)
        return max(0, current - self._oldest_enqueued_ms)

    async def put(self, item: Any) -> None:
        if self.empty():
            self._oldest_enqueued_ms = int(time.time() * 1000)
        await super().put(item)

    def put_nowait(self, item: Any) -> None:
        if self.empty():
            self._oldest_enqueued_ms = int(time.time() * 1000)
        super().put_nowait(item)

    async def get(self) -> Any:
        item = await super().get()
        if self.empty():
            self._oldest_enqueued_ms = None
        return item

    def get_nowait(self) -> Any:
        item = super().get_nowait()
        if self.empty():
            self._oldest_enqueued_ms = None
        return item


def create_subscriber_queue(
    topic: str,
    subscriber_id: str,
    *,
    max_queue_size: int,
) -> TrackedSubscriberQueue:
    maxsize = max(0, int(max_queue_size))
    return TrackedSubscriberQueue(
        topic,
        subscriber_id,
        maxsize=0 if maxsize <= 0 else maxsize,
    )


async def deliver_to_subscriber(
    queue: TrackedSubscriberQueue,
    event: Any,
    *,
    max_queue_size: int,
    policy: BusBackpressurePolicy,
) -> str:
    """Deliver one event to a subscriber queue under the configured backpressure policy."""

    unlimited = int(max_queue_size) <= 0
    if unlimited:
        await queue.put(event)
        return "delivered"

    limit = int(max_queue_size)
    if queue.qsize() < limit:
        await queue.put(event)
        return "delivered"

    if policy == "block":
        started = time.perf_counter()
        await queue.put(event)
        _metrics().record_publish_blocked_ms(
            queue.topic,
            queue.subscriber_id,
            (time.perf_counter() - started) * 1000.0,
        )
        return "delivered"

    if policy == "drop_oldest":
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        await queue.put(event)
        _metrics().record_event_dropped(queue.topic, queue.subscriber_id, policy="drop_oldest")
        return "dropped_oldest"

    _metrics().record_event_dropped(queue.topic, queue.subscriber_id, policy="drop_newest")
    return "dropped_newest"


async def deliver_to_subscribers(
    queues: list[TrackedSubscriberQueue],
    event: Any,
    *,
    max_queue_size: int,
    policy: BusBackpressurePolicy,
) -> None:
    """Fan out one event without letting a full subscriber stall the others.

    A single ``block`` subscriber still waits for capacity. With two or more
    subscribers, a full queue is treated as ``drop_oldest`` so a leaked or
    stalled consumer cannot freeze the topic for live consumers.
    """

    if not queues:
        return
    isolate_full = policy == "block" and len(queues) > 1
    limit = int(max_queue_size)
    for queue in queues:
        queue_policy = policy
        if isolate_full and limit > 0 and queue.qsize() >= limit:
            queue_policy = "drop_oldest"
        await deliver_to_subscriber(
            queue,
            event,
            max_queue_size=max_queue_size,
            policy=queue_policy,
        )
    record_subscriber_queue_metrics(queues)


def record_subscriber_queue_metrics(queues: list[TrackedSubscriberQueue]) -> None:
    now_ms = int(time.time() * 1000)
    metrics = _metrics()
    for queue in queues:
        metrics.record_subscriber_queue(
            queue.topic,
            queue.subscriber_id,
            depth=queue.qsize(),
            lag_ms=queue.lag_ms(now_ms),
        )


def _metrics():
    from mm.observability import bus_metrics

    return bus_metrics
