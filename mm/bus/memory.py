from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from mm.bus.base import AbstractEventBus


class InMemoryBus(AbstractEventBus):
    """Development event bus: one asyncio.Queue per subscriber."""

    def __init__(self, max_queue_size: int = 0):
        self._max_queue_size = max_queue_size
        self._subscribers: dict[str, list[asyncio.Queue[Any]]] = defaultdict(list)
        self._closed = False

    async def publish(self, topic: str, event: Any) -> None:
        if self._closed:
            return
        subscribers = list(self._subscribers.get(topic, ()))
        for queue in subscribers:
            await queue.put(event)

    async def collect(self, topic: str, limit: int) -> list[Any]:
        out = []
        async for event in self.subscribe(topic):
            out.append(event)
            if len(out) >= limit:
                break
        return out

    def subscribe_queue(self, topic: str) -> asyncio.Queue[Any]:
        """Register a subscriber queue explicitly for tests and in-process actors."""
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers[topic].append(queue)
        return queue

    async def _unsubscribe(self, topic: str, queue: asyncio.Queue[Any]) -> None:
        subscribers = self._subscribers.get(topic)
        if subscribers and queue in subscribers:
            subscribers.remove(queue)

    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers[topic].append(queue)
        try:
            while not self._closed:
                yield await queue.get()
        finally:
            await self._unsubscribe(topic, queue)

    async def close(self) -> None:
        self._closed = True
        self._subscribers.clear()
