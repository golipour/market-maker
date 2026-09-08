from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class AbstractEventBus(ABC):
    @abstractmethod
    async def publish(self, topic: str, event: Any) -> None:
        ...

    @abstractmethod
    def subscribe(self, topic: str) -> AsyncIterator[Any]:
        ...

    def subscribe_queue(self, topic: str) -> asyncio.Queue[Any]:
        """Register a queue backed by this bus.

        In-process actors use queues today. Durable buses can support the same
        runtime contract by forwarding their async iterator into a local queue.
        """

        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def _forward() -> None:
            async for event in self.subscribe(topic):
                await queue.put(event)

        asyncio.create_task(_forward())
        return queue

    async def unsubscribe_queue(self, queue: asyncio.Queue[Any] | None) -> None:
        """Remove a queue registered by ``subscribe_queue``.

        ``subscribe()`` already unregisters in ``finally``. Queue subscribers
        must call this on stop or the queue stays in the fan-out list and can
        stall every publisher under ``block`` backpressure.
        """

        return None

    async def close(self) -> None:
        """Release bus resources. In-memory implementations may no-op."""

        return None

