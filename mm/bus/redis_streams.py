from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.events import decode_envelope, encode_envelope, envelope_event


class RedisStreamsBus(AbstractEventBus):
    """Redis Streams implementation of the runtime bus.

    Each logical topic maps to one Redis stream. Events are stored as versioned
    envelopes so the same stream can be consumed by persistence, replay, and UI
    daemons.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        stream_prefix: str = "mm",
        maxlen: int | None = 100_000,
        read_block_ms: int = 1_000,
        run_id: str = "",
    ):
        self.url = url
        self.stream_prefix = stream_prefix.strip(":") or "mm"
        self.maxlen = maxlen
        self.read_block_ms = max(1, int(read_block_ms))
        self.run_id = run_id
        self._redis: Any | None = None
        self._closed = False
        self._forward_tasks: list[asyncio.Task] = []

    async def publish(self, topic: str, event: Any) -> None:
        r = await self._client()
        envelope = envelope_event(topic, event, run_id=self.run_id, source="redis-streams-bus")
        kwargs: dict[str, Any] = {}
        if self.maxlen is not None and self.maxlen > 0:
            kwargs = {"maxlen": int(self.maxlen), "approximate": True}
        await r.xadd(self._stream(topic), {"data": encode_envelope(envelope)}, **kwargs)

    async def collect(self, topic: str, limit: int) -> list[Any]:
        out = []
        async for event in self.subscribe(topic):
            out.append(event)
            if len(out) >= limit:
                break
        return out

    def subscribe_queue(self, topic: str) -> asyncio.Queue[Any]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        task = asyncio.create_task(self._forward(topic, queue))
        self._forward_tasks.append(task)
        return queue

    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        r = await self._client()
        stream = self._stream(topic)
        last_id = "$"
        while not self._closed:
            rows = await r.xread({stream: last_id}, block=self.read_block_ms, count=100)
            for _, messages in rows:
                for message_id, fields in messages:
                    last_id = message_id
                    raw = fields.get("data") if isinstance(fields, dict) else None
                    if raw is None:
                        continue
                    yield decode_envelope(raw).payload

    async def subscribe_envelopes(self, topic: str, *, start_id: str = "0-0") -> AsyncIterator[Any]:
        """Read envelopes from a topic, useful for replay and persistence daemons."""

        r = await self._client()
        stream = self._stream(topic)
        last_id = start_id
        while not self._closed:
            rows = await r.xread({stream: last_id}, block=self.read_block_ms, count=100)
            for _, messages in rows:
                for message_id, fields in messages:
                    last_id = message_id
                    raw = fields.get("data") if isinstance(fields, dict) else None
                    if raw is not None:
                        yield decode_envelope(raw)

    async def close(self) -> None:
        self._closed = True
        for task in self._forward_tasks:
            task.cancel()
        if self._forward_tasks:
            await asyncio.gather(*self._forward_tasks, return_exceptions=True)
        self._forward_tasks.clear()
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def _forward(self, topic: str, queue: asyncio.Queue[Any]) -> None:
        async for event in self.subscribe(topic):
            await queue.put(event)

    async def _client(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self.url, decode_responses=True)
        return self._redis

    def _stream(self, topic: str) -> str:
        normalized = topic.replace(":", ".").strip(".")
        return f"{self.stream_prefix}.{normalized}"
