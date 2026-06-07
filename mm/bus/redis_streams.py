from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.events import decode_envelope, encode_envelope, envelope_event
from mm.core.models.private import Balance, PrivateStateSnapshot


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
        self._instance_id = uuid.uuid4().hex
        self._redis: Any | None = None
        self._closed = False
        self._forward_tasks: set[asyncio.Task] = set()
        self._subscribers: dict[str, list[asyncio.Queue[Any]]] = defaultdict(list)
        self._last_error_log_monotonic: dict[tuple[str, str], float] = {}

    async def publish(self, topic: str, event: Any) -> None:
        for queue in list(self._subscribers.get(topic, ())):
            await queue.put(event)
        stream_event = _compact_stream_event(topic, event)
        if stream_event is None:
            return
        envelope = envelope_event(topic, stream_event, run_id=self.run_id, source=self._source)
        kwargs: dict[str, Any] = {}
        if self.maxlen is not None and self.maxlen > 0:
            kwargs = {"maxlen": int(self.maxlen), "approximate": True}
        try:
            r = await self._client()
            await r.xadd(self._stream(topic), {"data": encode_envelope(envelope)}, **kwargs)
        except Exception as exc:
            self._log_stream_error("redis_stream_publish_failed", topic, exc)

    async def collect(self, topic: str, limit: int) -> list[Any]:
        out = []
        async for event in self.subscribe(topic):
            out.append(event)
            if len(out) >= limit:
                break
        return out

    def subscribe_queue(self, topic: str) -> asyncio.Queue[Any]:
        queue = self._subscribe_local_queue(topic)
        self._track_forward_task(asyncio.create_task(self._forward_remote_stream(topic, queue)))
        return queue

    def _subscribe_local_queue(self, topic: str) -> asyncio.Queue[Any]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._subscribers[topic].append(queue)
        return queue

    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        queue = self._subscribe_local_queue(topic)
        task = asyncio.create_task(self._forward_remote_stream(topic, queue))
        self._track_forward_task(task)
        try:
            while not self._closed:
                yield await queue.get()
        finally:
            subscribers = self._subscribers.get(topic)
            if subscribers and queue in subscribers:
                subscribers.remove(queue)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def subscribe_envelopes(self, topic: str, *, start_id: str = "0-0") -> AsyncIterator[Any]:
        """Read envelopes from a topic, useful for replay and persistence daemons."""

        r = await self._client()
        if not hasattr(r, "xread"):
            return
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
        for task in list(self._forward_tasks):
            task.cancel()
        if self._forward_tasks:
            await asyncio.gather(*self._forward_tasks, return_exceptions=True)
        self._forward_tasks.clear()
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        self._subscribers.clear()

    @property
    def _source(self) -> str:
        return f"redis-streams-bus:{self._instance_id}"

    def _track_forward_task(self, task: asyncio.Task) -> None:
        self._forward_tasks.add(task)
        task.add_done_callback(self._forward_tasks.discard)

    async def _forward_remote_stream(self, topic: str, queue: asyncio.Queue[Any]) -> None:
        while not self._closed:
            try:
                async for envelope in self.subscribe_envelopes(topic, start_id="$"):
                    if getattr(envelope, "source", "") == self._source:
                        continue
                    await queue.put(envelope.payload)
                if not self._closed:
                    await asyncio.sleep(max(0.1, self.read_block_ms / 1000.0))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_stream_error("redis_stream_subscribe_failed", topic, exc)
                await asyncio.sleep(max(0.1, self.read_block_ms / 1000.0))

    async def _client(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self.url, decode_responses=True)
        return self._redis

    def _stream(self, topic: str) -> str:
        normalized = topic.replace(":", ".").strip(".")
        return f"{self.stream_prefix}.{normalized}"

    def _log_stream_error(self, event: str, topic: str, exc: Exception) -> None:
        key = (event, topic)
        now = time.monotonic()
        previous = self._last_error_log_monotonic.get(key, 0.0)
        if now - previous < 10.0:
            return
        self._last_error_log_monotonic[key] = now
        try:
            from mm.observability.logs import log_event

            log_event(
                event,
                {
                    "topic": topic,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )
        except Exception:
            return


def _balance_is_zero(balance: Balance) -> bool:
    total = float(balance.total)
    if abs(total) > 1e-12:
        return False
    return float(balance.free) == 0.0 and float(balance.locked) == 0.0


def _compact_stream_event(topic: str, event: Any) -> Any | None:
    if topic.endswith(":balances") and isinstance(event, Balance):
        if _balance_is_zero(event):
            return None
        return event
    if topic.endswith(":snapshot") and isinstance(event, PrivateStateSnapshot):
        balances = {
            asset: balance
            for asset, balance in event.balances.items()
            if not _balance_is_zero(balance)
        }
        return {
            "type": "PrivateStateSnapshotSummary",
            "exchange": event.exchange,
            "timestamp_ms": event.timestamp_ms,
            "balance_count": len(event.balances),
            "nonzero_balance_count": len(balances),
            "balances": balances,
            "open_order_count": len(event.open_orders),
            "open_orders": event.open_orders,
        }
    return event
