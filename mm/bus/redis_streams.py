from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.bus.delivery import (
    BusBackpressurePolicy,
    TrackedSubscriberQueue,
    create_subscriber_queue,
    deliver_to_subscribers,
)
from mm.core.events import decode_envelope, encode_envelope, envelope_event
from mm.core.payload_schema import (
    PRIVATE_STATE_SNAPSHOT_TYPE,
    market_event_event,
    private_state_snapshot_event,
)
from mm.core.models.private import Balance, PrivateStateSnapshot
from mm.observability import counter_inc, gauge
from mm.observability.event_payloads import observe_envelope, observe_runtime_event
from mm.observability.schema import (
    MM_BUS_ENVELOPE_BYTES,
    MM_BUS_ENVELOPE_DROPPED_TOTAL,
    MM_BUS_EVENTS_DROPPED_TOTAL,
)


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
        envelope_size_max_bytes: int | None = 64 * 1024,
        read_block_ms: int = 1_000,
        run_id: str = "",
        max_queue_size: int = 1024,
        backpressure_policy: BusBackpressurePolicy = "block",
        max_connections: int = 500,
    ):
        self.url = url
        self.stream_prefix = stream_prefix.strip(":") or "mm"
        self.maxlen = maxlen
        self.envelope_size_max_bytes = (
            int(envelope_size_max_bytes) if envelope_size_max_bytes else None
        )
        self.read_block_ms = max(1, int(read_block_ms))
        self.run_id = run_id
        self._max_queue_size = int(max_queue_size)
        self._backpressure_policy = backpressure_policy
        self.max_connections = max(1, int(max_connections))
        self._instance_id = uuid.uuid4().hex
        self._redis: Any | None = None
        self._closed = False
        self._forward_tasks: set[asyncio.Task] = set()
        self._remote_tasks_by_topic: dict[str, asyncio.Task] = {}
        self._subscribers: dict[str, list[TrackedSubscriberQueue]] = defaultdict(list)
        self._subscriber_ids = itertools.count(1)
        self._last_error_log_monotonic: dict[tuple[str, str], float] = {}

    async def publish(self, topic: str, event: Any) -> None:
        observe_runtime_event(topic, event, stage="bus_publish")
        await deliver_to_subscribers(
            list(self._subscribers.get(topic, ())),
            event,
            max_queue_size=self._max_queue_size,
            policy=self._backpressure_policy,
        )
        stream_event = _compact_stream_event(topic, event)
        if stream_event is None:
            return
        envelope = envelope_event(topic, stream_event, run_id=self.run_id, source=self._source)
        observe_envelope(envelope, stage="redis_stream")
        encoded = encode_envelope(envelope)
        encoded_size = len(encoded.encode("utf-8") if isinstance(encoded, str) else encoded)
        gauge(MM_BUS_ENVELOPE_BYTES, float(encoded_size), labels={"topic": topic})
        if (
            self.envelope_size_max_bytes is not None
            and self.envelope_size_max_bytes > 0
            and encoded_size > self.envelope_size_max_bytes
        ):
            counter_inc(
                MM_BUS_ENVELOPE_DROPPED_TOTAL,
                labels={"topic": topic, "reason": "size_cap"},
            )
            counter_inc(
                MM_BUS_EVENTS_DROPPED_TOTAL,
                labels={"topic": topic, "reason": "size_cap"},
            )
            self._log_stream_error(
                "redis_stream_envelope_dropped",
                topic,
                ValueError(
                    f"envelope size {encoded_size} > cap {self.envelope_size_max_bytes}"
                ),
            )
            return
        kwargs: dict[str, Any] = {}
        if self.maxlen is not None and self.maxlen > 0:
            kwargs = {"maxlen": int(self.maxlen), "approximate": True}
        try:
            r = await self._client()
            await r.xadd(self._stream(topic), {"data": encoded}, **kwargs)
        except Exception as exc:
            if _is_redis_connection_error(exc):
                await self._reset_client()
            self._log_stream_error("redis_stream_publish_failed", topic, exc)

    async def collect(self, topic: str, limit: int) -> list[Any]:
        out = []
        async for event in self.subscribe(topic):
            out.append(event)
            if len(out) >= limit:
                break
        return out

    def subscribe_queue(self, topic: str) -> TrackedSubscriberQueue:
        queue = self._subscribe_local_queue(topic)
        self._ensure_remote_subscription(topic)
        return queue

    def _subscribe_local_queue(self, topic: str) -> TrackedSubscriberQueue:
        queue = create_subscriber_queue(
            topic,
            f"subscriber-{next(self._subscriber_ids)}",
            max_queue_size=self._max_queue_size,
        )
        self._subscribers[topic].append(queue)
        return queue

    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        queue = self._subscribe_local_queue(topic)
        self._ensure_remote_subscription(topic)
        try:
            while not self._closed:
                yield await queue.get()
        finally:
            await self._remove_subscriber(topic, queue)

    async def latest_event(
        self,
        topic: str,
        *,
        event_type: str | None = None,
        limit: int = 100,
    ) -> Any | None:
        """Return the newest matching stream payload, or None if the stream is empty."""

        try:
            r = await self._client()
        except Exception:
            return None
        xrevrange = getattr(r, "xrevrange", None)
        if not callable(xrevrange):
            return None
        try:
            rows = await xrevrange(
                self._stream(topic),
                max="+",
                min="-",
                count=max(1, int(limit)),
            )
        except Exception as exc:
            if _is_redis_connection_error(exc):
                await self._reset_client()
            self._log_stream_error("redis_stream_latest_failed", topic, exc)
            return None
        for _message_id, fields in rows or ():
            raw = fields.get("data") if isinstance(fields, dict) else None
            if raw is None:
                continue
            try:
                envelope = decode_envelope(raw)
            except Exception:
                continue
            event = _parse_remote_stream_payload(topic, envelope.payload)
            if event_type is None:
                return event
            actual_type = ""
            if isinstance(event, dict):
                actual_type = str(event.get("type", "")).strip()
            if not actual_type:
                actual_type = str(getattr(envelope, "event_type", "")).strip()
            if actual_type == event_type:
                return event
        return None

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
        self._remote_tasks_by_topic.clear()
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

    def _ensure_remote_subscription(self, topic: str) -> None:
        task = self._remote_tasks_by_topic.get(topic)
        if task is not None and not task.done():
            return
        task = asyncio.create_task(self._forward_remote_stream(topic))
        self._remote_tasks_by_topic[topic] = task
        self._track_forward_task(task)

    async def unsubscribe_queue(self, queue: asyncio.Queue[Any] | None) -> None:
        if queue is None:
            return
        topic = getattr(queue, "topic", None)
        if topic:
            await self._remove_subscriber(str(topic), queue)
            return
        for topic_name, subscribers in list(self._subscribers.items()):
            if queue in subscribers:
                await self._remove_subscriber(topic_name, queue)
                return

    async def _remove_subscriber(self, topic: str, queue: asyncio.Queue[Any]) -> None:
        subscribers = self._subscribers.get(topic)
        if subscribers and queue in subscribers:
            subscribers.remove(queue)
        if subscribers:
            return
        self._subscribers.pop(topic, None)
        task = self._remote_tasks_by_topic.pop(topic, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _forward_remote_stream(self, topic: str) -> None:
        while not self._closed:
            try:
                async for envelope in self.subscribe_envelopes(topic, start_id="$"):
                    if getattr(envelope, "source", "") == self._source:
                        continue
                    event = _parse_remote_stream_payload(topic, envelope.payload)
                    await deliver_to_subscribers(
                        list(self._subscribers.get(topic, ())),
                        event,
                        max_queue_size=self._max_queue_size,
                        policy=self._backpressure_policy,
                    )
                if not self._closed:
                    await asyncio.sleep(max(0.1, self.read_block_ms / 1000.0))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if _is_redis_connection_error(exc):
                    await self._reset_client()
                self._log_stream_error("redis_stream_subscribe_failed", topic, exc)
                await asyncio.sleep(_redis_retry_sleep_seconds(exc, read_block_ms=self.read_block_ms))

    async def _reset_client(self) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.aclose()
        except Exception:
            pass
        self._redis = None

    async def _client(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self.url,
                decode_responses=True,
                max_connections=self.max_connections,
                socket_connect_timeout=5.0,
                socket_timeout=5.0,
                retry_on_timeout=True,
                health_check_interval=30,
            )
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


def _is_redis_connection_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, ConnectionRefusedError, ConnectionResetError, TimeoutError, OSError)):
        return True
    try:
        from redis.exceptions import BusyLoadingError
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import TimeoutError as RedisTimeoutError
    except Exception:
        return False
    return isinstance(exc, (RedisConnectionError, RedisTimeoutError, BusyLoadingError))


def _redis_retry_sleep_seconds(exc: Exception, *, read_block_ms: int) -> float:
    try:
        from redis.exceptions import BusyLoadingError
    except Exception:
        return max(0.1, read_block_ms / 1000.0)
    if isinstance(exc, BusyLoadingError):
        return max(2.0, read_block_ms / 1000.0)
    return max(0.1, read_block_ms / 1000.0)


def _balance_is_zero(balance: Balance) -> bool:
    total = float(balance.total)
    if abs(total) > 1e-12:
        return False
    return float(balance.free) == 0.0 and float(balance.locked) == 0.0


def _parse_remote_stream_payload(topic: str, payload: Any) -> Any:
    if topic.startswith("strategy:joint:frames:"):
        from mm.core.payload_schema import joint_strategy_frame_from_payload

        parsed = joint_strategy_frame_from_payload(payload)
        if parsed is not None:
            return parsed
    if topic.startswith("market:events:"):
        from mm.core.payload_schema import market_event_from_payload

        parsed = market_event_from_payload(payload)
        if parsed is not None:
            return parsed
    return payload


def _compact_stream_event(topic: str, event: Any) -> Any | None:
    if topic.startswith("market:events:"):
        from mm.core.models.market import MarketEvent
        from mm.core.payload_schema import market_event_event, market_event_from_payload

        parsed = market_event_from_payload(event)
        if parsed is None and isinstance(event, MarketEvent):
            return market_event_event(event)
        if parsed is not None:
            return market_event_event(parsed)
        return None
    if topic.startswith("strategy:joint:frames:"):
        from mm.core.models.joint_strategy import JointStrategyFrame
        from mm.core.payload_schema import joint_strategy_frame_event, joint_strategy_frame_from_payload

        if isinstance(event, JointStrategyFrame):
            return joint_strategy_frame_event(event)
        parsed = joint_strategy_frame_from_payload(event)
        if parsed is not None:
            return joint_strategy_frame_event(parsed)
        return None
    if topic.endswith(":balances") and isinstance(event, Balance):
        if _balance_is_zero(event):
            return None
        return event
    if topic.endswith(":snapshot"):
        snapshot = event
        if isinstance(event, dict) and event.get("type") == PRIVATE_STATE_SNAPSHOT_TYPE:
            from mm.core.payload_schema import private_state_snapshot_from_payload

            snapshot = private_state_snapshot_from_payload(event)
        if isinstance(snapshot, PrivateStateSnapshot):
            filtered = PrivateStateSnapshot(
                exchange=snapshot.exchange,
                timestamp_ms=snapshot.timestamp_ms,
                balances={
                    asset: balance
                    for asset, balance in snapshot.balances.items()
                    if not _balance_is_zero(balance)
                },
                open_orders=snapshot.open_orders,
            )
            payload = private_state_snapshot_event(filtered)
            payload["balance_count"] = len(snapshot.balances)
            payload["nonzero_balance_count"] = len(filtered.balances)
            return payload
    return event
