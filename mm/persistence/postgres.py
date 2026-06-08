from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.events import EventEnvelope, envelope_event, to_jsonable
from mm.observability import counter_inc, gauge, histogram_observe, log_event
from mm.observability import schema as obs_schema
from mm.observability.event_payloads import observe_envelope
from mm.persistence.compaction import compact_envelope_for_persistence

_DDL = """
CREATE TABLE IF NOT EXISTS mm_events (
    id BIGSERIAL PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    schema TEXT NOT NULL,
    version INT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL,
    event_type TEXT NOT NULL,
    exchange TEXT,
    symbol TEXT,
    actor_id TEXT,
    timestamp_ms BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    envelope JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS mm_events_run_topic_ts_idx ON mm_events (run_id, topic, ts DESC);
CREATE INDEX IF NOT EXISTS mm_events_exchange_symbol_ts_idx ON mm_events (exchange, symbol, ts DESC);
CREATE INDEX IF NOT EXISTS mm_events_topic_type_ts_idx ON mm_events (topic, event_type, ts DESC);
CREATE INDEX IF NOT EXISTS mm_events_actor_ts_idx ON mm_events (actor_id, ts DESC) WHERE actor_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS mm_latest_state (
    state_key TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    event_id TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    envelope JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS admin_audit_log (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    command_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '',
    command TEXT NOT NULL,
    status TEXT NOT NULL,
    payload JSONB NOT NULL
);
"""


def _strict_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class PostgresEventWriter:
    """Best-effort async writer for durable runtime event envelopes."""

    def __init__(
        self,
        *,
        bus: AbstractEventBus,
        topics: tuple[str, ...],
        database_url: str,
        batch_size: int = 100,
        flush_interval_ms: int = 1000,
        run_id: str = "",
        connect_timeout_s: int = 5,
    ):
        self.bus = bus
        self.topics = topics
        self.database_url = database_url
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_ms = max(50, int(flush_interval_ms))
        self.run_id = run_id
        self.connect_timeout_s = max(1, int(connect_timeout_s))
        self._queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=self.batch_size * 10)
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._lock = threading.Lock()
        self._ready = False
        self._conn: Any | None = None

    async def start(self) -> None:
        self._stopped.clear()
        self._tasks = [asyncio.create_task(self._consume(topic)) for topic in self.topics]
        self._tasks.append(asyncio.create_task(self._write_loop()))
        self._tasks.append(asyncio.create_task(self._ensure_ready_background()))
        log_event(
            "postgres_event_writer_started",
            {
                "topics": list(self.topics),
                "batch_size": self.batch_size,
                "ready": False,
                "startup": "non_blocking",
                "connect_timeout_s": self.connect_timeout_s,
            },
        )

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await asyncio.to_thread(self._close_connection)

    async def _consume(self, topic: str) -> None:
        try:
            async for event in self.bus.subscribe(topic):
                envelope = envelope_event(topic, event, run_id=self.run_id, source="postgres-writer")
                envelope = compact_envelope_for_persistence(envelope, sink="postgres")
                observe_envelope(envelope, stage="postgres_enqueue")
                await self._queue.put(envelope)
                gauge(
                    obs_schema.MM_PERSISTENCE_QUEUE_DEPTH,
                    float(self._queue.qsize()),
                    labels={"writer": "postgres"},
                    help="Persistence writer queue depth",
                )
        except asyncio.CancelledError:
            raise

    async def _write_loop(self) -> None:
        batch: list[EventEnvelope] = []
        deadline = self.flush_interval_ms / 1000.0
        try:
            while not self._stopped.is_set():
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=deadline)
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass
                if batch and (len(batch) >= self.batch_size or self._queue.empty()):
                    to_write = batch
                    batch = []
                    queue_depth = self._queue.qsize()
                    gauge(
                        obs_schema.MM_PERSISTENCE_QUEUE_DEPTH,
                        float(queue_depth),
                        labels={"writer": "postgres"},
                        help="Persistence writer queue depth",
                    )
                    await asyncio.to_thread(self._write_batch, to_write, queue_depth)
        except asyncio.CancelledError:
            if batch:
                await asyncio.to_thread(self._write_batch, batch, self._queue.qsize())
            raise

    def _write_batch(self, batch: list[EventEnvelope], queue_depth: int = 0) -> None:
        if not batch:
            return
        started_ns = time.monotonic_ns()
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            counter_inc("mm_persistence_errors_total", labels={"writer": "postgres", "error": "ImportError"})
            log_event(
                "postgres_event_writer_unavailable",
                {
                    "error": "ImportError",
                    "message": "psycopg2 is not installed; postgres persistence batch was not written",
                    "batch_size": len(batch),
                },
            )
            return
        with self._lock:
            try:
                conn = self._connection(psycopg2)
                with conn.cursor() as cur:
                    if not self._ready:
                        cur.execute(_DDL)
                        self._ready = True
                    rows = [_event_row(envelope) for envelope in batch]
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO mm_events (
                            event_id, schema, version, run_id, source, topic, event_type,
                            exchange, symbol, actor_id, timestamp_ms, ts, payload, envelope
                        ) VALUES %s
                        ON CONFLICT (event_id) DO NOTHING
                        """,
                        rows,
                    )
                    latest_rows = [
                        (
                            _latest_key(envelope),
                            envelope.topic,
                            envelope.event_id,
                            envelope.run_id,
                            _ts(envelope.timestamp_ms),
                            psycopg2.extras.Json(to_jsonable(envelope.payload), dumps=_strict_json_dumps),
                            psycopg2.extras.Json(envelope.as_dict(), dumps=_strict_json_dumps),
                        )
                        for envelope in batch
                        if _is_latest_state(envelope)
                    ]
                    latest_rows = _dedupe_latest_rows_by_state_key(latest_rows)
                    if latest_rows:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO mm_latest_state (
                                state_key, topic, event_id, run_id, updated_at, payload, envelope
                            ) VALUES %s
                            ON CONFLICT (state_key) DO UPDATE SET
                                topic = EXCLUDED.topic,
                                event_id = EXCLUDED.event_id,
                                run_id = EXCLUDED.run_id,
                                updated_at = EXCLUDED.updated_at,
                                payload = EXCLUDED.payload,
                                envelope = EXCLUDED.envelope
                            """,
                            latest_rows,
                        )
                conn.commit()
                latency_ms = _elapsed_ms(started_ns)
                counter_inc("mm_persistence_events_written_total", len(batch), labels={"writer": "postgres"})
                gauge("mm_persistence_batch_size", len(batch), labels={"writer": "postgres"})
                gauge(
                    obs_schema.MM_PERSISTENCE_QUEUE_DEPTH,
                    float(queue_depth),
                    labels={"writer": "postgres"},
                    help="Persistence writer queue depth",
                )
                histogram_observe(
                    obs_schema.MM_PERSISTENCE_WRITE_LATENCY_MS,
                    latency_ms,
                    labels={"writer": "postgres"},
                    help="Persistence batch write latency",
                )
                log_event(
                    obs_schema.EVENT_PERSISTENCE_BATCH_WRITTEN,
                    {
                        "writer": "postgres",
                        "batch_size": len(batch),
                        "queue_depth": int(queue_depth),
                        "latency_ms": latency_ms,
                    },
                )
            except Exception as exc:
                self._rollback_connection()
                self._drop_connection()
                counter_inc("mm_persistence_errors_total", labels={"writer": "postgres", "error": type(exc).__name__})
                log_event(
                    "postgres_event_writer_error",
                    {
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "batch_size": len(batch),
                        "latency_ms": _elapsed_ms(started_ns),
                    },
                )
                log_event(
                    obs_schema.EVENT_PERSISTENCE_BATCH_FAILED,
                    {
                        "writer": "postgres",
                        "batch_size": len(batch),
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "retryable": True,
                    },
                )
                return

    def _ensure_ready(self) -> bool:
        try:
            import psycopg2
        except ImportError:
            counter_inc("mm_persistence_errors_total", labels={"writer": "postgres", "error": "ImportError"})
            log_event(
                "postgres_event_writer_unavailable",
                {
                    "error": "ImportError",
                    "message": "psycopg2 is not installed; postgres persistence is disabled until dependency is installed",
                },
            )
            return False
        with self._lock:
            try:
                conn = self._connection(psycopg2)
                with conn.cursor() as cur:
                    cur.execute(_DDL)
                conn.commit()
                self._ready = True
                log_event("postgres_event_writer_ready", {"database_url_configured": True})
                return True
            except Exception as exc:
                self._rollback_connection()
                self._drop_connection()
                counter_inc("mm_persistence_errors_total", labels={"writer": "postgres", "error": type(exc).__name__})
                log_event("postgres_event_writer_error", {"error": type(exc).__name__, "message": str(exc)})
                return False

    async def _ensure_ready_background(self) -> None:
        ready = await asyncio.to_thread(self._ensure_ready)
        if not ready and not self._stopped.is_set():
            log_event(
                "postgres_event_writer_degraded",
                {
                    "message": "postgres persistence is not ready; runtime continues and write batches will retry",
                    "connect_timeout_s": self.connect_timeout_s,
                },
            )

    def _connection(self, psycopg2: Any) -> Any:
        if self._conn is None or getattr(self._conn, "closed", False):
            self._conn = psycopg2.connect(
                self.database_url,
                connect_timeout=self.connect_timeout_s,
            )
            self._ready = False
        return self._conn

    def _rollback_connection(self) -> None:
        conn = self._conn
        if conn is None or getattr(conn, "closed", False):
            return
        rollback = getattr(conn, "rollback", None)
        if callable(rollback):
            try:
                rollback()
            except Exception:
                pass

    def _drop_connection(self) -> None:
        self._close_connection_locked()
        self._ready = False

    def _close_connection(self) -> None:
        with self._lock:
            self._close_connection_locked()

    def _close_connection_locked(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is None or getattr(conn, "closed", False):
            return
        try:
            conn.close()
        except Exception:
            pass


def _event_row(envelope: EventEnvelope) -> tuple[Any, ...]:
    import psycopg2.extras

    return (
        envelope.event_id,
        envelope.schema,
        envelope.version,
        envelope.run_id,
        envelope.source,
        envelope.topic,
        envelope.event_type,
        envelope.exchange,
        envelope.symbol,
        envelope.actor_id,
        envelope.timestamp_ms,
        _ts(envelope.timestamp_ms),
        psycopg2.extras.Json(to_jsonable(envelope.payload), dumps=_strict_json_dumps),
        psycopg2.extras.Json(envelope.as_dict(), dumps=_strict_json_dumps),
    )


def _ts(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)


def _is_latest_state(envelope: EventEnvelope) -> bool:
    return envelope.topic.startswith(("execution:state:", "private:state:")) or envelope.event_type.endswith("Snapshot")


def _latest_key(envelope: EventEnvelope) -> str:
    if envelope.exchange:
        return f"{envelope.topic}:{envelope.exchange}"
    if envelope.actor_id:
        return f"{envelope.topic}:{envelope.actor_id}"
    return envelope.topic


def _dedupe_latest_rows_by_state_key(
    latest_rows: list[tuple[Any, ...]],
) -> list[tuple[Any, ...]]:
    """One row per ``state_key`` per INSERT.

    Postgres raises ``CardinalityViolation`` if ``ON CONFLICT (state_key) DO UPDATE``
    sees duplicate ``state_key`` values in the same VALUES list. Many ``Balance``
    events share ``topic`` + ``exchange`` and therefore the same ``state_key``.
    """

    by_key: dict[str, tuple[Any, ...]] = {}
    for row in latest_rows:
        state_key = str(row[0])
        by_key[state_key] = row
    return list(by_key.values())


def _elapsed_ms(started_ns: int) -> float:
    return (time.monotonic_ns() - started_ns) / 1_000_000
