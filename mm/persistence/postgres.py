from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
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

CREATE TABLE IF NOT EXISTS mm_market_facts (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval_ms BIGINT,
    t_start_ms BIGINT,
    t_end_ms BIGINT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    orderbook_timestamp_ms BIGINT,
    best_bid DOUBLE PRECISION,
    best_bid_qty DOUBLE PRECISION,
    best_ask DOUBLE PRECISION,
    best_ask_qty DOUBLE PRECISION,
    mid DOUBLE PRECISION,
    spread DOUBLE PRECISION,
    microprice DOUBLE PRECISION,
    vwap DOUBLE PRECISION,
    buy_volume DOUBLE PRECISION,
    sell_volume DOUBLE PRECISION,
    total_trade_volume DOUBLE PRECISION,
    imbalance DOUBLE PRECISION,
    trade_count INT,
    bid_depth_levels INT,
    ask_depth_levels INT,
    orderbook_is_snapshot BOOLEAN
);
CREATE INDEX IF NOT EXISTS mm_market_facts_exchange_symbol_ts_idx
ON mm_market_facts (exchange, symbol, ts DESC);
CREATE INDEX IF NOT EXISTS mm_market_facts_run_exchange_symbol_ts_idx
ON mm_market_facts (run_id, exchange, symbol, ts DESC);

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
        retry_attempts: int = 2,
        retry_backoff_ms: int = 100,
        failed_batch_spool_dir: str | None = "persistence_spool/postgres",
    ):
        self.bus = bus
        self.topics = topics
        self.database_url = database_url
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_ms = max(50, int(flush_interval_ms))
        self.run_id = run_id
        self.connect_timeout_s = max(1, int(connect_timeout_s))
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_ms = max(0, int(retry_backoff_ms))
        self.failed_batch_spool_dir = (
            Path(failed_batch_spool_dir).expanduser()
            if failed_batch_spool_dir
            else None
        )
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
        except ImportError as exc:
            counter_inc("mm_persistence_errors_total", labels={"writer": "postgres", "error": "ImportError"})
            log_event(
                "postgres_event_writer_unavailable",
                {
                    "error": "ImportError",
                    "message": "psycopg2 is not installed; postgres persistence batch was not written",
                    "batch_size": len(batch),
                },
            )
            self._spill_failed_batch(batch, exc, started_ns)
            return
        attempts = self.retry_attempts + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                self._write_batch_once(psycopg2, batch, queue_depth, started_ns)
                return
            except Exception as exc:
                last_error = exc
                retryable = attempt < self.retry_attempts
                counter_inc("mm_persistence_errors_total", labels={"writer": "postgres", "error": type(exc).__name__})
                log_event(
                    "postgres_event_writer_error",
                    {
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "batch_size": len(batch),
                        "latency_ms": _elapsed_ms(started_ns),
                        "attempt": attempt + 1,
                        "max_attempts": attempts,
                        "retryable": retryable,
                    },
                )
                log_event(
                    obs_schema.EVENT_PERSISTENCE_BATCH_FAILED,
                    {
                        "writer": "postgres",
                        "batch_size": len(batch),
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "retryable": retryable,
                        "attempt": attempt + 1,
                        "max_attempts": attempts,
                    },
                )
                if retryable and self.retry_backoff_ms > 0:
                    time.sleep((self.retry_backoff_ms / 1000.0) * (2 ** attempt))
        if last_error is not None:
            self._spill_failed_batch(batch, last_error, started_ns)

    def _write_batch_once(
        self,
        psycopg2: Any,
        batch: list[EventEnvelope],
        queue_depth: int,
        started_ns: int,
    ) -> None:
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
                    market_rows = [
                        row
                        for envelope in batch
                        if (row := _market_fact_row(envelope)) is not None
                    ]
                    if market_rows:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO mm_market_facts (
                                event_id, run_id, topic, exchange, symbol, interval_ms,
                                t_start_ms, t_end_ms, timestamp_ms, ts, orderbook_timestamp_ms,
                                best_bid, best_bid_qty, best_ask, best_ask_qty, mid, spread,
                                microprice, vwap, buy_volume, sell_volume, total_trade_volume,
                                imbalance, trade_count, bid_depth_levels, ask_depth_levels,
                                orderbook_is_snapshot
                            ) VALUES %s
                            ON CONFLICT (event_id) DO NOTHING
                            """,
                            market_rows,
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
                raise

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

    def _spill_failed_batch(
        self,
        batch: list[EventEnvelope],
        error: Exception,
        started_ns: int,
    ) -> None:
        if self.failed_batch_spool_dir is None:
            return
        spilled_at_ms = int(time.time() * 1000)
        dt = datetime.fromtimestamp(spilled_at_ms / 1000.0, tz=timezone.utc)
        path = (
            self.failed_batch_spool_dir
            / f"event_date={dt.strftime('%Y-%m-%d')}"
            / f"hour={dt.strftime('%H')}"
            / f"postgres-failed-{spilled_at_ms}-{uuid.uuid4().hex}.jsonl"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for envelope in batch:
                    f.write(
                        _strict_json_dumps(
                            {
                                "schema": "mm.persistence.failed_batch",
                                "version": 1,
                                "writer": "postgres",
                                "spilled_at_ms": spilled_at_ms,
                                "error": type(error).__name__,
                                "message": str(error),
                                "envelope": envelope.as_dict(),
                            }
                        )
                        + "\n"
                    )
            log_event(
                "postgres_event_writer_batch_spooled",
                {
                    "writer": "postgres",
                    "batch_size": len(batch),
                    "path": str(path),
                    "error": type(error).__name__,
                    "latency_ms": _elapsed_ms(started_ns),
                },
            )
        except Exception as exc:
            counter_inc("mm_persistence_errors_total", labels={"writer": "postgres", "error": type(exc).__name__})
            log_event(
                "postgres_event_writer_spool_failed",
                {
                    "writer": "postgres",
                    "batch_size": len(batch),
                    "path": str(path),
                    "error": type(exc).__name__,
                    "message": str(exc),
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


def _market_fact_row(envelope: EventEnvelope) -> tuple[Any, ...] | None:
    if envelope.event_type != "MarketEvent" or not isinstance(envelope.payload, dict):
        return None
    payload = to_jsonable(envelope.payload)
    if not isinstance(payload, dict):
        return None
    exchange = _nonempty_str(payload.get("exchange") or envelope.exchange)
    symbol = _nonempty_str(payload.get("symbol") or envelope.symbol)
    t_end_ms = _int_or_none(payload.get("t_end_ms"))
    if not exchange or not symbol or t_end_ms is None:
        return None

    orderbook = payload.get("orderbook")
    bids = _levels(orderbook, "bids")
    asks = _levels(orderbook, "asks")
    best_bid = bids[0][0] if bids else None
    best_bid_qty = bids[0][1] if bids else None
    best_ask = asks[0][0] if asks else None
    best_ask_qty = asks[0][1] if asks else None
    mid = (
        (best_bid + best_ask) / 2.0
        if best_bid is not None and best_ask is not None
        else None
    )
    spread = (
        best_ask - best_bid
        if best_bid is not None and best_ask is not None
        else None
    )
    buy_volume = _float_or_none(payload.get("buy_volume"))
    sell_volume = _float_or_none(payload.get("sell_volume"))
    total_trade_volume = (
        (buy_volume or 0.0) + (sell_volume or 0.0)
        if buy_volume is not None or sell_volume is not None
        else None
    )
    timestamp_ms = _int_or_none(payload.get("t_end_ms")) or envelope.timestamp_ms
    return (
        envelope.event_id,
        envelope.run_id,
        envelope.topic,
        exchange,
        symbol,
        _int_or_none(payload.get("interval_ms")),
        _int_or_none(payload.get("t_start_ms")),
        t_end_ms,
        timestamp_ms,
        _ts(timestamp_ms),
        _orderbook_int(orderbook, "timestamp_ms"),
        best_bid,
        best_bid_qty,
        best_ask,
        best_ask_qty,
        mid,
        spread,
        _float_or_none(payload.get("microprice")),
        _float_or_none(payload.get("vwap")),
        buy_volume,
        sell_volume,
        total_trade_volume,
        _float_or_none(payload.get("imbalance")),
        _int_or_none(payload.get("trade_count")) or _list_len(payload.get("trades")),
        len(bids),
        len(asks),
        _orderbook_bool(orderbook, "is_snapshot"),
    )


def _nonempty_str(value: Any) -> str:
    out = str(value or "").strip()
    return out


def _levels(orderbook: Any, side: str) -> list[tuple[float, float]]:
    if not isinstance(orderbook, dict):
        return []
    raw = orderbook.get(side)
    if not isinstance(raw, list):
        return []
    out: list[tuple[float, float]] = []
    for level in raw:
        price = qty = None
        if isinstance(level, dict):
            price = _float_or_none(level.get("price"))
            qty = _float_or_none(level.get("qty"))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price = _float_or_none(level[0])
            qty = _float_or_none(level[1])
        if price is not None and qty is not None:
            out.append((price, qty))
    return out


def _orderbook_int(orderbook: Any, key: str) -> int | None:
    if not isinstance(orderbook, dict):
        return None
    return _int_or_none(orderbook.get(key))


def _orderbook_bool(orderbook: Any, key: str) -> bool | None:
    if not isinstance(orderbook, dict):
        return None
    value = orderbook.get(key)
    if value is None:
        return None
    return bool(value)


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


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
