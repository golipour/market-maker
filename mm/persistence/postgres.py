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
DROP INDEX IF EXISTS mm_events_run_topic_ts_idx;
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

CREATE TABLE IF NOT EXISTS mm_action_stage_facts (
    event_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL,
    actor_id TEXT,
    exchange TEXT,
    symbol TEXT,
    timestamp_ms BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    t_end_ms BIGINT,
    order_count INT,
    buy_count INT,
    sell_count INT,
    buy_qty DOUBLE PRECISION,
    sell_qty DOUBLE PRECISION,
    buy_notional DOUBLE PRECISION,
    sell_notional DOUBLE PRECISION,
    rejection_count INT,
    rejection_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    orders_compacted_count INT,
    rejections_compacted_count INT,
    orders_digest TEXT,
    rejections_digest TEXT,
    frame_id TEXT,
    decision_id TEXT,
    target_book_id TEXT,
    PRIMARY KEY (event_id, stage)
);
CREATE INDEX IF NOT EXISTS mm_action_stage_facts_actor_stage_ts_idx
ON mm_action_stage_facts (actor_id, stage, ts DESC);
CREATE INDEX IF NOT EXISTS mm_action_stage_facts_exchange_symbol_stage_ts_idx
ON mm_action_stage_facts (exchange, symbol, stage, ts DESC);

CREATE TABLE IF NOT EXISTS mm_execution_facts (
    event_id TEXT NOT NULL,
    row_key TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL,
    event_type TEXT NOT NULL,
    action TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    success BOOLEAN,
    state_kind TEXT,
    actor_id TEXT,
    exchange TEXT,
    symbol TEXT,
    timestamp_ms BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    client_order_id TEXT,
    exchange_order_id TEXT,
    side TEXT,
    price DOUBLE PRECISION,
    qty DOUBLE PRECISION,
    remaining_qty DOUBLE PRECISION,
    filled_qty DOUBLE PRECISION,
    fee DOUBLE PRECISION,
    order_type TEXT,
    status TEXT,
    error_type TEXT,
    error_message TEXT,
    latency_ms DOUBLE PRECISION,
    count_value INT,
    frame_id TEXT,
    decision_id TEXT,
    target_book_id TEXT,
    execution_plan_id TEXT,
    PRIMARY KEY (event_id, row_key)
);
CREATE INDEX IF NOT EXISTS mm_execution_facts_exchange_symbol_ts_idx
ON mm_execution_facts (exchange, symbol, ts DESC);
CREATE INDEX IF NOT EXISTS mm_execution_facts_order_ts_idx
ON mm_execution_facts (exchange_order_id, ts DESC) WHERE exchange_order_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS mm_execution_facts_action_lifecycle_ts_idx
ON mm_execution_facts (action, lifecycle, ts DESC);

CREATE TABLE IF NOT EXISTS mm_target_order_facts (
    event_id TEXT NOT NULL,
    row_key TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL,
    actor_id TEXT,
    mode TEXT,
    venue_role TEXT NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    policy TEXT,
    order_index INT NOT NULL,
    side TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    qty DOUBLE PRECISION NOT NULL,
    notional DOUBLE PRECISION NOT NULL,
    order_type TEXT,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    frame_id TEXT,
    decision_id TEXT,
    target_book_id TEXT,
    execution_plan_id TEXT,
    PRIMARY KEY (event_id, row_key)
);
CREATE INDEX IF NOT EXISTS mm_target_order_facts_actor_venue_ts_idx
ON mm_target_order_facts (actor_id, venue_role, exchange, symbol, ts DESC);
CREATE INDEX IF NOT EXISTS mm_target_order_facts_exchange_symbol_side_ts_idx
ON mm_target_order_facts (exchange, symbol, side, ts DESC);

CREATE TABLE IF NOT EXISTS mm_model_score_facts (
    event_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL,
    actor_id TEXT,
    exchange TEXT,
    symbol TEXT,
    timestamp_ms BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    t_end_ms BIGINT,
    frame_step INT,
    model_kind TEXT,
    updated BOOLEAN,
    skip_reason TEXT,
    log_likelihood DOUBLE PRECISION,
    nll DOUBLE PRECISION,
    standardized_residual DOUBLE PRECISION,
    observation_count INT,
    params_before_count INT,
    params_after_count INT,
    frame_id TEXT,
    decision_id TEXT,
    target_book_id TEXT,
    PRIMARY KEY (event_id, model_name)
);
CREATE INDEX IF NOT EXISTS mm_model_score_facts_actor_model_ts_idx
ON mm_model_score_facts (actor_id, model_name, ts DESC);
CREATE INDEX IF NOT EXISTS mm_model_score_facts_exchange_symbol_model_ts_idx
ON mm_model_score_facts (exchange, symbol, model_name, ts DESC);

CREATE TABLE IF NOT EXISTS mm_optimizer_facts (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL,
    actor_id TEXT,
    exchange TEXT,
    symbol TEXT,
    timestamp_ms BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    t_end_ms BIGINT,
    frame_step INT,
    optimizer_kind TEXT,
    controller_total_ms DOUBLE PRECISION,
    optimize_ms DOUBLE PRECISION,
    clean_ms DOUBLE PRECISION,
    model_score_ms DOUBLE PRECISION,
    model_update_ms DOUBLE PRECISION,
    discretize_ms DOUBLE PRECISION,
    risk_ms DOUBLE PRECISION,
    publish_ms DOUBLE PRECISION,
    execute_ms DOUBLE PRECISION,
    initial_value DOUBLE PRECISION,
    final_value DOUBLE PRECISION,
    best_value DOUBLE PRECISION,
    initial_loss DOUBLE PRECISION,
    final_loss DOUBLE PRECISION,
    best_loss DOUBLE PRECISION,
    best_minus_initial DOUBLE PRECISION,
    final_minus_initial DOUBLE PRECISION,
    best_expected_total_reward DOUBLE PRECISION,
    best_expected_total_return DOUBLE PRECISION,
    best_expected_next_reward DOUBLE PRECISION,
    best_main_reward_mean DOUBLE PRECISION,
    best_global_reward_mean DOUBLE PRECISION,
    gradient_issue_count INT,
    rollout_count INT,
    frame_id TEXT,
    decision_id TEXT,
    target_book_id TEXT
);
CREATE INDEX IF NOT EXISTS mm_optimizer_facts_actor_ts_idx
ON mm_optimizer_facts (actor_id, ts DESC);
CREATE INDEX IF NOT EXISTS mm_optimizer_facts_exchange_symbol_ts_idx
ON mm_optimizer_facts (exchange, symbol, ts DESC);

CREATE TABLE IF NOT EXISTS mm_optimizer_component_facts (
    event_id TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    component TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL,
    actor_id TEXT,
    exchange TEXT,
    symbol TEXT,
    timestamp_ms BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    t_end_ms BIGINT,
    frame_step INT,
    value DOUBLE PRECISION,
    contribution_fraction DOUBLE PRECISION,
    frame_id TEXT,
    decision_id TEXT,
    target_book_id TEXT,
    PRIMARY KEY (event_id, snapshot, component)
);
CREATE INDEX IF NOT EXISTS mm_optimizer_component_facts_actor_component_ts_idx
ON mm_optimizer_component_facts (actor_id, component, ts DESC);
CREATE INDEX IF NOT EXISTS mm_optimizer_component_facts_exchange_symbol_component_ts_idx
ON mm_optimizer_component_facts (exchange, symbol, component, ts DESC);

ALTER TABLE mm_action_stage_facts ADD COLUMN IF NOT EXISTS frame_id TEXT;
ALTER TABLE mm_action_stage_facts ADD COLUMN IF NOT EXISTS decision_id TEXT;
ALTER TABLE mm_action_stage_facts ADD COLUMN IF NOT EXISTS target_book_id TEXT;
ALTER TABLE mm_execution_facts ADD COLUMN IF NOT EXISTS frame_id TEXT;
ALTER TABLE mm_execution_facts ADD COLUMN IF NOT EXISTS decision_id TEXT;
ALTER TABLE mm_execution_facts ADD COLUMN IF NOT EXISTS target_book_id TEXT;
ALTER TABLE mm_execution_facts ADD COLUMN IF NOT EXISTS execution_plan_id TEXT;
ALTER TABLE mm_target_order_facts ADD COLUMN IF NOT EXISTS frame_id TEXT;
ALTER TABLE mm_target_order_facts ADD COLUMN IF NOT EXISTS decision_id TEXT;
ALTER TABLE mm_target_order_facts ADD COLUMN IF NOT EXISTS target_book_id TEXT;
ALTER TABLE mm_target_order_facts ADD COLUMN IF NOT EXISTS execution_plan_id TEXT;
ALTER TABLE mm_model_score_facts ADD COLUMN IF NOT EXISTS frame_id TEXT;
ALTER TABLE mm_model_score_facts ADD COLUMN IF NOT EXISTS decision_id TEXT;
ALTER TABLE mm_model_score_facts ADD COLUMN IF NOT EXISTS target_book_id TEXT;
ALTER TABLE mm_optimizer_facts ADD COLUMN IF NOT EXISTS frame_id TEXT;
ALTER TABLE mm_optimizer_facts ADD COLUMN IF NOT EXISTS decision_id TEXT;
ALTER TABLE mm_optimizer_facts ADD COLUMN IF NOT EXISTS target_book_id TEXT;
ALTER TABLE mm_optimizer_component_facts ADD COLUMN IF NOT EXISTS frame_id TEXT;
ALTER TABLE mm_optimizer_component_facts ADD COLUMN IF NOT EXISTS decision_id TEXT;
ALTER TABLE mm_optimizer_component_facts ADD COLUMN IF NOT EXISTS target_book_id TEXT;

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
                    action_stage_rows = [
                        row
                        for envelope in batch
                        for row in _action_stage_fact_rows(envelope)
                    ]
                    if action_stage_rows:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO mm_action_stage_facts (
                                event_id, stage, run_id, topic, actor_id, exchange, symbol,
                                timestamp_ms, ts, t_end_ms, order_count, buy_count, sell_count,
                                buy_qty, sell_qty, buy_notional, sell_notional, rejection_count,
                                rejection_counts, orders_compacted_count,
                                rejections_compacted_count, orders_digest, rejections_digest,
                                frame_id, decision_id, target_book_id
                            ) VALUES %s
                            ON CONFLICT (event_id, stage) DO NOTHING
                            """,
                            action_stage_rows,
                        )
                    execution_rows = [
                        row
                        for envelope in batch
                        for row in _execution_fact_rows(envelope)
                    ]
                    if execution_rows:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO mm_execution_facts (
                                event_id, row_key, run_id, topic, event_type, action,
                                lifecycle, success, state_kind, actor_id, exchange, symbol,
                                timestamp_ms, ts, client_order_id, exchange_order_id, side,
                                price, qty, remaining_qty, filled_qty, fee, order_type,
                                status, error_type, error_message, latency_ms, count_value,
                                frame_id, decision_id, target_book_id, execution_plan_id
                            ) VALUES %s
                            ON CONFLICT (event_id, row_key) DO NOTHING
                            """,
                            execution_rows,
                        )
                    target_order_rows = [
                        row
                        for envelope in batch
                        for row in _target_order_fact_rows(envelope)
                    ]
                    if target_order_rows:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO mm_target_order_facts (
                                event_id, row_key, run_id, topic, actor_id, mode,
                                venue_role, exchange, symbol, timestamp_ms, ts, policy,
                                order_index, side, price, qty, notional, order_type, tags,
                                frame_id, decision_id, target_book_id, execution_plan_id
                            ) VALUES %s
                            ON CONFLICT (event_id, row_key) DO NOTHING
                            """,
                            target_order_rows,
                        )
                    model_score_rows = [
                        row
                        for envelope in batch
                        for row in _model_score_fact_rows(envelope)
                    ]
                    if model_score_rows:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO mm_model_score_facts (
                                event_id, model_name, run_id, topic, actor_id, exchange,
                                symbol, timestamp_ms, ts, t_end_ms, frame_step,
                                model_kind, updated, skip_reason, log_likelihood, nll,
                                standardized_residual, observation_count,
                                params_before_count, params_after_count,
                                frame_id, decision_id, target_book_id
                            ) VALUES %s
                            ON CONFLICT (event_id, model_name) DO NOTHING
                            """,
                            model_score_rows,
                        )
                    optimizer_rows = [
                        row
                        for envelope in batch
                        if (row := _optimizer_fact_row(envelope)) is not None
                    ]
                    if optimizer_rows:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO mm_optimizer_facts (
                                event_id, run_id, topic, actor_id, exchange, symbol,
                                timestamp_ms, ts, t_end_ms, frame_step, optimizer_kind,
                                controller_total_ms, optimize_ms, clean_ms,
                                model_score_ms, model_update_ms, discretize_ms, risk_ms,
                                publish_ms, execute_ms, initial_value, final_value,
                                best_value, initial_loss, final_loss, best_loss,
                                best_minus_initial, final_minus_initial,
                                best_expected_total_reward, best_expected_total_return,
                                best_expected_next_reward, best_main_reward_mean,
                                best_global_reward_mean, gradient_issue_count, rollout_count,
                                frame_id, decision_id, target_book_id
                            ) VALUES %s
                            ON CONFLICT (event_id) DO NOTHING
                            """,
                            optimizer_rows,
                        )
                    optimizer_component_rows = [
                        row
                        for envelope in batch
                        for row in _optimizer_component_fact_rows(envelope)
                    ]
                    if optimizer_component_rows:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO mm_optimizer_component_facts (
                                event_id, snapshot, component, run_id, topic, actor_id,
                                exchange, symbol, timestamp_ms, ts, t_end_ms, frame_step,
                                value, contribution_fraction, frame_id, decision_id,
                                target_book_id
                            ) VALUES %s
                            ON CONFLICT (event_id, snapshot, component) DO NOTHING
                            """,
                            optimizer_component_rows,
                        )
                    latest_rows = [
                        (
                            _latest_key(envelope),
                            envelope.topic,
                            envelope.event_id,
                            envelope.run_id,
                            _ts(envelope.timestamp_ms),
                            psycopg2.extras.Json(to_jsonable(envelope.payload), dumps=_strict_json_dumps),
                            psycopg2.extras.Json({}, dumps=_strict_json_dumps),
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
        psycopg2.extras.Json({}, dumps=_strict_json_dumps),
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


def _action_stage_fact_rows(envelope: EventEnvelope) -> list[tuple[Any, ...]]:
    if envelope.event_type != "ControllerDiagnostics" or not isinstance(envelope.payload, dict):
        return []
    payload = to_jsonable(envelope.payload)
    if not isinstance(payload, dict):
        return []
    pipeline = payload.get("action_pipeline")
    if not isinstance(pipeline, dict):
        return []

    import psycopg2.extras

    rows: list[tuple[Any, ...]] = []
    timestamp_ms = envelope.timestamp_ms
    t_end_ms = _int_or_none(payload.get("t_end_ms"))
    for stage, stage_payload in pipeline.items():
        if not isinstance(stage_payload, dict):
            continue
        rejection_counts = _dict_or_empty(stage_payload.get("rejection_counts"))
        rejection_count = _sum_numeric_values(rejection_counts)
        rejections_compacted_count = _int_or_none(stage_payload.get("rejections_compacted_count"))
        if rejections_compacted_count is not None:
            rejection_count = rejections_compacted_count
        rows.append(
            (
                envelope.event_id,
                str(stage),
                envelope.run_id,
                envelope.topic,
                _nonempty_str(payload.get("actor_id") or envelope.actor_id) or None,
                _nonempty_str(payload.get("exchange") or envelope.exchange) or None,
                _nonempty_str(payload.get("symbol") or envelope.symbol) or None,
                timestamp_ms,
                _ts(timestamp_ms),
                t_end_ms,
                _int_or_none(stage_payload.get("order_count")),
                _int_or_none(stage_payload.get("buy_count")),
                _int_or_none(stage_payload.get("sell_count")),
                _float_or_none(stage_payload.get("buy_qty")),
                _float_or_none(stage_payload.get("sell_qty")),
                _float_or_none(stage_payload.get("buy_notional")),
                _float_or_none(stage_payload.get("sell_notional")),
                rejection_count,
                psycopg2.extras.Json(rejection_counts, dumps=_strict_json_dumps),
                _int_or_none(stage_payload.get("orders_compacted_count")),
                rejections_compacted_count,
                _optional_str(stage_payload.get("orders_digest")),
                _optional_str(stage_payload.get("rejections_digest")),
                _optional_str(payload.get("frame_id")),
                _optional_str(payload.get("decision_id")),
                _optional_str(payload.get("target_book_id")),
            )
        )
    return rows


def _execution_fact_rows(envelope: EventEnvelope) -> list[tuple[Any, ...]]:
    payload = to_jsonable(envelope.payload)
    if not isinstance(payload, dict):
        return []
    event_type = envelope.event_type
    topic = envelope.topic
    if not _is_execution_fact_event(topic, event_type):
        return []

    timestamp_ms = _int_or_none(payload.get("timestamp_ms")) or envelope.timestamp_ms
    exchange = _nonempty_str(payload.get("exchange") or envelope.exchange) or None
    actor_id = _nonempty_str(payload.get("actor_id") or envelope.actor_id) or None
    symbol = _nonempty_str(payload.get("symbol") or envelope.symbol) or None
    error_type = _optional_str(payload.get("error"))
    error_message = _optional_str(payload.get("message"))
    latency_ms = _float_or_none(payload.get("latency_ms"))
    lineage = _execution_lineage(payload)

    if event_type.startswith("ExecutionSync"):
        lifecycle = _execution_lifecycle(event_type)
        return [
            _execution_row(
                envelope,
                row_key="sync",
                action="sync",
                lifecycle=lifecycle,
                success=_success_for_lifecycle(lifecycle),
                state_kind=None,
                actor_id=actor_id,
                exchange=exchange,
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                error_type=error_type,
                error_message=error_message,
                latency_ms=latency_ms,
                count_value=_int_or_none(payload.get("target_count")),
                **lineage,
            )
        ]

    if event_type in {"PlaceOrdersRequested", "PlaceOrdersAccepted", "PlaceOrdersFailed"}:
        lifecycle = _order_lifecycle(event_type)
        orders = payload.get("orders")
        order_ids = payload.get("order_ids")
        rows: list[tuple[Any, ...]] = []
        if isinstance(orders, list) and orders:
            for index, order in enumerate(orders):
                if not isinstance(order, dict):
                    continue
                exchange_order_id = _order_id_at(order_ids, index) or _optional_str(order.get("exchange_order_id"))
                rows.append(
                    _execution_row(
                        envelope,
                        row_key=f"place:{index}:{exchange_order_id or ''}",
                        action="place",
                        lifecycle=lifecycle,
                        success=_success_for_lifecycle(lifecycle),
                        state_kind="requested" if lifecycle == "requested" else "pending" if lifecycle == "accepted" else None,
                        actor_id=actor_id,
                        exchange=_nonempty_str(order.get("exchange") or exchange) or None,
                        symbol=_nonempty_str(order.get("symbol") or symbol) or None,
                        timestamp_ms=timestamp_ms,
                        order=order,
                        exchange_order_id=exchange_order_id,
                        error_type=error_type,
                        error_message=error_message,
                        **lineage,
                    )
                )
        else:
            rows.append(
                _execution_row(
                    envelope,
                    row_key="place",
                    action="place",
                    lifecycle=lifecycle,
                    success=_success_for_lifecycle(lifecycle),
                    state_kind=None,
                    actor_id=actor_id,
                    exchange=exchange,
                    symbol=symbol,
                    timestamp_ms=timestamp_ms,
                    error_type=error_type,
                    error_message=error_message,
                    count_value=_int_or_none(payload.get("count")),
                    **lineage,
                )
            )
        return rows

    if event_type in {
        "CancelOrdersRequested",
        "CancelOrdersAccepted",
        "CancelOrdersFailed",
        "CancelActorOrdersRequested",
    }:
        lifecycle = _order_lifecycle(event_type)
        order_ids = payload.get("order_ids")
        ids = [str(item) for item in order_ids] if isinstance(order_ids, list) else []
        if not ids:
            ids = [""]
        return [
            _execution_row(
                envelope,
                row_key=f"cancel:{index}:{order_id}",
                action="cancel",
                lifecycle=lifecycle,
                success=_success_for_lifecycle(lifecycle),
                state_kind="requested" if lifecycle == "requested" else "pending" if lifecycle == "accepted" else None,
                actor_id=actor_id,
                exchange=exchange,
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                exchange_order_id=order_id or None,
                status="cancel_requested" if lifecycle == "requested" else None,
                error_type=error_type,
                error_message=error_message,
                count_value=_int_or_none(payload.get("count")),
                **lineage,
            )
            for index, order_id in enumerate(ids)
        ]

    if event_type == "OrderValidationRejected":
        return [
            _execution_row(
                envelope,
                row_key="validation:0",
                action="validation",
                lifecycle="rejected",
                success=False,
                state_kind=None,
                actor_id=actor_id,
                exchange=exchange,
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                order=payload,
                status=_optional_str(payload.get("reason")),
                error_type=_optional_str(payload.get("reason")),
                **lineage,
            )
        ]

    if event_type == "Fill":
        return [
            _execution_row(
                envelope,
                row_key=f"fill:{payload.get('exchange_order_id') or ''}",
                action="fill",
                lifecycle="filled",
                success=True,
                state_kind="confirmed",
                actor_id=actor_id,
                exchange=exchange,
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                order=payload,
                exchange_order_id=_optional_str(payload.get("exchange_order_id")),
                filled_qty=_float_or_none(payload.get("qty")),
                fee=_float_or_none(payload.get("fee")),
                status="filled",
                **lineage,
            )
        ]

    if event_type == "ExecutionSnapshot":
        return _execution_snapshot_rows(
            envelope,
            payload,
            timestamp_ms=timestamp_ms,
            exchange=exchange,
            lineage=lineage,
        )

    return []


def _target_order_fact_rows(envelope: EventEnvelope) -> list[tuple[Any, ...]]:
    if envelope.event_type != "JointTargetBook" or not isinstance(envelope.payload, dict):
        return []
    payload = to_jsonable(envelope.payload)
    if not isinstance(payload, dict):
        return []

    import psycopg2.extras

    timestamp_ms = _int_or_none(payload.get("timestamp_ms")) or envelope.timestamp_ms
    actor_id = _optional_str(payload.get("actor_id") or envelope.actor_id)
    mode = _optional_str(payload.get("mode"))
    rows: list[tuple[Any, ...]] = []
    for venue_role, book_key in (("local", "local"), ("global", "global")):
        book = payload.get(book_key)
        if not isinstance(book, dict):
            continue
        exchange = _nonempty_str(book.get("exchange"))
        symbol = _nonempty_str(book.get("symbol"))
        if not exchange or not symbol:
            continue
        book_actor_id = _optional_str(book.get("actor_id")) or actor_id
        policy = _optional_str(book.get("policy"))
        orders = book.get("orders")
        if not isinstance(orders, list):
            continue
        for index, order in enumerate(orders):
            if not isinstance(order, dict):
                continue
            side = _optional_str(order.get("side"))
            price = _float_or_none(order.get("price"))
            qty = _float_or_none(order.get("qty"))
            if side is None or price is None or qty is None:
                continue
            tags = order.get("tags")
            tag_list = list(tags) if isinstance(tags, list) else []
            rows.append(
                (
                    envelope.event_id,
                    f"{venue_role}:{index}",
                    envelope.run_id,
                    envelope.topic,
                    book_actor_id,
                    mode,
                    venue_role,
                    _nonempty_str(order.get("exchange") or exchange),
                    _nonempty_str(order.get("symbol") or symbol),
                    timestamp_ms,
                    _ts(timestamp_ms),
                    policy,
                    index,
                    side,
                    price,
                    qty,
                    price * qty,
                    _optional_str(order.get("order_type")),
                    psycopg2.extras.Json(tag_list, dumps=_strict_json_dumps),
                    _optional_str(payload.get("frame_id") or book.get("frame_id")),
                    _optional_str(payload.get("decision_id") or book.get("decision_id")),
                    _optional_str(payload.get("target_book_id") or book.get("target_book_id")),
                    _optional_str(book.get("execution_plan_id")),
                )
            )
    return rows


def _controller_diagnostics_payload(envelope: EventEnvelope) -> dict[str, Any] | None:
    if envelope.event_type != "ControllerDiagnostics" or not isinstance(envelope.payload, dict):
        return None
    payload = to_jsonable(envelope.payload)
    return payload if isinstance(payload, dict) else None


def _controller_fact_base(envelope: EventEnvelope, payload: dict[str, Any]) -> tuple[Any, ...]:
    timestamp_ms = _int_or_none(payload.get("created_at_ms")) or envelope.timestamp_ms
    return (
        envelope.run_id,
        envelope.topic,
        _optional_str(payload.get("actor_id") or envelope.actor_id),
        _optional_str(payload.get("exchange") or envelope.exchange),
        _optional_str(payload.get("symbol") or envelope.symbol),
        timestamp_ms,
        _ts(timestamp_ms),
        _int_or_none(payload.get("t_end_ms")),
        _int_or_none(payload.get("frame_step")),
    )


def _model_score_fact_rows(envelope: EventEnvelope) -> list[tuple[Any, ...]]:
    payload = _controller_diagnostics_payload(envelope)
    if payload is None:
        return []
    models = payload.get("models")
    if not isinstance(models, dict):
        return []
    base = _controller_fact_base(envelope, payload)
    rows: list[tuple[Any, ...]] = []
    for model_name, score in models.items():
        if not isinstance(score, dict):
            continue
        rows.append(
            (
                envelope.event_id,
                str(model_name),
                *base,
                _optional_str(score.get("kind")),
                _bool_or_none(score.get("updated")),
                _optional_str(score.get("skip_reason")),
                _float_or_none(score.get("log_likelihood")),
                _float_or_none(score.get("nll")),
                _float_or_none(score.get("standardized_residual")),
                _dict_len(score.get("observation")),
                _dict_len(score.get("params_before")),
                _dict_len(score.get("params_after")),
                _optional_str(payload.get("frame_id")),
                _optional_str(payload.get("decision_id")),
                _optional_str(payload.get("target_book_id")),
            )
        )
    return rows


def _optimizer_fact_row(envelope: EventEnvelope) -> tuple[Any, ...] | None:
    payload = _controller_diagnostics_payload(envelope)
    if payload is None:
        return None
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict):
        return None
    config = _dict_or_empty(optimizer.get("config"))
    timing = _dict_or_empty(optimizer.get("timing"))
    objective = _dict_or_empty(optimizer.get("objective"))
    gradients = _dict_or_empty(optimizer.get("gradients"))
    rollout = _dict_or_empty(optimizer.get("rollout"))
    initial = _objective_snapshot(objective, "initial")
    final = _objective_snapshot(objective, "final")
    best = _objective_snapshot(objective, "best")
    improvement = _dict_or_empty(objective.get("improvement"))
    initial_value = _objective_value(initial)
    final_value = _objective_value(final)
    best_value = _objective_value(best)
    return (
        envelope.event_id,
        *_controller_fact_base(envelope, payload),
        _optional_str(config.get("kind") or optimizer.get("kind")),
        _timing_float(timing, "controller_total_ms"),
        _timing_float(timing, "optimize_ms", "optimizer_ms", "total_ms"),
        _timing_float(timing, "clean_ms"),
        _timing_float(timing, "model_score_ms"),
        _timing_float(timing, "model_update_ms"),
        _timing_float(timing, "discretize_ms"),
        _timing_float(timing, "risk_ms"),
        _timing_float(timing, "publish_ms"),
        _timing_float(timing, "execute_ms"),
        initial_value,
        final_value,
        best_value,
        _float_or_none(initial.get("loss")),
        _float_or_none(final.get("loss")),
        _float_or_none(best.get("loss")),
        _coalesce_float(
            _float_or_none(improvement.get("best_minus_initial")),
            _float_diff(best_value, initial_value),
        ),
        _coalesce_float(
            _float_or_none(improvement.get("final_minus_initial")),
            _float_diff(final_value, initial_value),
        ),
        _float_or_none(best.get("expected_total_reward")),
        _float_or_none(best.get("expected_total_return")),
        _float_or_none(best.get("expected_next_reward")),
        _float_or_none(best.get("main_reward_mean")),
        _float_or_none(best.get("global_reward_mean")),
        _payload_count(gradients),
        _payload_count(rollout),
        _optional_str(payload.get("frame_id")),
        _optional_str(payload.get("decision_id")),
        _optional_str(payload.get("target_book_id")),
    )


def _optimizer_component_fact_rows(envelope: EventEnvelope) -> list[tuple[Any, ...]]:
    payload = _controller_diagnostics_payload(envelope)
    if payload is None:
        return []
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict):
        return []
    objective = _dict_or_empty(optimizer.get("objective"))
    base = _controller_fact_base(envelope, payload)
    rows: list[tuple[Any, ...]] = []
    for snapshot_name in ("initial", "final", "best"):
        snapshot = _objective_snapshot(objective, snapshot_name)
        components = _dict_or_empty(snapshot.get("components"))
        fractions = _dict_or_empty(snapshot.get("contribution_fractions"))
        for component, raw_value in components.items():
            value = _float_or_none(raw_value)
            fraction = _float_or_none(fractions.get(component))
            if value is None and fraction is None:
                continue
            rows.append(
                (
                    envelope.event_id,
                    snapshot_name,
                    str(component),
                    *base,
                    value,
                    fraction,
                    _optional_str(payload.get("frame_id")),
                    _optional_str(payload.get("decision_id")),
                    _optional_str(payload.get("target_book_id")),
                )
            )
    return rows


def _execution_snapshot_rows(
    envelope: EventEnvelope,
    payload: dict[str, Any],
    *,
    timestamp_ms: int,
    exchange: str | None,
    lineage: dict[str, str | None] | None = None,
) -> list[tuple[Any, ...]]:
    lineage = lineage or {}
    rows = [
        _execution_row(
            envelope,
            row_key="snapshot",
            action="snapshot",
            lifecycle="snapshot",
            success=True,
            state_kind="aggregate",
            actor_id=None,
            exchange=exchange,
            symbol=None,
            timestamp_ms=timestamp_ms,
            count_value=_int_or_none(payload.get("live_order_count")),
            **lineage,
        )
    ]
    live_orders = payload.get("live_orders")
    if isinstance(live_orders, dict):
        for order_id, open_order in live_orders.items():
            row = _open_order_execution_row(
                envelope,
                row_key=f"live:{order_id}",
                state_kind="confirmed",
                timestamp_ms=timestamp_ms,
                exchange=exchange,
                order_id=str(order_id),
                open_order=open_order,
                lineage=lineage,
            )
            if row is not None:
                rows.append(row)
    for key, state_kind in (("pending_places", "pending_place"), ("pending_cancels", "pending_cancel")):
        pending = payload.get(key)
        if isinstance(pending, dict):
            for order_id, change in pending.items():
                row = _pending_execution_row(
                    envelope,
                    row_key=f"{state_kind}:{order_id}",
                    state_kind=state_kind,
                    timestamp_ms=timestamp_ms,
                    exchange=exchange,
                    order_id=str(order_id),
                    change=change,
                    lineage=lineage,
                )
                if row is not None:
                    rows.append(row)
    return rows


def _open_order_execution_row(
    envelope: EventEnvelope,
    *,
    row_key: str,
    state_kind: str,
    timestamp_ms: int,
    exchange: str | None,
    order_id: str,
    open_order: Any,
    lineage: dict[str, str | None] | None = None,
) -> tuple[Any, ...] | None:
    if not isinstance(open_order, dict):
        return None
    order = open_order.get("order")
    return _execution_row(
        envelope,
        row_key=row_key,
        action="snapshot",
        lifecycle="open",
        success=True,
        state_kind=state_kind,
        actor_id=_optional_str(open_order.get("actor_id")),
        exchange=_nonempty_str((order or {}).get("exchange") if isinstance(order, dict) else exchange) or exchange,
        symbol=_nonempty_str((order or {}).get("symbol") if isinstance(order, dict) else None) or None,
        timestamp_ms=timestamp_ms,
        order=order if isinstance(order, dict) else None,
        exchange_order_id=order_id,
        remaining_qty=_float_or_none(open_order.get("remaining_qty")),
        filled_qty=_float_or_none(open_order.get("filled_qty")),
        status=_optional_str(open_order.get("status")),
        **(lineage or {}),
    )


def _pending_execution_row(
    envelope: EventEnvelope,
    *,
    row_key: str,
    state_kind: str,
    timestamp_ms: int,
    exchange: str | None,
    order_id: str,
    change: Any,
    lineage: dict[str, str | None] | None = None,
) -> tuple[Any, ...] | None:
    if not isinstance(change, dict):
        return None
    order = change.get("order")
    return _execution_row(
        envelope,
        row_key=row_key,
        action=_optional_str(change.get("operation")) or "pending",
        lifecycle="accepted",
        success=True,
        state_kind=state_kind,
        actor_id=_optional_str(change.get("actor_id")),
        exchange=_nonempty_str((order or {}).get("exchange") if isinstance(order, dict) else exchange) or exchange,
        symbol=_nonempty_str((order or {}).get("symbol") if isinstance(order, dict) else None) or None,
        timestamp_ms=timestamp_ms,
        order=order if isinstance(order, dict) else None,
        exchange_order_id=order_id,
        status=_optional_str(change.get("reason")) or state_kind,
        **(lineage or {}),
    )


def _execution_row(
    envelope: EventEnvelope,
    *,
    row_key: str,
    action: str,
    lifecycle: str,
    success: bool | None,
    state_kind: str | None,
    actor_id: str | None,
    exchange: str | None,
    symbol: str | None,
    timestamp_ms: int,
    order: dict[str, Any] | None = None,
    client_order_id: str | None = None,
    exchange_order_id: str | None = None,
    remaining_qty: float | None = None,
    filled_qty: float | None = None,
    fee: float | None = None,
    status: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    latency_ms: float | None = None,
    count_value: int | None = None,
    frame_id: str | None = None,
    decision_id: str | None = None,
    target_book_id: str | None = None,
    execution_plan_id: str | None = None,
) -> tuple[Any, ...]:
    order = order or {}
    return (
        envelope.event_id,
        row_key,
        envelope.run_id,
        envelope.topic,
        envelope.event_type,
        action,
        lifecycle,
        success,
        state_kind,
        actor_id,
        exchange,
        symbol,
        timestamp_ms,
        _ts(timestamp_ms),
        client_order_id or _optional_str(order.get("client_order_id")),
        exchange_order_id or _optional_str(order.get("exchange_order_id")),
        _optional_str(order.get("side")),
        _float_or_none(order.get("price")),
        _float_or_none(order.get("qty")),
        remaining_qty,
        filled_qty,
        fee,
        _optional_str(order.get("order_type")),
        status,
        error_type,
        error_message,
        latency_ms,
        count_value,
        frame_id,
        decision_id,
        target_book_id,
        execution_plan_id,
    )


def _is_execution_fact_event(topic: str, event_type: str) -> bool:
    return (
        topic.startswith("execution:orders:")
        or topic.startswith("execution:state:")
        or topic.startswith("private:state:") and event_type == "Fill"
        or event_type in {"Fill", "ExecutionSnapshot"}
    )


def _execution_lineage(payload: dict[str, Any]) -> dict[str, str | None]:
    return {
        "frame_id": _optional_str(payload.get("frame_id")),
        "decision_id": _optional_str(payload.get("decision_id")),
        "target_book_id": _optional_str(payload.get("target_book_id")),
        "execution_plan_id": _optional_str(payload.get("execution_plan_id")),
    }


def _execution_lifecycle(event_type: str) -> str:
    if event_type.endswith("Requested"):
        return "requested"
    if event_type.endswith("Completed"):
        return "completed"
    if event_type.endswith("Failed"):
        return "failed"
    return "event"


def _order_lifecycle(event_type: str) -> str:
    if event_type.endswith("Requested"):
        return "requested"
    if event_type.endswith("Accepted"):
        return "accepted"
    if event_type.endswith("Failed"):
        return "failed"
    return "event"


def _success_for_lifecycle(lifecycle: str) -> bool | None:
    if lifecycle in {"accepted", "completed", "filled"}:
        return True
    if lifecycle in {"failed", "rejected"}:
        return False
    return None


def _order_id_at(order_ids: Any, index: int) -> str | None:
    if not isinstance(order_ids, list) or index >= len(order_ids):
        return None
    return _optional_str(order_ids[index])


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


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _sum_numeric_values(value: dict[str, Any]) -> int:
    total = 0
    for item in value.values():
        numeric = _int_or_none(item)
        if numeric is not None:
            total += numeric
    return total


def _dict_len(value: Any) -> int:
    return len(value) if isinstance(value, dict) else 0


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes"}:
            return True
        if normalized in {"false", "f", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _objective_snapshot(objective: dict[str, Any], name: str) -> dict[str, Any]:
    snapshot = objective.get(name)
    return dict(snapshot) if isinstance(snapshot, dict) else {}


def _objective_value(snapshot: dict[str, Any]) -> float | None:
    for key in ("value", "objective_value"):
        value = _float_or_none(snapshot.get(key))
        if value is not None:
            return value
    return None


def _timing_float(timing: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(timing.get(key))
        if value is not None:
            return value
    return None


def _float_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _coalesce_float(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _payload_count(value: Any) -> int:
    if isinstance(value, dict):
        issues = value.get("issues")
        if isinstance(issues, list):
            return len(issues)
        for key in ("count", "sample_count", "n_samples", "rollout_count"):
            count = _int_or_none(value.get(key))
            if count is not None:
                return count
        return len(value)
    if isinstance(value, list):
        return len(value)
    return 0


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


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
    return (
        envelope.topic.startswith(("execution:state:", "private:state:"))
        or envelope.event_type.endswith("Snapshot")
        or envelope.event_type == "RuntimeControlState"
    )


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
