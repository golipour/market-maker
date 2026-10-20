"""Durable order-attribution store.

Mappings are queued in memory and flushed off the execution loop. Open rows stay
until the order is released. Closed rows stay until the retention window passes,
which is long enough for a late fill or a reconciliation snapshot to resolve.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Protocol

from mm.execution.order_attribution import (
    AttributionRecord,
    OrderAttributionRegistry,
    read_legacy_attribution,
)

DEFAULT_ORDER_ATTRIBUTION_RETENTION_HOURS = 24
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS mm_order_attribution (
        exchange TEXT NOT NULL,
        exchange_order_id TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        client_order_id TEXT NOT NULL DEFAULT '',
        wire_client_order_id TEXT NOT NULL DEFAULT '',
        registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        closed_at TIMESTAMPTZ,
        PRIMARY KEY (exchange, exchange_order_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS mm_order_attribution_closed_at_idx
    ON mm_order_attribution (exchange, closed_at)
    WHERE closed_at IS NOT NULL
    """,
)

_UPSERT_SQL = """
INSERT INTO mm_order_attribution (
    exchange,
    exchange_order_id,
    actor_id,
    client_order_id,
    wire_client_order_id,
    closed_at
) VALUES %s
ON CONFLICT (exchange, exchange_order_id) DO UPDATE SET
    actor_id = EXCLUDED.actor_id,
    client_order_id = CASE
        WHEN EXCLUDED.client_order_id <> '' THEN EXCLUDED.client_order_id
        ELSE mm_order_attribution.client_order_id
    END,
    wire_client_order_id = CASE
        WHEN EXCLUDED.wire_client_order_id <> '' THEN EXCLUDED.wire_client_order_id
        ELSE mm_order_attribution.wire_client_order_id
    END,
    closed_at = EXCLUDED.closed_at
"""


class AttributionDatabase(Protocol):
    def ensure_schema(self) -> None: ...

    def load(self, exchange: str, closed_after: datetime) -> Sequence[AttributionRecord]: ...

    def load_recent_placements(
        self,
        exchange: str,
        placed_after: datetime,
    ) -> Sequence[AttributionRecord]: ...

    def apply(self, records: Sequence[AttributionRecord]) -> None: ...

    def delete_expired(self, exchange: str, closed_before: datetime) -> Sequence[str]: ...


class MemoryAttributionDatabase:
    """Process-local stand-in used when ``MM_DATABASE_URL`` is unset."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], AttributionRecord] = {}
        self._lock = threading.Lock()

    def ensure_schema(self) -> None:
        return

    def load_recent_placements(
        self,
        exchange: str,
        placed_after: datetime,
    ) -> Sequence[AttributionRecord]:
        del exchange, placed_after
        return ()

    def load(self, exchange: str, closed_after: datetime) -> Sequence[AttributionRecord]:
        with self._lock:
            return [
                record
                for record in self.rows.values()
                if record.exchange == exchange
                and (record.closed_at is None or record.closed_at >= closed_after)
            ]

    def apply(self, records: Sequence[AttributionRecord]) -> None:
        with self._lock:
            for record in records:
                self.rows[(record.exchange, record.exchange_order_id)] = record

    def delete_expired(self, exchange: str, closed_before: datetime) -> Sequence[str]:
        removed: list[str] = []
        with self._lock:
            for key, record in list(self.rows.items()):
                if (
                    record.exchange == exchange
                    and record.closed_at is not None
                    and record.closed_at < closed_before
                ):
                    removed.append(record.exchange_order_id)
                    del self.rows[key]
        return removed


class PostgresAttributionDatabase:
    """Postgres table of one row per exchange order."""

    def __init__(self, database_url: str, *, connect_timeout_s: int = 5) -> None:
        self.database_url = database_url
        self.connect_timeout_s = max(1, int(connect_timeout_s))

    def ensure_schema(self) -> None:
        global _SCHEMA_READY
        if _SCHEMA_READY:
            return
        with _SCHEMA_LOCK:
            if _SCHEMA_READY:
                return
            conn = self._connection()
            try:
                with conn.cursor() as cur:
                    for statement in _DDL:
                        cur.execute(statement)
                conn.commit()
                _SCHEMA_READY = True
            finally:
                conn.close()

    def load(self, exchange: str, closed_after: datetime) -> Sequence[AttributionRecord]:
        conn = self._connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT exchange_order_id, actor_id, client_order_id, wire_client_order_id, closed_at
                    FROM mm_order_attribution
                    WHERE exchange = %s
                      AND (closed_at IS NULL OR closed_at >= %s)
                    """,
                    (exchange, closed_after),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            AttributionRecord(
                exchange=exchange,
                exchange_order_id=str(order_id),
                actor_id=str(actor_id),
                client_order_id=str(client_order_id or ""),
                wire_client_order_id=str(wire_client_order_id or ""),
                closed_at=closed_at,
            )
            for order_id, actor_id, client_order_id, wire_client_order_id, closed_at in rows
        ]

    def load_recent_placements(
        self,
        exchange: str,
        placed_after: datetime,
    ) -> Sequence[AttributionRecord]:
        """Best-effort seed from place accepts already stored in execution facts."""

        conn = self._connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '15s'")
                cur.execute(
                    """
                    SELECT DISTINCT ON (exchange_order_id)
                        exchange_order_id,
                        actor_id,
                        COALESCE(client_order_id, '')
                    FROM mm_execution_facts
                    WHERE exchange = %s
                      AND action = 'place'
                      AND lifecycle = 'accepted'
                      AND exchange_order_id IS NOT NULL
                      AND exchange_order_id <> ''
                      AND actor_id IS NOT NULL
                      AND actor_id <> ''
                      AND ts >= %s
                    ORDER BY exchange_order_id, ts DESC
                    """,
                    (exchange, placed_after),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            AttributionRecord(
                exchange=exchange,
                exchange_order_id=str(order_id),
                actor_id=str(actor_id),
                client_order_id=str(client_order_id or ""),
            )
            for order_id, actor_id, client_order_id in rows
        ]

    def apply(self, records: Sequence[AttributionRecord]) -> None:
        if not records:
            return
        values = [
            (
                record.exchange,
                record.exchange_order_id,
                record.actor_id,
                record.client_order_id,
                record.wire_client_order_id,
                record.closed_at,
            )
            for record in records
        ]
        conn = self._connection()
        try:
            with conn.cursor() as cur:
                import psycopg2.extras

                for start in range(0, len(values), 500):
                    psycopg2.extras.execute_values(
                        cur,
                        _UPSERT_SQL,
                        values[start : start + 500],
                    )
            conn.commit()
        finally:
            conn.close()

    def delete_expired(self, exchange: str, closed_before: datetime) -> Sequence[str]:
        conn = self._connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM mm_order_attribution
                    WHERE exchange = %s
                      AND closed_at IS NOT NULL
                      AND closed_at < %s
                    RETURNING exchange_order_id
                    """,
                    (exchange, closed_before),
                )
                rows = cur.fetchall()
            conn.commit()
        finally:
            conn.close()
        return [str(row[0]) for row in rows]

    def _connection(self):
        import psycopg2

        return psycopg2.connect(
            self.database_url,
            connect_timeout=self.connect_timeout_s,
        )


class OrderAttributionService:
    """Load, flush, and expire attribution mappings for one exchange."""

    def __init__(
        self,
        registry: OrderAttributionRegistry,
        database: AttributionDatabase,
        *,
        exchange: str,
        retention_hours: int | None = None,
        flush_interval_s: float = 1.0,
        retention_interval_s: float = 600.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._registry = registry
        self._database = database
        self.exchange = exchange.lower().strip()
        self.retention_hours = _retention_hours(retention_hours)
        self.flush_interval_s = max(0.05, float(flush_interval_s))
        self.retention_interval_s = max(self.flush_interval_s, float(retention_interval_s))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._pending: dict[tuple[str, str], AttributionRecord] = {}
        self._lock = threading.Lock()
        self._stopped = asyncio.Event()
        self._task: asyncio.Task | None = None
        registry.bind_sink(self)

    def record(self, record: AttributionRecord) -> None:
        """Queue one mapping change. This is the execution-loop sink."""

        if record.exchange != self.exchange or not record.exchange_order_id:
            return
        with self._lock:
            self._pending[(record.exchange, record.exchange_order_id)] = record

    async def start(self) -> None:
        self._stopped.clear()
        await self._restore()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(self.flush)

    async def _restore(self) -> None:
        try:
            db_records, legacy, aliases, db_ok = await asyncio.to_thread(self._fetch)
        except Exception as exc:
            _log_failure("order_attribution_load_failed", exc)
            legacy, aliases = read_legacy_attribution(self.exchange)
            db_records, db_ok = (), False
        self._registry.apply_loaded(db_records)
        self._registry.apply_client_aliases(self.exchange, aliases)
        if db_ok and db_records:
            _log_loaded(self.exchange, len(db_records), 0)
            return
        if db_ok and not legacy:
            try:
                legacy = await asyncio.to_thread(
                    self._database.load_recent_placements,
                    self.exchange,
                    self._cutoff(),
                )
            except Exception as exc:
                _log_failure("order_attribution_placement_seed_failed", exc)
                legacy = ()
        if not legacy:
            _log_loaded(self.exchange, len(db_records), 0)
            return
        now = self._clock()
        imported: list[AttributionRecord] = []
        for record in legacy:
            if record.exchange != self.exchange:
                continue
            if self._registry.has_order(record.exchange, record.exchange_order_id):
                continue
            imported.append(
                AttributionRecord(
                    exchange=record.exchange,
                    exchange_order_id=record.exchange_order_id,
                    actor_id=record.actor_id,
                    client_order_id=record.client_order_id,
                    wire_client_order_id=record.wire_client_order_id,
                    closed_at=now,
                )
            )
        if imported:
            self._registry.apply_loaded(imported)
            if db_ok:
                for record in imported:
                    self.record(record)
        _log_loaded(self.exchange, len(db_records), len(imported))

    def _fetch(
        self,
    ) -> tuple[Sequence[AttributionRecord], Sequence[AttributionRecord], Sequence[tuple[str, str]], bool]:
        legacy, aliases = read_legacy_attribution(self.exchange)
        self._database.ensure_schema()
        records = self._database.load(self.exchange, self._cutoff())
        return records, legacy, aliases, True

    def flush(self) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        if not pending:
            return
        try:
            self._database.apply(pending)
        except Exception as exc:
            with self._lock:
                for record in pending:
                    self._pending.setdefault((record.exchange, record.exchange_order_id), record)
            _log_failure("order_attribution_flush_failed", exc)

    def retain_expired(self) -> list[str]:
        cutoff, expired = self._delete_expired()
        self._registry.forget_closed(self.exchange, expired, older_than=cutoff)
        return expired

    def _delete_expired(self) -> tuple[datetime, list[str]]:
        cutoff = self._cutoff()
        expired = [str(order_id) for order_id in self._database.delete_expired(self.exchange, cutoff)]
        return cutoff, expired

    async def _run(self) -> None:
        next_retention = time.monotonic() + self.retention_interval_s
        try:
            while not self._stopped.is_set():
                await asyncio.sleep(self.flush_interval_s)
                await asyncio.to_thread(self.flush)
                if time.monotonic() < next_retention:
                    continue
                next_retention = time.monotonic() + self.retention_interval_s
                try:
                    cutoff, expired = await asyncio.to_thread(self._delete_expired)
                except Exception as exc:
                    _log_failure("order_attribution_retention_failed", exc)
                    continue
                self._registry.forget_closed(self.exchange, expired, older_than=cutoff)
        except asyncio.CancelledError:
            raise

    def _cutoff(self) -> datetime:
        return self._clock() - timedelta(hours=self.retention_hours)


def build_order_attribution_service(
    registry: OrderAttributionRegistry,
    *,
    exchange: str,
) -> OrderAttributionService:
    """Bind a registry to Postgres, or to memory when no database URL is set."""

    url = os.environ.get("MM_DATABASE_URL", "").strip()
    database: AttributionDatabase
    if url:
        database = PostgresAttributionDatabase(url)
    else:
        database = MemoryAttributionDatabase()
    return OrderAttributionService(registry, database, exchange=exchange)


def _retention_hours(explicit: int | None) -> int:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("retention_hours must be positive")
        return int(explicit)
    raw = os.environ.get("MM_ORDER_ATTRIBUTION_RETENTION_HOURS", "").strip()
    if not raw:
        return DEFAULT_ORDER_ATTRIBUTION_RETENTION_HOURS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_ORDER_ATTRIBUTION_RETENTION_HOURS
    if value <= 0:
        return DEFAULT_ORDER_ATTRIBUTION_RETENTION_HOURS
    return value


def _log_loaded(exchange: str, database_rows: int, legacy_rows: int) -> None:
    from mm.observability import log_event

    log_event(
        "order_attribution_loaded",
        {
            "exchange": exchange,
            "database_rows": database_rows,
            "legacy_rows": legacy_rows,
        },
    )


def _log_failure(event: str, exc: BaseException) -> None:
    from mm.observability import log_event

    log_event(
        event,
        {
            "error": type(exc).__name__,
            "message": str(exc),
        },
    )
