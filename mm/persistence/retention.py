"""Retention and tiering policies for Postgres, Parquet, and persistence spool data."""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_EVENT_DATE_RE = re.compile(r"^event_date=(\d{4}-\d{2}-\d{2})$")

POSTGRES_EVENTS_TABLE = "mm_events"
POSTGRES_FACT_TABLES: tuple[str, ...] = (
    "mm_market_facts",
    "mm_action_stage_facts",
    "mm_execution_facts",
    "mm_target_order_facts",
    "mm_model_score_facts",
    "mm_optimizer_facts",
    "mm_optimizer_component_facts",
)
POSTGRES_LATEST_STATE_TABLE = "mm_latest_state"
POSTGRES_ADMIN_AUDIT_TABLE = "admin_audit_log"
POSTGRES_DELETE_BATCH = 50_000
POSTGRES_TS_INDEXES: tuple[tuple[str, str], ...] = (
    ("mm_events", "mm_events_ts_idx"),
    ("mm_market_facts", "mm_market_facts_ts_idx"),
    ("mm_action_stage_facts", "mm_action_stage_facts_ts_idx"),
    ("mm_execution_facts", "mm_execution_facts_ts_idx"),
    ("mm_target_order_facts", "mm_target_order_facts_ts_idx"),
    ("mm_model_score_facts", "mm_model_score_facts_ts_idx"),
    ("mm_optimizer_facts", "mm_optimizer_facts_ts_idx"),
    ("mm_optimizer_component_facts", "mm_optimizer_component_facts_ts_idx"),
    ("admin_audit_log", "admin_audit_log_ts_idx"),
)


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Age-based retention tiers for persisted runtime data."""

    events_days: int | None = 14
    facts_days: int | None = 90
    latest_state_days: int | None = None
    admin_audit_days: int | None = 365
    parquet_days: int | None = 90
    spool_archive_days: int | None = 7
    spool_failed_days: int | None = 30

    def as_dict(self) -> dict[str, int | None]:
        return {
            "events_days": self.events_days,
            "facts_days": self.facts_days,
            "latest_state_days": self.latest_state_days,
            "admin_audit_days": self.admin_audit_days,
            "parquet_days": self.parquet_days,
            "spool_archive_days": self.spool_archive_days,
            "spool_failed_days": self.spool_failed_days,
        }

    def __post_init__(self) -> None:
        for name, value in (
            ("events_days", self.events_days),
            ("facts_days", self.facts_days),
            ("latest_state_days", self.latest_state_days),
            ("admin_audit_days", self.admin_audit_days),
            ("parquet_days", self.parquet_days),
            ("spool_archive_days", self.spool_archive_days),
            ("spool_failed_days", self.spool_failed_days),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when set")


@dataclass(frozen=True, slots=True)
class PostgresTableRetentionResult:
    table: str
    tier: str
    retention_days: int
    rows_affected: int


@dataclass(frozen=True, slots=True)
class PostgresRetentionResult:
    tables: tuple[PostgresTableRetentionResult, ...] = ()

    @property
    def rows_affected(self) -> int:
        return sum(item.rows_affected for item in self.tables)


@dataclass(frozen=True, slots=True)
class ParquetRetentionResult:
    partitions_removed: int = 0
    files_removed: int = 0
    bytes_removed: int = 0


@dataclass(frozen=True, slots=True)
class SpoolRetentionResult:
    files_removed: int = 0
    bytes_removed: int = 0


@dataclass(frozen=True, slots=True)
class RetentionResult:
    postgres: PostgresRetentionResult = field(default_factory=PostgresRetentionResult)
    parquet: ParquetRetentionResult = field(default_factory=ParquetRetentionResult)
    spool: SpoolRetentionResult = field(default_factory=SpoolRetentionResult)
    dry_run: bool = False


def retention_policy_from_mapping(payload: Any) -> RetentionPolicy:
    if not isinstance(payload, dict):
        return RetentionPolicy()
    return RetentionPolicy(
        events_days=_optional_positive_int(payload.get("events_days"), default=14),
        facts_days=_optional_positive_int(payload.get("facts_days"), default=90),
        latest_state_days=_optional_positive_int(payload.get("latest_state_days")),
        admin_audit_days=_optional_positive_int(payload.get("admin_audit_days"), default=365),
        parquet_days=_optional_positive_int(payload.get("parquet_days"), default=90),
        spool_archive_days=_optional_positive_int(payload.get("spool_archive_days"), default=7),
        spool_failed_days=_optional_positive_int(payload.get("spool_failed_days"), default=30),
    )


def apply_retention(
    policy: RetentionPolicy,
    *,
    database_url: str | None = None,
    parquet_root: str | Path | None = None,
    spool_dir: str | Path = "persistence_spool/postgres",
    now: datetime | None = None,
    dry_run: bool = False,
    write_canary: bool = True,
) -> RetentionResult:
    """Apply configured retention tiers to available persistence stores."""

    current = now or datetime.now(timezone.utc)
    postgres = PostgresRetentionResult()
    canary_age_ms = 0
    if database_url:
        if not dry_run:
            ensure_postgres_retention_indexes(database_url)
        postgres = apply_postgres_retention(
            database_url,
            policy,
            now=current,
            dry_run=dry_run,
        )
        if write_canary and not dry_run:
            try:
                canary_age_ms = write_postgres_canary(database_url, now=current)
            except Exception as exc:  # noqa: BLE001
                # A canary failure is not fatal; surface it via the result.
                postgres = PostgresRetentionResult(
                    tables=postgres.tables,
                )
                try:
                    from mm.observability.logs import log_event

                    log_event(
                        "retention_canary_failed",
                        {"error": type(exc).__name__, "message": str(exc)},
                    )
                except Exception:  # noqa: BLE001
                    pass
    parquet = ParquetRetentionResult()
    if parquet_root is not None:
        parquet = apply_parquet_retention(
            parquet_root,
            policy,
            now=current,
            dry_run=dry_run,
        )
    spool = apply_spool_retention(
        spool_dir,
        policy,
        now=current,
        dry_run=dry_run,
    )
    result = RetentionResult(
        postgres=postgres,
        parquet=parquet,
        spool=spool,
        dry_run=dry_run,
    )
    # Stash canary age on the result for the CLI to print; ignore if dataclass
    # is frozen (it is — use object.__setattr__ via __dict__ if needed).
    try:
        object.__setattr__(result, "postgres_canary_age_ms", canary_age_ms)
    except Exception:  # noqa: BLE001
        pass
    return result


def apply_postgres_retention(
    database_url: str,
    policy: RetentionPolicy,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    vacuum: bool = True,
) -> PostgresRetentionResult:
    import psycopg2

    current = now or datetime.now(timezone.utc)
    tiers: list[tuple[str, str, str, int]] = []
    if policy.events_days is not None:
        tiers.append(("events", POSTGRES_EVENTS_TABLE, "ts", policy.events_days))
    if policy.facts_days is not None:
        for table in POSTGRES_FACT_TABLES:
            tiers.append(("facts", table, "ts", policy.facts_days))
    if policy.latest_state_days is not None:
        tiers.append(
            ("latest_state", POSTGRES_LATEST_STATE_TABLE, "updated_at", policy.latest_state_days)
        )
    if policy.admin_audit_days is not None:
        tiers.append(("admin_audit", POSTGRES_ADMIN_AUDIT_TABLE, "ts", policy.admin_audit_days))

    results: list[PostgresTableRetentionResult] = []

    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            if not dry_run:
                cur.execute("DROP INDEX IF EXISTS mm_events_run_topic_ts_idx")
                conn.commit()
            results.extend(
                apply_postgres_dashboard_storage_compaction(
                    cur,
                    conn=conn,
                    dry_run=dry_run,
                )
            )
            for tier, table, ts_column, retention_days in tiers:
                cutoff = current - timedelta(days=int(retention_days))
                rows = _postgres_delete_before(
                    cur,
                    conn=conn,
                    table=table,
                    ts_column=ts_column,
                    cutoff=cutoff,
                    dry_run=dry_run,
                )
                results.append(
                    PostgresTableRetentionResult(
                        table=table,
                        tier=tier,
                        retention_days=int(retention_days),
                        rows_affected=rows,
                    )
                )
            if vacuum and not dry_run and any(
                r.table == POSTGRES_EVENTS_TABLE for r in results
            ):
                # Without VACUUM ANALYZE, the next insert churns the heap and
                # future retention deletes look slow. Cheap when run once a day.
                started = time.monotonic()
                cur.execute(f"VACUUM ANALYZE {POSTGRES_EVENTS_TABLE}")
                conn.commit()
                duration_ms = (time.monotonic() - started) * 1000.0
                _emit_retention_metric("vacuum_duration_ms", duration_ms, {"table": POSTGRES_EVENTS_TABLE})
        if dry_run:
            conn.rollback()
    finally:
        conn.close()
    _emit_retention_metric(
        "rows_deleted",
        float(sum(r.rows_affected for r in results)),
        {"store": "postgres"},
    )
    return PostgresRetentionResult(tables=tuple(results))


def _emit_retention_metric(name: str, value: float, labels: dict[str, str]) -> None:
    """Best-effort metrics emission. Never raise — retention must not fail on a missing metric."""
    try:
        from mm.observability import counter_inc, gauge
        from mm.observability.schema import (
            MM_RETENTION_BYTES_DELETED_TOTAL,
            MM_RETENTION_ROWS_DELETED_TOTAL,
            MM_RETENTION_VACUUM_DURATION_MS,
        )
        if name == "rows_deleted":
            counter_inc(MM_RETENTION_ROWS_DELETED_TOTAL, inc=value, labels=labels)
        elif name == "vacuum_duration_ms":
            gauge(MM_RETENTION_VACUUM_DURATION_MS, value, labels=labels)
        elif name == "bytes_deleted":
            counter_inc(MM_RETENTION_BYTES_DELETED_TOTAL, inc=value, labels=labels)
    except Exception:  # noqa: BLE001
        return


def apply_parquet_retention(
    parquet_root: str | Path,
    policy: RetentionPolicy,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
) -> ParquetRetentionResult:
    if policy.parquet_days is None:
        return ParquetRetentionResult()

    root = Path(parquet_root).expanduser()
    if not root.exists():
        return ParquetRetentionResult()

    current = (now or datetime.now(timezone.utc)).date()
    cutoff_date = current - timedelta(days=int(policy.parquet_days))
    partitions_removed = 0
    files_removed = 0
    bytes_removed = 0

    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        match = _EVENT_DATE_RE.match(path.name)
        if match is None:
            continue
        partition_date = date.fromisoformat(match.group(1))
        if partition_date >= cutoff_date:
            continue
        partition_bytes = _tree_size(path)
        partition_files = sum(1 for child in path.rglob("*") if child.is_file())
        if dry_run:
            partitions_removed += 1
            files_removed += partition_files
            bytes_removed += partition_bytes
            continue
        shutil.rmtree(path)
        partitions_removed += 1
        files_removed += partition_files
        bytes_removed += partition_bytes

    _emit_retention_metric(
        "rows_deleted",
        float(partitions_removed),
        {"store": "parquet"},
    )
    _emit_retention_metric(
        "bytes_deleted",
        float(bytes_removed),
        {"store": "parquet"},
    )
    return ParquetRetentionResult(
        partitions_removed=partitions_removed,
        files_removed=files_removed,
        bytes_removed=bytes_removed,
    )


def apply_spool_retention(
    spool_dir: str | Path,
    policy: RetentionPolicy,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
) -> SpoolRetentionResult:
    root = Path(spool_dir).expanduser()
    if not root.exists():
        return SpoolRetentionResult()

    current = now or datetime.now(timezone.utc)
    files_removed = 0
    bytes_removed = 0
    archive_root = root / "reingested"

    for path in sorted(root.rglob("*.jsonl")):
        if not path.is_file():
            continue
        in_archive = _is_inside(path, archive_root)
        retention_days = policy.spool_archive_days if in_archive else policy.spool_failed_days
        if retention_days is None:
            continue
        age_seconds = current.timestamp() - path.stat().st_mtime
        if age_seconds <= int(retention_days) * 86_400:
            continue
        size = path.stat().st_size
        if dry_run:
            files_removed += 1
            bytes_removed += size
            continue
        path.unlink()
        files_removed += 1
        bytes_removed += size
        if in_archive:
            _remove_empty_parents(path.parent, stop=root)

    _emit_retention_metric(
        "rows_deleted",
        float(files_removed),
        {"store": "spool"},
    )
    _emit_retention_metric(
        "bytes_deleted",
        float(bytes_removed),
        {"store": "spool"},
    )
    return SpoolRetentionResult(files_removed=files_removed, bytes_removed=bytes_removed)


def ensure_postgres_retention_indexes(database_url: str) -> None:
    """Create ts indexes concurrently so retention deletes do not seq-scan."""

    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for table, index_name in POSTGRES_TS_INDEXES:
                _assert_known_postgres_table(table)
                cur.execute(
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} ON {table} (ts)"
                )
    finally:
        conn.close()


def write_postgres_canary(database_url: str, *, now: datetime | None = None) -> int:
    """Write a canary row to ``mm_events`` and return the previous canary's age in ms.

    The canary is a single row keyed by ``event_id = '_retention_canary'``,
    refreshed on every run. If the row is more than 2× the cron interval old,
    a downstream alert (cluster_health) should fire. Returns 0 on first run.
    """

    import psycopg2

    current = now or datetime.now(timezone.utc)
    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mm_events (
                    event_id, schema, version, run_id, source, topic, event_type,
                    timestamp_ms, ts, payload, envelope
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (event_id) DO UPDATE SET
                    timestamp_ms = EXCLUDED.timestamp_ms,
                    ts = EXCLUDED.ts,
                    payload = EXCLUDED.payload,
                    envelope = EXCLUDED.envelope
                RETURNING ts
                """,
                (
                    "_retention_canary",
                    "retention.canary",
                    1,
                    "retention",
                    "apply_retention",
                    "_retention",
                    "_retention_canary",
                    int(current.timestamp() * 1000),
                    current,
                    "{}",
                    "{}",
                ),
            )
            cur.execute(
                "SELECT ts FROM mm_events WHERE event_id = %s",
                ("_retention_canary",),
            )
            row = cur.fetchone()
            if row is None:
                return 0
            age_ms = int((current - row[0]).total_seconds() * 1000.0)
    finally:
        conn.close()
    try:
        from mm.observability import gauge
        from mm.observability.schema import MM_RETENTION_CANARY_AGE_MS
        gauge(MM_RETENTION_CANARY_AGE_MS, float(age_ms), labels={"store": "postgres"})
    except Exception:  # noqa: BLE001
        pass
    return age_ms


def _postgres_delete_before(
    cursor: Any,
    *,
    conn: Any,
    table: str,
    ts_column: str,
    cutoff: datetime,
    dry_run: bool,
) -> int:
    _assert_known_postgres_table(table)
    _assert_known_postgres_ts_column(ts_column)
    if dry_run:
        cursor.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {ts_column} < %s",
            (cutoff,),
        )
        row = cursor.fetchone()
        return int(row[0] if row else 0)
    deleted = 0
    while True:
        cursor.execute(
            f"DELETE FROM {table} WHERE ctid IN ("
            f"SELECT ctid FROM {table} WHERE {ts_column} < %s LIMIT %s"
            f")",
            (cutoff, POSTGRES_DELETE_BATCH),
        )
        batch = int(cursor.rowcount or 0)
        conn.commit()
        deleted += batch
        if batch < POSTGRES_DELETE_BATCH:
            break
    return deleted


DASHBOARD_UNUSED_EVENT_PREDICATE = """(
  event_id <> '_retention_canary'
  AND (
    topic IN ('portfolio:allocations', 'worker:capability', 'strategy:actions')
    OR topic LIKE 'strategy:heartbeat:%'
    OR topic LIKE 'strategy:joint:frames:%'
    OR event_type = 'JointStrategyFrame'
  )
)"""


def apply_postgres_dashboard_storage_compaction(
    cursor: Any,
    *,
    conn: Any,
    dry_run: bool,
) -> list[PostgresTableRetentionResult]:
    """Drop bus topics Grafana does not query, and strip duplicate envelope JSON."""

    unused = _postgres_delete_matching(
        cursor,
        conn=conn,
        table=POSTGRES_EVENTS_TABLE,
        predicate=DASHBOARD_UNUSED_EVENT_PREDICATE,
        dry_run=dry_run,
    )
    envelopes = _postgres_empty_json_column(
        cursor,
        conn=conn,
        table=POSTGRES_EVENTS_TABLE,
        column="envelope",
        dry_run=dry_run,
    )
    latest_envelopes = _postgres_empty_json_column(
        cursor,
        conn=conn,
        table=POSTGRES_LATEST_STATE_TABLE,
        column="envelope",
        dry_run=dry_run,
    )
    return [
        PostgresTableRetentionResult(
            table=POSTGRES_EVENTS_TABLE,
            tier="unused_topics",
            retention_days=0,
            rows_affected=unused,
        ),
        PostgresTableRetentionResult(
            table=POSTGRES_EVENTS_TABLE,
            tier="empty_envelope",
            retention_days=0,
            rows_affected=envelopes,
        ),
        PostgresTableRetentionResult(
            table=POSTGRES_LATEST_STATE_TABLE,
            tier="empty_envelope",
            retention_days=0,
            rows_affected=latest_envelopes,
        ),
    ]


def _postgres_delete_matching(
    cursor: Any,
    *,
    conn: Any,
    table: str,
    predicate: str,
    dry_run: bool,
) -> int:
    _assert_known_postgres_table(table)
    if dry_run:
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {predicate}")
        row = cursor.fetchone()
        return int(row[0] if row else 0)
    deleted = 0
    while True:
        cursor.execute(
            f"DELETE FROM {table} WHERE ctid IN ("
            f"SELECT ctid FROM {table} WHERE {predicate} LIMIT %s"
            f")",
            (POSTGRES_DELETE_BATCH,),
        )
        batch = int(cursor.rowcount or 0)
        conn.commit()
        deleted += batch
        if batch < POSTGRES_DELETE_BATCH:
            break
    return deleted


def _postgres_empty_json_column(
    cursor: Any,
    *,
    conn: Any,
    table: str,
    column: str,
    dry_run: bool,
) -> int:
    _assert_known_postgres_table(table)
    if column != "envelope":
        raise ValueError(f"unsupported json compaction column: {column!r}")
    if dry_run:
        cursor.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} <> '{{}}'::jsonb"
        )
        row = cursor.fetchone()
        return int(row[0] if row else 0)
    updated = 0
    while True:
        cursor.execute(
            f"UPDATE {table} SET {column} = '{{}}'::jsonb WHERE ctid IN ("
            f"SELECT ctid FROM {table} WHERE {column} <> '{{}}'::jsonb LIMIT %s"
            f")",
            (POSTGRES_DELETE_BATCH,),
        )
        batch = int(cursor.rowcount or 0)
        conn.commit()
        updated += batch
        if batch < POSTGRES_DELETE_BATCH:
            break
    return updated


def _assert_known_postgres_table(table: str) -> None:
    allowed = {
        POSTGRES_EVENTS_TABLE,
        POSTGRES_LATEST_STATE_TABLE,
        POSTGRES_ADMIN_AUDIT_TABLE,
        *POSTGRES_FACT_TABLES,
    }
    if table not in allowed:
        raise ValueError(f"unsupported postgres retention table: {table!r}")


def _assert_known_postgres_ts_column(column: str) -> None:
    if column not in {"ts", "updated_at"}:
        raise ValueError(f"unsupported postgres retention timestamp column: {column!r}")


def _optional_positive_int(value: Any, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    days = int(value)
    if days <= 0:
        raise ValueError("retention day values must be positive")
    return days


def _tree_size(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _remove_empty_parents(path: Path, *, stop: Path) -> None:
    current = path
    while current != stop and current.exists():
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent
