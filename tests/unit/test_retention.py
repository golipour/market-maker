from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from mm.persistence.retention import (
    RetentionPolicy,
    apply_parquet_retention,
    apply_postgres_retention,
    apply_retention,
    apply_spool_retention,
    retention_policy_from_mapping,
)
from mm.runtime.config import runtime_config_from_mapping
from tests.unit.conftest import venues_payload
from tests.unit.test_postgres_writer_startup import _install_fake_psycopg2


def test_retention_policy_from_mapping_uses_defaults() -> None:
    policy = retention_policy_from_mapping({})

    assert policy.events_days == 14
    assert policy.facts_days == 90
    assert policy.latest_state_days is None
    assert policy.parquet_days == 90


def test_runtime_config_loads_persistence_retention() -> None:
    payload = venues_payload(local={"exchange": "nobitex", "symbol": "BTC_IRT"})
    payload["persistence"] = {
        "enabled": True,
        "retention": {"events_days": 7, "facts_days": 30, "latest_state_days": 180},
    }
    config = runtime_config_from_mapping(payload)

    assert config.persistence.retention.events_days == 7
    assert config.persistence.retention.facts_days == 30
    assert config.persistence.retention.latest_state_days == 180


def test_apply_parquet_retention_removes_old_partitions(tmp_path) -> None:
    old = tmp_path / "event_date=2020-01-01" / "exchange=nobitex" / "symbol=BTC_IRT" / "event_type=MarketEvent" / "hour=00"
    new = tmp_path / "event_date=2099-01-01" / "exchange=nobitex" / "symbol=BTC_IRT" / "event_type=MarketEvent" / "hour=00"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    old_file = old / "part-old.parquet"
    new_file = new / "part-new.parquet"
    old_file.write_bytes(b"old")
    new_file.write_bytes(b"new")

    result = apply_parquet_retention(
        tmp_path,
        RetentionPolicy(parquet_days=30),
        now=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )

    assert result.partitions_removed == 1
    assert result.files_removed == 1
    assert not old.exists()
    assert new_file.exists()


def test_apply_parquet_retention_dry_run_keeps_files(tmp_path) -> None:
    partition = tmp_path / "event_date=2020-01-01"
    partition.mkdir()
    file_path = partition / "part-old.parquet"
    file_path.write_bytes(b"old")

    result = apply_parquet_retention(
        tmp_path,
        RetentionPolicy(parquet_days=30),
        now=datetime(2026, 6, 9, tzinfo=timezone.utc),
        dry_run=True,
    )

    assert result.partitions_removed == 1
    assert file_path.exists()


def test_apply_spool_retention_removes_old_failed_and_archived_files(tmp_path) -> None:
    failed = tmp_path / "failed-batch.jsonl"
    archived = tmp_path / "reingested" / "done.jsonl"
    archived.parent.mkdir(parents=True)
    failed.write_text("failed\n", encoding="utf-8")
    archived.write_text("archived\n", encoding="utf-8")
    old = time.time() - (40 * 86_400)
    os.utime(failed, (old, old))
    os.utime(archived, (old, old))

    result = apply_spool_retention(
        tmp_path,
        RetentionPolicy(spool_failed_days=30, spool_archive_days=7),
        now=datetime.now(timezone.utc),
    )

    assert result.files_removed == 2
    assert not failed.exists()
    assert not archived.exists()


def test_apply_postgres_retention_deletes_by_tier(monkeypatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    policy = RetentionPolicy(
        events_days=14,
        facts_days=90,
        latest_state_days=None,
        admin_audit_days=365,
    )

    result = apply_postgres_retention(
        "postgresql://example/db",
        policy,
        now=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )

    conn = fake.connections[0]
    delete_sql = [call["sql"] for call in conn.execute_calls if "DELETE FROM" in call["sql"]]
    assert any("DELETE FROM mm_events WHERE ctid IN" in sql for sql in delete_sql)
    assert any("DELETE FROM mm_market_facts WHERE ctid IN" in sql for sql in delete_sql)
    assert any("DELETE FROM admin_audit_log WHERE ctid IN" in sql for sql in delete_sql)
    assert not any("mm_latest_state" in sql for sql in delete_sql)
    assert result.rows_affected == len(delete_sql)
    assert conn.commit_count >= 1


def test_apply_postgres_retention_dry_run_counts_without_delete(monkeypatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)

    result = apply_postgres_retention(
        "postgresql://example/db",
        RetentionPolicy(
            events_days=7,
            facts_days=None,
            latest_state_days=None,
            admin_audit_days=None,
            parquet_days=None,
            spool_archive_days=None,
            spool_failed_days=None,
        ),
        dry_run=True,
    )

    conn = fake.connections[0]
    assert conn.execute_calls[0]["sql"].startswith("SELECT COUNT(*) FROM mm_events")
    assert conn.commit_count == 0
    assert conn.rollback_count == 1
    assert len(result.tables) == 1
    assert result.tables[0].table == "mm_events"


def test_apply_retention_combines_all_tiers(tmp_path, monkeypatch) -> None:
    _install_fake_psycopg2(monkeypatch)
    parquet_root = tmp_path / "parquet"
    partition = parquet_root / "event_date=2020-01-01"
    partition.mkdir(parents=True)
    (partition / "part-old.parquet").write_bytes(b"x")
    spool = tmp_path / "spool"
    spool.mkdir()
    spool_file = spool / "failed.jsonl"
    spool_file.write_text("x\n", encoding="utf-8")
    now = datetime(2026, 6, 9, tzinfo=timezone.utc)
    old = now.timestamp() - (40 * 86_400)
    os.utime(spool_file, (old, old))

    result = apply_retention(
        RetentionPolicy(),
        database_url="postgresql://example/db",
        parquet_root=parquet_root,
        spool_dir=spool,
        now=now,
    )

    assert result.postgres.rows_affected > 0
    assert result.parquet.partitions_removed == 1
    assert result.spool.files_removed == 1


def test_retention_policy_rejects_non_positive_days() -> None:
    with pytest.raises(ValueError, match="events_days"):
        RetentionPolicy(events_days=0)


def test_apply_postgres_retention_runs_vacuum_after_events_delete(monkeypatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    apply_postgres_retention(
        "postgresql://example/db",
        RetentionPolicy(events_days=7, facts_days=None, latest_state_days=None, admin_audit_days=None),
        now=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )
    conn = fake.connections[0]
    sqls = [call["sql"] for call in conn.execute_calls]
    assert any("VACUUM ANALYZE mm_events" in s for s in sqls)


def test_apply_postgres_retention_skips_vacuum_on_dry_run(monkeypatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    apply_postgres_retention(
        "postgresql://example/db",
        RetentionPolicy(events_days=7, facts_days=None, latest_state_days=None, admin_audit_days=None),
        dry_run=True,
    )
    conn = fake.connections[0]
    sqls = [call["sql"] for call in conn.execute_calls]
    assert not any("VACUUM" in s for s in sqls)


def test_write_postgres_canary_writes_a_canary_row(monkeypatch) -> None:
    from mm.persistence.retention import write_postgres_canary

    fake = _install_fake_psycopg2(monkeypatch)
    # The fake fetchone returns (0,) by default; we want the canary query to
    # return the current time so the function can compute an age.
    fake.connections[0].fetchone_results = [(datetime(2026, 6, 9, tzinfo=timezone.utc),)]
    age_ms = write_postgres_canary(
        "postgresql://example/db",
        now=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )
    conn = fake.connections[0]
    sqls = [call["sql"] for call in conn.execute_calls]
    assert any("INSERT INTO mm_events" in s and "ON CONFLICT" in s for s in sqls)
    assert any("event_id = %s" in s for s in sqls)
    assert age_ms == 0


def test_apply_retention_reflects_canary_age_in_result(monkeypatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    fake.connections[0].fetchone_results = [(datetime(2026, 6, 9, tzinfo=timezone.utc),)]
    result = apply_retention(
        RetentionPolicy(events_days=7, facts_days=None, latest_state_days=None, admin_audit_days=None),
        database_url="postgresql://example/db",
        now=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )
    assert getattr(result, "postgres_canary_age_ms", None) is not None


def test_bus_config_rejects_non_positive_envelope_size() -> None:
    from mm.runtime.config import BusRuntimeConfig

    with pytest.raises(ValueError, match="envelope_size_max_bytes"):
        BusRuntimeConfig(kind="redis_streams", envelope_size_max_bytes=0)


def test_bus_config_accepts_envelope_size_override() -> None:
    from mm.runtime.config import _bus_from_mapping

    config = _bus_from_mapping(
        {
            "kind": "redis_streams",
            "envelope_size_max_bytes": 4096,
        }
    )
    assert config.envelope_size_max_bytes == 4096
