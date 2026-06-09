from __future__ import annotations

import json

from mm.core.events import envelope_event
from mm.tools.reingest_postgres_spool import reingest_spool
from tests.unit.test_postgres_writer_startup import _install_fake_psycopg2


def test_postgres_spool_reingest_writes_events_and_archives_files(monkeypatch, tmp_path) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    spool_file = tmp_path / "event_date=2026-06-09" / "hour=12" / "postgres-failed.jsonl"
    envelope = envelope_event(
        "market:events:nobitex:BTC_IRT:1000",
        {
            "type": "MarketEvent",
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "t_end_ms": 2_000,
        },
    )
    _write_spool(spool_file, [envelope.as_dict()])

    result = reingest_spool(
        spool_dir=tmp_path,
        database_url="postgresql://example/db",
        archive_dir=tmp_path / "done",
    )

    assert result.files_seen == 1
    assert result.files_reingested == 1
    assert result.events_reingested == 1
    assert not spool_file.exists()
    archived = tmp_path / "done" / "event_date=2026-06-09" / "hour=12" / "postgres-failed.jsonl"
    assert archived.exists()
    conn = fake.connections[0]
    event_call = next(call for call in conn.execute_values_calls if "INSERT INTO mm_events" in call["sql"])
    [row] = event_call["rows"]
    assert row[0] == envelope.event_id


def test_postgres_spool_reingest_dry_run_does_not_write_or_archive(monkeypatch, tmp_path) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    spool_file = tmp_path / "postgres-failed.jsonl"
    envelope = envelope_event(
        "strategy:decisions",
        {"type": "ControllerFallback", "frame_id": "frame-1", "reason": "timeout"},
    )
    _write_spool(spool_file, [envelope.as_dict()])

    result = reingest_spool(
        spool_dir=tmp_path,
        database_url="postgresql://example/db",
        dry_run=True,
    )

    assert result.files_seen == 1
    assert result.files_reingested == 1
    assert result.events_seen == 1
    assert result.events_reingested == 1
    assert spool_file.exists()
    assert fake.connections == []


def _write_spool(path, envelopes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "schema": "mm.persistence.failed_batch",
                "version": 1,
                "writer": "postgres",
                "error": "RuntimeError",
                "message": "commit failed",
                "envelope": envelope,
            }
        )
        for envelope in envelopes
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
