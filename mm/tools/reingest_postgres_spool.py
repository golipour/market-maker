"""Reingest failed Postgres persistence spool files.

Run:
    python -m mm.tools.reingest_postgres_spool --database-url "$MM_DATABASE_URL"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mm.bus.memory import InMemoryBus
from mm.core.events import EventEnvelope, decode_envelope
from mm.persistence.postgres import PostgresEventWriter


@dataclass(frozen=True, slots=True)
class ReingestResult:
    files_seen: int = 0
    files_reingested: int = 0
    files_failed: int = 0
    events_seen: int = 0
    events_reingested: int = 0
    archive_dir: Path | None = None


def reingest_spool(
    *,
    spool_dir: str | Path = "persistence_spool/postgres",
    database_url: str,
    archive: bool = True,
    archive_dir: str | Path | None = None,
    dry_run: bool = False,
    batch_size: int = 500,
) -> ReingestResult:
    root = Path(spool_dir).expanduser()
    files = sorted(root.rglob("*.jsonl")) if root.exists() else []
    archive_root = (
        Path(archive_dir).expanduser()
        if archive_dir is not None
        else root / "reingested" / str(int(time.time() * 1000))
    )
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url=database_url,
        batch_size=batch_size,
        failed_batch_spool_dir=None,
    )
    files_reingested = 0
    files_failed = 0
    events_seen = 0
    events_reingested = 0
    for path in files:
        if _is_inside(path, root / "reingested"):
            continue
        try:
            envelopes = list(_read_spool_file(path))
            events_seen += len(envelopes)
            if not dry_run and envelopes:
                _write_envelopes(writer, envelopes)
                events_reingested += len(envelopes)
            elif dry_run:
                events_reingested += len(envelopes)
            if archive and not dry_run:
                _archive_spool_file(path, root=root, archive_root=archive_root)
            files_reingested += 1
        except Exception:
            files_failed += 1
            raise
    writer._close_connection()
    return ReingestResult(
        files_seen=len([path for path in files if not _is_inside(path, root / "reingested")]),
        files_reingested=files_reingested,
        files_failed=files_failed,
        events_seen=events_seen,
        events_reingested=events_reingested,
        archive_dir=archive_root if archive and not dry_run else None,
    )


def _write_envelopes(writer: PostgresEventWriter, envelopes: list[EventEnvelope]) -> None:
    import psycopg2

    writer._write_batch_once(
        psycopg2,
        envelopes,
        0,
        time.monotonic_ns(),
    )


def _read_spool_file(path: Path) -> list[EventEnvelope]:
    envelopes: list[EventEnvelope] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
                envelope = record.get("envelope") if isinstance(record, dict) else None
                if envelope is None:
                    envelope = record
                envelopes.append(decode_envelope(envelope))
            except Exception as exc:
                raise ValueError(f"invalid spool record in {path}:{line_no}: {exc}") from exc
    return envelopes


def _archive_spool_file(path: Path, *, root: Path, archive_root: Path) -> Path:
    relative = path.relative_to(root)
    destination = archive_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    _remove_empty_parents(path.parent, stop=root)
    return destination


def _remove_empty_parents(path: Path, *, stop: Path) -> None:
    current = path
    while current != stop and current.exists():
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool-dir", default="persistence_spool/postgres")
    parser.add_argument("--database-url", default=os.environ.get("MM_DATABASE_URL", ""))
    parser.add_argument("--archive-dir", default=None)
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("--database-url or MM_DATABASE_URL is required")
    result = reingest_spool(
        spool_dir=args.spool_dir,
        database_url=args.database_url,
        archive=not args.no_archive,
        archive_dir=args.archive_dir,
        dry_run=bool(args.dry_run),
        batch_size=int(args.batch_size),
    )
    print(
        json.dumps(
            {
                "files_seen": result.files_seen,
                "files_reingested": result.files_reingested,
                "files_failed": result.files_failed,
                "events_seen": result.events_seen,
                "events_reingested": result.events_reingested,
                "archive_dir": str(result.archive_dir) if result.archive_dir else None,
            },
            sort_keys=True,
        )
    )
    return 0 if result.files_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
