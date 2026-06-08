from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mm.core.events import decode_envelope


def read_event_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read persisted event rows from Parquet or JSONL fallback files."""

    p = Path(path).expanduser()
    if p.is_dir():
        for child in sorted(p.rglob("*")):
            if child.suffix in (".parquet", ".jsonl"):
                yield from read_event_rows(child)
        return
    if p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read parquet replay files") from exc
        # Avoid Hive partition-column inference from event_date=/exchange=/symbol=/
        # paths colliding with the real columns stored in each event file.
        for row in pq.ParquetFile(p).read().to_pylist():
            yield row
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_envelopes(path: str | Path) -> Iterator[Any]:
    """Yield decoded `mm.event` envelopes from persisted event rows."""

    for row in read_event_rows(path):
        raw = row.get("envelope_json") or row.get("envelope") or row
        yield decode_envelope(raw)


def read_payloads(path: str | Path, *, topic: str | None = None) -> Iterator[Any]:
    """Yield payloads in timestamp order for backtests/replay smoke tests."""

    envelopes = sorted(read_envelopes(path), key=lambda envelope: envelope.timestamp_ms)
    for envelope in envelopes:
        if topic is None or envelope.topic == topic:
            yield envelope.payload
