from __future__ import annotations

import asyncio
import json
import os
import time
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.events import EventEnvelope, envelope_event, to_jsonable
from mm.observability import counter_inc, gauge, histogram_observe, log_event
from mm.observability import schema as obs_schema
from mm.observability.event_payloads import observe_envelope


class ParquetEventWriter:
    """Append runtime events to partitioned Parquet files when pyarrow is present.

    If pyarrow is not installed, the writer falls back to JSONL files in the same
    partition layout so replay data is still captured during development.
    """

    def __init__(
        self,
        *,
        bus: AbstractEventBus,
        topics: tuple[str, ...],
        root_path: str,
        batch_size: int = 100,
        flush_interval_ms: int = 1000,
        run_id: str = "",
        compact_payloads: bool = True,
    ):
        self.bus = bus
        self.topics = topics
        self.root_path = Path(root_path).expanduser()
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_ms = max(50, int(flush_interval_ms))
        self.run_id = run_id
        self.compact_payloads = bool(compact_payloads)
        self._queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=self.batch_size * 10)
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._stopped.clear()
        self._tasks = [asyncio.create_task(self._consume(topic)) for topic in self.topics]
        self._tasks.append(asyncio.create_task(self._write_loop()))

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _consume(self, topic: str) -> None:
        try:
            async for event in self.bus.subscribe(topic):
                envelope = envelope_event(topic, event, run_id=self.run_id, source="parquet-writer")
                observe_envelope(envelope, stage="parquet_enqueue")
                await self._queue.put(envelope)
                gauge(
                    obs_schema.MM_PERSISTENCE_QUEUE_DEPTH,
                    float(self._queue.qsize()),
                    labels={"writer": "parquet"},
                    help="Persistence writer queue depth",
                )
        except asyncio.CancelledError:
            raise

    async def _write_loop(self) -> None:
        batch: list[EventEnvelope] = []
        timeout = self.flush_interval_ms / 1000.0
        try:
            while not self._stopped.is_set():
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=timeout))
                except asyncio.TimeoutError:
                    pass
                if batch and (len(batch) >= self.batch_size or self._queue.empty()):
                    to_write = batch
                    batch = []
                    queue_depth = self._queue.qsize()
                    gauge(
                        obs_schema.MM_PERSISTENCE_QUEUE_DEPTH,
                        float(queue_depth),
                        labels={"writer": "parquet"},
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
        grouped: dict[Path, list[dict[str, Any]]] = {}
        for envelope in batch:
            stored = _compact_envelope(envelope) if self.compact_payloads else envelope
            path = self._partition_path(stored)
            grouped.setdefault(path, []).append(_row(stored))
        sink = "parquet"
        written_count = 0
        for path, rows in grouped.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                schema = _parquet_schema(pa)
                table = pa.Table.from_pylist(rows, schema=schema)
                tmp = path.with_suffix(path.suffix + ".tmp")
                if path.exists():
                    existing = _read_existing_parquet(pa, pq, path, schema)
                    table = pa.concat_tables([existing, table], promote_options="default")
                pq.write_table(table, tmp)
                os.replace(tmp, path)
                written_count += len(rows)
            except ImportError:
                sink = "jsonl"
                fallback = path.with_suffix(".jsonl")
                try:
                    with fallback.open("a", encoding="utf-8") as f:
                        for row in rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written_count += len(rows)
                except Exception as exc:
                    counter_inc(
                        "mm_persistence_errors_total",
                        labels={"writer": "parquet", "error": type(exc).__name__},
                    )
                    log_event(
                        obs_schema.EVENT_PERSISTENCE_BATCH_FAILED,
                        {
                            "writer": "parquet",
                            "batch_size": len(batch),
                            "error": type(exc).__name__,
                            "message": str(exc),
                            "path": str(fallback),
                            "retryable": True,
                        },
                    )
            except Exception as exc:
                counter_inc(
                    "mm_persistence_errors_total",
                    labels={"writer": "parquet", "error": type(exc).__name__},
                )
                log_event(
                    obs_schema.EVENT_PERSISTENCE_BATCH_FAILED,
                    {
                        "writer": "parquet",
                        "batch_size": len(batch),
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "path": str(path),
                        "retryable": True,
                    },
                )
                continue
        latency_ms = _elapsed_ms(started_ns)
        if written_count <= 0:
            return
        counter_inc("mm_persistence_events_written_total", written_count, labels={"writer": "parquet"})
        gauge("mm_persistence_batch_size", written_count, labels={"writer": "parquet"})
        gauge(
            obs_schema.MM_PERSISTENCE_QUEUE_DEPTH,
            float(queue_depth),
            labels={"writer": "parquet"},
            help="Persistence writer queue depth",
        )
        histogram_observe(
            obs_schema.MM_PERSISTENCE_WRITE_LATENCY_MS,
            latency_ms,
            labels={"writer": "parquet"},
            help="Persistence batch write latency",
        )
        log_event(
            obs_schema.EVENT_PERSISTENCE_BATCH_WRITTEN,
            {
                "writer": "parquet",
                "sink": sink,
                "batch_size": written_count,
                "queue_depth": int(queue_depth),
                "latency_ms": latency_ms,
            },
        )

    def _partition_path(self, envelope: EventEnvelope) -> Path:
        dt = datetime.fromtimestamp(envelope.timestamp_ms / 1000.0, tz=timezone.utc)
        event_date = dt.strftime("%Y-%m-%d")
        event_hour = dt.strftime("%H")
        exchange = envelope.exchange or "none"
        symbol = envelope.symbol or "none"
        event_type = envelope.event_type or "event"
        filename = f"{event_hour}-{event_type}.parquet"
        return self.root_path / f"event_date={event_date}" / f"exchange={exchange}" / f"symbol={symbol}" / filename


def _row(envelope: EventEnvelope) -> dict[str, Any]:
    return {
        "event_id": envelope.event_id,
        "schema": envelope.schema,
        "version": envelope.version,
        "run_id": envelope.run_id,
        "source": envelope.source,
        "topic": envelope.topic,
        "event_type": envelope.event_type,
        "exchange": envelope.exchange,
        "symbol": envelope.symbol,
        "actor_id": envelope.actor_id,
        "timestamp_ms": envelope.timestamp_ms,
        "envelope_json": json.dumps(envelope.as_dict(), ensure_ascii=False),
    }


def _parquet_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("event_id", pa.string()),
            ("schema", pa.string()),
            ("version", pa.int64()),
            ("run_id", pa.string()),
            ("source", pa.string()),
            ("topic", pa.string()),
            ("event_type", pa.string()),
            ("exchange", pa.string()),
            ("symbol", pa.string()),
            ("actor_id", pa.string()),
            ("timestamp_ms", pa.int64()),
            ("envelope_json", pa.string()),
        ]
    )


def _read_existing_parquet(pa: Any, pq: Any, path: Path, schema: Any) -> Any:
    # pq.read_table() can infer Hive partition columns from event_date=/exchange=/
    # directories and then collide with the real string columns stored in the file.
    table = pq.ParquetFile(path).read()
    return _align_table_to_schema(pa, table, schema)


def _align_table_to_schema(pa: Any, table: Any, schema: Any) -> Any:
    arrays = []
    names = set(table.column_names)
    for field in schema:
        if field.name in names:
            arrays.append(table[field.name].cast(field.type))
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _compact_envelope(envelope: EventEnvelope) -> EventEnvelope:
    payload = _compact_payload(envelope.event_type, envelope.payload)
    if payload is envelope.payload:
        return envelope
    return dataclasses.replace(envelope, payload=payload)


def _compact_payload(event_type: str, payload: Any) -> Any:
    value = to_jsonable(payload)
    if not isinstance(value, dict):
        return value
    if event_type == "MarketEvent":
        return _compact_market_event_payload(value)
    if event_type == "PrivateStateSnapshot":
        return _compact_private_snapshot_payload(value)
    return value


def _compact_market_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    trades = out.get("trades")
    if isinstance(trades, list) and trades:
        out["trade_count"] = len(trades)
        first_ts = _first_numeric(trades, "timestamp_ms")
        last_ts = _last_numeric(trades, "timestamp_ms")
        if first_ts is not None:
            out["first_trade_timestamp_ms"] = first_ts
        if last_ts is not None:
            out["last_trade_timestamp_ms"] = last_ts
        out["trades"] = []
        out["trades_compacted"] = True
    return out


def _compact_private_snapshot_payload(payload: dict[str, Any]) -> dict[str, Any]:
    balances = payload.get("balances")
    if not isinstance(balances, dict) or not balances:
        return payload
    nonzero_balances = {
        asset: row
        for asset, row in balances.items()
        if not _balance_payload_is_zero(row)
    }
    if len(nonzero_balances) == len(balances):
        return payload
    out = dict(payload)
    out["balance_count"] = len(balances)
    out["nonzero_balance_count"] = len(nonzero_balances)
    out["balances"] = nonzero_balances
    out["balances_compacted"] = True
    return out


def _balance_payload_is_zero(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    try:
        return (
            abs(float(row.get("total", 0.0) or 0.0)) <= 1e-12
            and abs(float(row.get("free", 0.0) or 0.0)) <= 1e-12
            and abs(float(row.get("locked", 0.0) or 0.0)) <= 1e-12
        )
    except Exception:
        return False


def _first_numeric(rows: list[Any], key: str) -> int | float | None:
    for row in rows:
        if isinstance(row, dict) and row.get(key) is not None:
            return row[key]
    return None


def _last_numeric(rows: list[Any], key: str) -> int | float | None:
    for row in reversed(rows):
        if isinstance(row, dict) and row.get(key) is not None:
            return row[key]
    return None


def _elapsed_ms(started_ns: int) -> float:
    return (time.monotonic_ns() - started_ns) / 1_000_000
