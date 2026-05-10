from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.events import EventEnvelope, envelope_event


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
    ):
        self.bus = bus
        self.topics = topics
        self.root_path = Path(root_path).expanduser()
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_ms = max(50, int(flush_interval_ms))
        self.run_id = run_id
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
                await self._queue.put(envelope_event(topic, event, run_id=self.run_id, source="parquet-writer"))
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
                    await asyncio.to_thread(self._write_batch, to_write)
        except asyncio.CancelledError:
            if batch:
                await asyncio.to_thread(self._write_batch, batch)
            raise

    def _write_batch(self, batch: list[EventEnvelope]) -> None:
        grouped: dict[Path, list[dict[str, Any]]] = {}
        for envelope in batch:
            path = self._partition_path(envelope)
            grouped.setdefault(path, []).append(_row(envelope))
        for path, rows in grouped.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pa.Table.from_pylist(rows)
                tmp = path.with_suffix(path.suffix + ".tmp")
                if path.exists():
                    existing = pq.read_table(path)
                    table = pa.concat_tables([existing, table], promote_options="default")
                pq.write_table(table, tmp)
                os.replace(tmp, path)
            except ImportError:
                fallback = path.with_suffix(".jsonl")
                with fallback.open("a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
        "payload_json": json.dumps(envelope.payload, ensure_ascii=False),
        "envelope_json": json.dumps(envelope.as_dict(), ensure_ascii=False),
    }
