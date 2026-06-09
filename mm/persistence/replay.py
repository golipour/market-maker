from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm.core.events import EventEnvelope, decode_envelope


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


@dataclass(frozen=True, slots=True)
class ReplayLineageChain:
    """Events connected to one strategy frame by explicit lineage ids."""

    frame_id: str
    decisions: tuple[EventEnvelope, ...] = ()
    diagnostics: tuple[EventEnvelope, ...] = ()
    target_books: tuple[EventEnvelope, ...] = ()
    execution_events: tuple[EventEnvelope, ...] = ()
    other_events: tuple[EventEnvelope, ...] = ()

    @property
    def decision_ids(self) -> tuple[str, ...]:
        return _unique_payload_values(self.decisions, "decision_id")

    @property
    def target_book_ids(self) -> tuple[str, ...]:
        return _unique_payload_values(
            (*self.decisions, *self.diagnostics, *self.target_books, *self.execution_events),
            "target_book_id",
        )

    @property
    def execution_plan_ids(self) -> tuple[str, ...]:
        return _unique_payload_values(self.execution_events, "execution_plan_id")

    def execution_events_for_plan(self, execution_plan_id: str) -> tuple[EventEnvelope, ...]:
        wanted = execution_plan_id.strip()
        if not wanted:
            return ()
        return tuple(
            envelope
            for envelope in self.execution_events
            if _payload_value(envelope, "execution_plan_id") == wanted
        )


def read_lineage_chains(path: str | Path) -> dict[str, ReplayLineageChain]:
    """Read persisted rows and group replay events by ``frame_id``.

    Events without a ``frame_id`` are ignored because they cannot be joined to a
    strategy frame without timestamp/topic inference.
    """

    return lineage_chains_from_envelopes(read_envelopes(path))


def lineage_chains_from_envelopes(envelopes: Iterable[EventEnvelope]) -> dict[str, ReplayLineageChain]:
    groups: dict[str, _MutableLineageChain] = {}
    for envelope in sorted(envelopes, key=lambda item: item.timestamp_ms):
        frame_id = _payload_value(envelope, "frame_id")
        if not frame_id:
            continue
        groups.setdefault(frame_id, _MutableLineageChain(frame_id)).add(envelope)
    return {frame_id: group.freeze() for frame_id, group in groups.items()}


@dataclass(slots=True)
class _MutableLineageChain:
    frame_id: str
    decisions: list[EventEnvelope] = field(default_factory=list)
    diagnostics: list[EventEnvelope] = field(default_factory=list)
    target_books: list[EventEnvelope] = field(default_factory=list)
    execution_events: list[EventEnvelope] = field(default_factory=list)
    other_events: list[EventEnvelope] = field(default_factory=list)

    def add(self, envelope: EventEnvelope) -> None:
        if envelope.event_type in {"StrategyDecision", "ControllerFallback", "ControllerError"}:
            self.decisions.append(envelope)
        elif envelope.event_type == "ControllerDiagnostics":
            self.diagnostics.append(envelope)
        elif envelope.event_type == "JointTargetBook":
            self.target_books.append(envelope)
        elif _is_execution_event(envelope):
            self.execution_events.append(envelope)
        else:
            self.other_events.append(envelope)

    def freeze(self) -> ReplayLineageChain:
        return ReplayLineageChain(
            frame_id=self.frame_id,
            decisions=tuple(self.decisions),
            diagnostics=tuple(self.diagnostics),
            target_books=tuple(self.target_books),
            execution_events=tuple(self.execution_events),
            other_events=tuple(self.other_events),
        )


def _is_execution_event(envelope: EventEnvelope) -> bool:
    return envelope.topic.startswith(("execution:orders:", "execution:state:")) or envelope.event_type in {
        "ExecutionSyncRequested",
        "ExecutionSyncCompleted",
        "ExecutionSyncFailed",
        "PlaceOrdersRequested",
        "PlaceOrdersAccepted",
        "PlaceOrdersFailed",
        "CancelOrdersRequested",
        "CancelOrdersAccepted",
        "CancelOrdersFailed",
        "OrderValidationRejected",
        "ExecutionSnapshot",
    }


def _payload_value(envelope: EventEnvelope, key: str) -> str:
    payload = envelope.payload
    if not isinstance(payload, dict):
        return ""
    value = payload.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _unique_payload_values(envelopes: tuple[EventEnvelope, ...], key: str) -> tuple[str, ...]:
    values = []
    seen = set()
    for envelope in envelopes:
        value = _payload_value(envelope, key)
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return tuple(values)
