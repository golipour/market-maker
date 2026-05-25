from __future__ import annotations

import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from mm.core.models.market import msgspec

EVENT_SCHEMA = "mm.event"
EVENT_VERSION = 1


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Versioned envelope shared by bus, persistence, replay, and admin APIs."""

    topic: str
    event_type: str
    payload: Any
    schema: str = EVENT_SCHEMA
    version: int = EVENT_VERSION
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str = ""
    source: str = "mm-runtime"
    exchange: str | None = None
    symbol: str | None = None
    actor_id: str | None = None
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "source": self.source,
            "topic": self.topic,
            "event_type": self.event_type,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "actor_id": self.actor_id,
            "timestamp_ms": self.timestamp_ms,
            "payload": to_jsonable(self.payload),
        }


def envelope_event(
    topic: str,
    event: Any,
    *,
    run_id: str = "",
    source: str = "mm-runtime",
    event_id: str | None = None,
) -> EventEnvelope:
    """Build a durable envelope while preserving the original event payload."""

    payload = to_jsonable(event)
    return EventEnvelope(
        topic=topic,
        event_type=_event_type(event, payload),
        payload=payload,
        event_id=event_id or uuid.uuid4().hex,
        run_id=run_id,
        source=source,
        exchange=_pick(payload, "exchange"),
        symbol=_pick(payload, "symbol"),
        actor_id=_pick(payload, "actor_id"),
        timestamp_ms=_timestamp_ms(payload),
    )


def encode_envelope(envelope: EventEnvelope) -> str:
    return json.dumps(envelope.as_dict(), ensure_ascii=False, separators=(",", ":"))


def decode_envelope(raw: str | bytes | dict[str, Any]) -> EventEnvelope:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, dict):
        raise ValueError("event envelope must decode to an object")
    if obj.get("schema") != EVENT_SCHEMA:
        raise ValueError(f"unsupported event schema: {obj.get('schema')!r}")
    return EventEnvelope(
        schema=str(obj.get("schema", EVENT_SCHEMA)),
        version=int(obj.get("version", EVENT_VERSION)),
        event_id=str(obj.get("event_id", "")),
        run_id=str(obj.get("run_id", "")),
        source=str(obj.get("source", "")),
        topic=str(obj.get("topic", "")),
        event_type=str(obj.get("event_type", "")),
        exchange=obj.get("exchange"),
        symbol=obj.get("symbol"),
        actor_id=obj.get("actor_id"),
        timestamp_ms=int(obj.get("timestamp_ms", 0)),
        payload=obj.get("payload"),
    )


def to_jsonable(value: Any) -> Any:
    """Convert msgspec/dataclass/runtime objects to JSON-compatible data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    as_payload = getattr(value, "as_payload", None)
    if callable(as_payload):
        return to_jsonable(as_payload())
    if msgspec is not None and msgspec.structs.is_struct(value):
        return {k: to_jsonable(v) for k, v in msgspec.structs.asdict(value).items()}
    if dataclasses.is_dataclass(value):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return to_jsonable(value.item())
        if isinstance(value, np.ndarray):
            return to_jsonable(value.tolist())
    except Exception:
        pass
    return str(value)


def _event_type(event: Any, payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("type"):
        return str(payload["type"])
    return type(event).__name__


def _pick(payload: Any, *keys: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return None


def _timestamp_ms(payload: Any) -> int:
    if isinstance(payload, dict):
        for key in ("timestamp_ms", "t_end_ms", "created_at_ms"):
            if payload.get(key) is not None:
                try:
                    return int(payload[key])
                except Exception:
                    break
    return int(time.time() * 1000)
