"""Map exchange order ids and client order ids back to strategy actor ids.

The registry itself is memory only. ``register`` and ``release`` update dictionaries
and, when the mapping actually changes, hand one record to a sink. Durability and
retention live in that sink so the execution loop never rewrites a growing file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from mm.core.models.execution import Fill
from mm.core.models.private import OpenOrder


def _norm_exchange(exchange: str) -> str:
    return exchange.lower().strip()


def _norm_actor(actor_id: str) -> str:
    return actor_id.strip()


def _attribution_cache_dir() -> Path:
    """Directory of the legacy JSON cache.

    New writes do not use this directory. Startup still reads an existing file
    once when Postgres has no rows yet, so an in-pod restart can recover mappings
    written by the previous process.
    """

    configured = os.environ.get("MM_ORDER_ATTRIBUTION_CACHE_DIR", "").strip()
    if configured:
        return Path(configured)
    image_dir = Path("/data/attribution")
    parent = image_dir.parent
    if parent.is_dir() and os.access(parent, os.W_OK):
        return image_dir
    return Path("data/attribution")


def _attribution_cache_path(exchange: str) -> Path:
    return _attribution_cache_dir() / f"{_norm_exchange(exchange)}.json"


def wire_client_order_id_for_exchange(exchange: str, client_order_id: str | None) -> str | None:
    """Apply exchange-specific client-order-id wire transform, if registered."""

    from mm.exchanges.registry import get_exchange_registration

    registration = get_exchange_registration(exchange)
    wire = registration.wire_client_order_id
    if wire is None:
        text = str(client_order_id or "").strip()
        return text or None
    return wire(client_order_id)


def actor_id_from_runtime_client_order_id(client_order_id: str | None) -> str:
    """Parse ``{actor_id}:{side}:{idx}`` runtime ids echoed back by some venues."""

    text = str(client_order_id or "").strip()
    if not text or ":" not in text:
        return ""
    return text.split(":", 1)[0].strip()


@dataclass(frozen=True, slots=True)
class AttributionRecord:
    """One exchange order's actor mapping, open or closed."""

    exchange: str
    exchange_order_id: str
    actor_id: str
    client_order_id: str = ""
    wire_client_order_id: str = ""
    closed_at: datetime | None = None


class OrderAttributionSink(Protocol):
    """Accepts mapping changes. Implementations must not do I/O in ``record``."""

    def record(self, record: AttributionRecord) -> None: ...


def _order_key(exchange: str, exchange_order_id: str) -> str:
    return f"{exchange}:{exchange_order_id}"


def _client_key(exchange: str, client_order_id: str) -> str:
    return f"{exchange}:{client_order_id}"


class OrderAttributionRegistry:
    """In-memory placement registry used to stamp ``actor_id`` on exchange fills."""

    def __init__(self) -> None:
        self._records: dict[str, AttributionRecord] = {}
        self._by_client_order_id: dict[str, str] = {}
        self._client_owner: dict[str, str] = {}
        self._sink: OrderAttributionSink | None = None

    def bind_sink(self, sink: OrderAttributionSink | None) -> None:
        self._sink = sink

    def register(
        self,
        exchange: str,
        exchange_order_id: str,
        actor_id: str,
        *,
        client_order_id: str | None = None,
        wire_client_order_id: str | None = None,
    ) -> None:
        ex = _norm_exchange(exchange)
        actor = _norm_actor(actor_id)
        if not ex or not actor:
            return
        order_id = str(exchange_order_id or "").strip()
        client_id = str(client_order_id or "").strip()
        wire_id = str(wire_client_order_id or "").strip()
        if not order_id and not client_id and not wire_id:
            return
        if order_id:
            previous = self._records.get(_order_key(ex, order_id))
            if previous is not None:
                if not client_id:
                    client_id = previous.client_order_id
                if not wire_id:
                    wire_id = previous.wire_client_order_id
        if not order_id:
            self._index_client_only(ex, actor, client_id, wire_id)
            return
        self._store(
            AttributionRecord(
                exchange=ex,
                exchange_order_id=order_id,
                actor_id=actor,
                client_order_id=client_id,
                wire_client_order_id=wire_id,
                closed_at=None,
            ),
            notify=True,
        )

    def release(
        self,
        exchange: str,
        exchange_order_id: str,
        *,
        closed_at: datetime,
    ) -> None:
        """Mark an order closed. The mapping stays resolvable until retention."""

        ex = _norm_exchange(exchange)
        order_id = str(exchange_order_id or "").strip()
        if not ex or not order_id:
            return
        existing = self._records.get(_order_key(ex, order_id))
        if existing is None or existing.closed_at is not None:
            return
        self._store(
            AttributionRecord(
                exchange=existing.exchange,
                exchange_order_id=existing.exchange_order_id,
                actor_id=existing.actor_id,
                client_order_id=existing.client_order_id,
                wire_client_order_id=existing.wire_client_order_id,
                closed_at=closed_at,
            ),
            notify=True,
        )

    def apply_loaded(self, records: Iterable[AttributionRecord]) -> None:
        """Install mappings restored from storage without enqueueing them again."""

        for record in records:
            if not record.exchange or not record.exchange_order_id or not record.actor_id:
                continue
            self._store(record, notify=False)

    def apply_client_aliases(self, exchange: str, aliases: Iterable[tuple[str, str]]) -> None:
        """Install client-id lookups that are not tied to one exchange order id."""

        ex = _norm_exchange(exchange)
        if not ex:
            return
        for client_order_id, actor_id in aliases:
            self._index_client_only(ex, _norm_actor(actor_id), str(client_order_id or "").strip(), "")

    def has_order(self, exchange: str, exchange_order_id: str) -> bool:
        ex = _norm_exchange(exchange)
        order_id = str(exchange_order_id or "").strip()
        if not ex or not order_id:
            return False
        return _order_key(ex, order_id) in self._records

    def forget_closed(
        self,
        exchange: str,
        exchange_order_ids: Sequence[str],
        *,
        older_than: datetime,
    ) -> None:
        """Drop mappings whose close time is strictly before ``older_than``."""

        ex = _norm_exchange(exchange)
        if not ex:
            return
        for raw_order_id in exchange_order_ids:
            order_id = str(raw_order_id or "").strip()
            key = _order_key(ex, order_id)
            record = self._records.get(key)
            if record is None or record.closed_at is None or record.closed_at >= older_than:
                continue
            self._drop(key)

    def resolve_actor_id(
        self,
        exchange: str,
        *,
        exchange_order_id: str = "",
        client_order_id: str = "",
    ) -> str:
        ex = _norm_exchange(exchange)
        order_id = str(exchange_order_id or "").strip()
        if order_id:
            record = self._records.get(_order_key(ex, order_id))
            if record is not None and record.actor_id:
                return record.actor_id
        client_id = str(client_order_id or "").strip()
        if client_id:
            actor = self._by_client_order_id.get(_client_key(ex, client_id))
            if actor:
                return actor
            actor = actor_id_from_runtime_client_order_id(client_id)
            if actor:
                return actor
        return ""

    def enrich_fill(self, fill: Fill, *, client_order_id: str = "") -> Fill:
        actor = (fill.actor_id or "").strip()
        if actor:
            return fill
        resolved = self.resolve_actor_id(
            fill.exchange,
            exchange_order_id=fill.exchange_order_id,
            client_order_id=client_order_id,
        )
        if not resolved:
            return fill
        return Fill(
            exchange_order_id=fill.exchange_order_id,
            actor_id=resolved,
            exchange=fill.exchange,
            symbol=fill.symbol,
            side=fill.side,
            price=fill.price,
            qty=fill.qty,
            fee=fill.fee,
            timestamp_ms=fill.timestamp_ms,
        )

    def _store(self, record: AttributionRecord, *, notify: bool) -> None:
        key = _order_key(record.exchange, record.exchange_order_id)
        previous = self._records.get(key)
        if previous == record:
            return
        if (
            previous is not None
            and previous.closed_at is not None
            and record.closed_at is not None
        ):
            return
        self._records[key] = record
        self._reindex(key, previous, record)
        if notify and self._sink is not None:
            self._sink.record(record)

    def _index_client_only(
        self,
        exchange: str,
        actor_id: str,
        client_order_id: str,
        wire_client_order_id: str,
    ) -> None:
        for client_id in (client_order_id, wire_client_order_id):
            if not client_id:
                continue
            self._by_client_order_id[_client_key(exchange, client_id)] = actor_id

    def _reindex(
        self,
        order_key: str,
        previous: AttributionRecord | None,
        record: AttributionRecord,
    ) -> None:
        if previous is not None:
            for client_id in (previous.client_order_id, previous.wire_client_order_id):
                if not client_id:
                    continue
                client_key = _client_key(previous.exchange, client_id)
                if self._client_owner.get(client_key) == order_key:
                    self._client_owner.pop(client_key, None)
                    self._by_client_order_id.pop(client_key, None)
        for client_id in (record.client_order_id, record.wire_client_order_id):
            if not client_id:
                continue
            client_key = _client_key(record.exchange, client_id)
            self._by_client_order_id[client_key] = record.actor_id
            self._client_owner[client_key] = order_key

    def _drop(self, order_key: str) -> None:
        record = self._records.pop(order_key, None)
        if record is None:
            return
        for client_id in (record.client_order_id, record.wire_client_order_id):
            if not client_id:
                continue
            client_key = _client_key(record.exchange, client_id)
            if self._client_owner.get(client_key) == order_key:
                self._client_owner.pop(client_key, None)
                self._by_client_order_id.pop(client_key, None)


def register_open_order_attribution(
    registry: OrderAttributionRegistry,
    exchange: str,
    open_order: OpenOrder,
) -> None:
    """Seed registry mappings from a live open order when actor id is known."""

    actor = (open_order.actor_id or "").strip()
    if not actor:
        actor = actor_id_from_runtime_client_order_id(open_order.order.client_order_id)
    if not actor:
        return
    client_order_id = open_order.order.client_order_id
    registry.register(
        exchange,
        open_order.exchange_order_id,
        actor,
        client_order_id=client_order_id,
        wire_client_order_id=wire_client_order_id_for_exchange(exchange, client_order_id),
    )


def read_legacy_attribution(
    exchange: str,
) -> tuple[tuple[AttributionRecord, ...], tuple[tuple[str, str], ...]]:
    """Read the retired JSON cache.

    Returns order mappings and ``(client_order_id, actor_id)`` aliases.
    Missing or unreadable files yield empty results.
    """

    path = _attribution_cache_path(exchange)
    if not path.is_file():
        return (), ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (), ()
    wanted = _norm_exchange(exchange)
    if not wanted or not isinstance(payload, dict):
        return (), ()
    prefix = f"{wanted}:"
    records: list[AttributionRecord] = []
    by_order = payload.get("by_exchange_order_id")
    if isinstance(by_order, dict):
        for raw_key, raw_actor in by_order.items():
            key = str(raw_key).strip()
            actor = str(raw_actor).strip()
            if not key.startswith(prefix) or not actor:
                continue
            order_id = key[len(prefix) :]
            if not order_id:
                continue
            records.append(
                AttributionRecord(
                    exchange=wanted,
                    exchange_order_id=order_id,
                    actor_id=actor,
                )
            )
    aliases: list[tuple[str, str]] = []
    by_client = payload.get("by_client_order_id")
    if isinstance(by_client, dict):
        for raw_key, raw_actor in by_client.items():
            key = str(raw_key).strip()
            actor = str(raw_actor).strip()
            if not key.startswith(prefix) or not actor:
                continue
            client_id = key[len(prefix) :]
            if client_id:
                aliases.append((client_id, actor))
    return tuple(records), tuple(aliases)


def read_legacy_attribution_records(exchange: str) -> tuple[AttributionRecord, ...]:
    records, _aliases = read_legacy_attribution(exchange)
    return records


def load_attribution_registry(exchange: str) -> OrderAttributionRegistry:
    """Load the legacy JSON cache into a memory-only registry.

    This does not enable writes. Runtime startup uses the attribution service,
    which reads Postgres and imports this file only when the table is empty.
    """

    registry = OrderAttributionRegistry()
    records, aliases = read_legacy_attribution(exchange)
    registry.apply_loaded(records)
    registry.apply_client_aliases(exchange, aliases)
    return registry
