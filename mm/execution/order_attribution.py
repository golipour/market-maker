"""Map exchange order ids and client order ids back to strategy actor ids."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from mm.core.models.execution import Fill
from mm.core.models.private import OpenOrder


def _norm_exchange(exchange: str) -> str:
    return exchange.lower().strip()


def _norm_actor(actor_id: str) -> str:
    return actor_id.strip()


def _attribution_cache_dir() -> Path:
    """Directory for the on-disk attribution cache.

    The runtime image runs as ``marketmaker`` with cwd ``/app``, which is not
    writable. The image owns ``/data``; relative ``data/attribution`` would
    try to create ``/app/data`` and fail with EACCES.
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


@dataclass(slots=True)
class OrderAttributionRegistry:
    """In-memory placement registry used to stamp ``actor_id`` on exchange fills."""

    _by_exchange_order_id: dict[str, str] = field(default_factory=dict)
    _by_client_order_id: dict[str, str] = field(default_factory=dict)
    _persist_exchange: str | None = None

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
        if order_id:
            self._by_exchange_order_id[f"{ex}:{order_id}"] = actor
        for key in (client_order_id, wire_client_order_id):
            text = str(key or "").strip()
            if text:
                self._by_client_order_id[f"{ex}:{text}"] = actor
        if self._persist_exchange and self._persist_exchange == ex:
            if not save_attribution_registry(ex, self):
                self._persist_exchange = None

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
            actor = self._by_exchange_order_id.get(f"{ex}:{order_id}")
            if actor:
                return actor
        client_id = str(client_order_id or "").strip()
        if client_id:
            actor = self._by_client_order_id.get(f"{ex}:{client_id}")
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


def load_attribution_registry(exchange: str) -> OrderAttributionRegistry:
    path = _attribution_cache_path(exchange)
    registry = OrderAttributionRegistry()
    registry._persist_exchange = _norm_exchange(exchange)
    if not path.exists():
        return registry
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return registry
    if isinstance(payload.get("by_exchange_order_id"), dict):
        registry._by_exchange_order_id = {
            str(key): str(value)
            for key, value in payload["by_exchange_order_id"].items()
            if str(key).strip() and str(value).strip()
        }
    if isinstance(payload.get("by_client_order_id"), dict):
        registry._by_client_order_id = {
            str(key): str(value)
            for key, value in payload["by_client_order_id"].items()
            if str(key).strip() and str(value).strip()
        }
    return registry


def save_attribution_registry(exchange: str, registry: OrderAttributionRegistry) -> bool:
    """Persist attribution mappings. Returns False if the cache cannot be written.

    This cache is restart convenience only. A disk error must not fail a live
    place/cancel that the exchange already accepted.
    """

    path = _attribution_cache_path(exchange)
    payload = {
        "by_exchange_order_id": dict(registry._by_exchange_order_id),
        "by_client_order_id": dict(registry._by_client_order_id),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        from mm.observability import log_event

        log_event(
            "order_attribution_cache_write_failed",
            {
                "exchange": _norm_exchange(exchange),
                "path": str(path),
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )
        return False
    return True
