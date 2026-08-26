"""Load ingestor feeds from the strategy registry."""

from __future__ import annotations

from typing import Any

from mm.registry.models import MarketFeedRecord
from mm.registry.store import RegistryStore
from mm.runtime.ingestor_config import IngestorFeedConfig


def _normalize_exclude_exchanges(exclude: Any) -> frozenset[str]:
    if exclude is None:
        return frozenset()
    if isinstance(exclude, str):
        items = [exclude]
    elif isinstance(exclude, (list, tuple, set, frozenset)):
        items = list(exclude)
    else:
        raise ValueError("exclude_exchanges must be a string or array of exchange names")
    return frozenset(str(item).lower().strip() for item in items if str(item).strip())


def market_feed_record_to_ingestor_feed(record: MarketFeedRecord) -> IngestorFeedConfig:
    return IngestorFeedConfig(
        exchange=record.exchange,
        symbol=record.symbol,
        interval_ms=record.interval_ms,
        mode=record.mode,
        depth_limit=record.depth_limit,
        trades_limit=record.trades_limit,
        poll_interval_ms=record.poll_interval_ms,
    )


def ingestor_feeds_from_registry(
    store: RegistryStore,
    *,
    active_only: bool = True,
    exclude_exchanges: frozenset[str] | None = None,
) -> tuple[IngestorFeedConfig, ...]:
    feeds = store.list_market_feeds()
    if active_only:
        active_sources = {
            row.source_id for row in store.list_market_sources() if row.status == "active"
        }
        feeds = [
            row for row in feeds if row.status == "active" and row.source_id in active_sources
        ]
    excluded = exclude_exchanges or frozenset()
    if excluded:
        feeds = [row for row in feeds if row.exchange.lower() not in excluded]
    if not feeds:
        raise ValueError("registry has no market feeds configured")
    return tuple(market_feed_record_to_ingestor_feed(row) for row in feeds)


def merge_registry_feeds_into_mapping(
    payload: dict,
    store: RegistryStore | None,
) -> dict:
    """When ``load_feeds_from_registry`` is true, replace ``feeds`` from Postgres."""

    if not payload.get("load_feeds_from_registry"):
        return payload
    if store is None:
        raise ValueError("load_feeds_from_registry requires a registry store")
    exclude_exchanges = _normalize_exclude_exchanges(payload.get("exclude_exchanges"))
    feeds = ingestor_feeds_from_registry(store, exclude_exchanges=exclude_exchanges)
    merged = dict(payload)
    merged["feeds"] = [
        {
            "exchange": feed.exchange,
            "symbol": feed.symbol,
            "interval_ms": feed.interval_ms,
            "mode": feed.mode,
            "depth_limit": feed.depth_limit,
            "trades_limit": feed.trades_limit,
            "poll_interval_ms": feed.poll_interval_ms,
        }
        for feed in feeds
    ]
    return merged


def merge_registry_actors_into_mapping(
    payload: dict,
    store: RegistryStore | None,
) -> dict:
    """Union configured persistence actors with active registry strategy actor ids."""

    if not payload.get("load_feeds_from_registry"):
        return payload
    if store is None:
        raise ValueError("load_feeds_from_registry requires a registry store")
    existing = payload.get("actors")
    if isinstance(existing, str):
        configured = [existing.strip()] if existing.strip() else []
    elif isinstance(existing, list):
        configured = [str(item).strip() for item in existing if str(item).strip()]
    else:
        configured = []
    from_registry = [
        str(row.actor_id).strip()
        for row in store.list_strategies()
        if row.status == "active" and str(row.actor_id).strip()
    ]
    merged = dict(payload)
    merged["actors"] = list(dict.fromkeys([*configured, *from_registry]))
    return merged
