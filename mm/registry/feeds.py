"""Load ingestor feeds from the strategy registry."""

from __future__ import annotations

from mm.registry.models import MarketFeedRecord
from mm.registry.store import RegistryStore
from mm.runtime.ingestor_config import IngestorFeedConfig


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
) -> tuple[IngestorFeedConfig, ...]:
    feeds = store.list_market_feeds()
    if active_only:
        active_sources = {
            row.source_id for row in store.list_market_sources() if row.status == "active"
        }
        feeds = [
            row for row in feeds if row.status == "active" and row.source_id in active_sources
        ]
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
    feeds = ingestor_feeds_from_registry(store)
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
