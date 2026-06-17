"""Resolve ingestor market-data settings from exchange registry config."""

from __future__ import annotations

from typing import Any

from mm.registry.models import ExchangeRecord, MarketFeedRecord, MarketSourceRecord

DEFAULT_MARKET_DATA_MODE = "rest"
DEFAULT_MARKET_DATA_DEPTH_LIMIT = 10
DEFAULT_MARKET_DATA_TRADES_LIMIT = 100
DEFAULT_MARKET_DATA_POLL_INTERVAL_MS = 1000


def market_data_settings_from_mapping(config: dict[str, Any] | None) -> dict[str, Any]:
    """Read venue-wide public market polling defaults from an exchange config object."""

    market_data = {}
    if isinstance(config, dict):
        raw = config.get("market_data")
        if isinstance(raw, dict):
            market_data = raw
    return {
        "mode": str(market_data.get("mode", DEFAULT_MARKET_DATA_MODE)),
        "depth_limit": max(1, int(market_data.get("depth_limit", DEFAULT_MARKET_DATA_DEPTH_LIMIT))),
        "trades_limit": max(1, int(market_data.get("trades_limit", DEFAULT_MARKET_DATA_TRADES_LIMIT))),
        "poll_interval_ms": max(
            1,
            int(market_data.get("poll_interval_ms", DEFAULT_MARKET_DATA_POLL_INTERVAL_MS)),
        ),
    }


def market_data_settings_from_exchange(record: ExchangeRecord) -> dict[str, Any]:
    return market_data_settings_from_mapping(record.config)


def market_source_as_dict(
    exchange: ExchangeRecord,
    record: MarketSourceRecord,
) -> dict[str, Any]:
    payload = record.as_dict()
    payload.update(market_data_settings_from_exchange(exchange))
    return payload


def enrich_market_feed_record(
    *,
    source: MarketSourceRecord,
    exchange: ExchangeRecord,
    record: MarketFeedRecord,
) -> MarketFeedRecord:
    resolved = market_data_settings_from_exchange(exchange)
    return MarketFeedRecord(
        feed_id=record.feed_id,
        source_id=record.source_id,
        interval_ms=record.interval_ms,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        exchange=source.exchange,
        symbol=source.symbol,
        mode=str(resolved["mode"]),
        depth_limit=int(resolved["depth_limit"]),
        trades_limit=int(resolved["trades_limit"]),
        poll_interval_ms=int(resolved["poll_interval_ms"]),
    )
