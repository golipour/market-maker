from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm.runtime.config import (
    BusRuntimeConfig,
    ObservabilityRuntimeConfig,
    SchedulerRuntimeConfig,
    _bus_from_mapping,
    _exchange_market_payload,
    _observability_from_mapping,
    _scheduler_from_mapping,
)


@dataclass(frozen=True, slots=True)
class IngestorFeedConfig:
    """One shared public market-data feed published to the runtime bus."""

    exchange: str
    symbol: str
    interval_ms: int
    mode: str = "rest"
    depth_limit: int = 10
    trades_limit: int = 100
    poll_interval_ms: int = 1000

    def normalized_exchange(self) -> str:
        return self.exchange.lower().strip()

    def market_event_topic(self) -> str:
        return f"market:events:{self.normalized_exchange()}:{self.symbol}:{self.interval_ms}"

    def market_error_topic(self) -> str:
        return f"market:errors:{self.normalized_exchange()}:{self.symbol}"

    def raw_trade_topic(self) -> str:
        return f"market:raw:{self.normalized_exchange()}:{self.symbol}:trades"

    def raw_orderbook_topic(self) -> str:
        return f"market:raw:{self.normalized_exchange()}:{self.symbol}:orderbook"

    def feed_key(self) -> str:
        return f"{self.normalized_exchange()}:{self.symbol}:{self.interval_ms}"

    def poll_source_key(self) -> str:
        return f"{self.normalized_exchange()}:{self.symbol}"


def group_feeds_by_poll_source(
    feeds: tuple[IngestorFeedConfig, ...],
) -> dict[str, tuple[IngestorFeedConfig, ...]]:
    """Group frame-interval feeds that share one exchange/symbol poll source."""

    grouped: dict[str, list[IngestorFeedConfig]] = {}
    for feed in feeds:
        grouped.setdefault(feed.poll_source_key(), []).append(feed)
    result: dict[str, tuple[IngestorFeedConfig, ...]] = {}
    for key, rows in grouped.items():
        primary = rows[0]
        poll_signature = (
            primary.mode,
            primary.depth_limit,
            primary.trades_limit,
            primary.poll_interval_ms,
        )
        for row in rows[1:]:
            other = (row.mode, row.depth_limit, row.trades_limit, row.poll_interval_ms)
            if other != poll_signature:
                raise ValueError(
                    f"conflicting poll settings for {key}: "
                    f"{poll_signature} vs {other}; use one market source per exchange/symbol"
                )
        result[key] = tuple(rows)
    return result


@dataclass(frozen=True, slots=True)
class IngestorRuntimeConfig:
    feeds: tuple[IngestorFeedConfig, ...]
    bus: BusRuntimeConfig = field(default_factory=BusRuntimeConfig)
    observability: ObservabilityRuntimeConfig = field(default_factory=ObservabilityRuntimeConfig)
    scheduler: SchedulerRuntimeConfig = field(default_factory=SchedulerRuntimeConfig)

    def __post_init__(self) -> None:
        if not self.feeds:
            raise ValueError("ingestor config requires at least one feed")
        if self.bus.kind != "redis_streams":
            raise ValueError("ingestor requires bus.kind='redis_streams'")

    def as_contract(self) -> dict[str, Any]:
        return {
            "feeds": [
                {
                    "exchange": feed.normalized_exchange(),
                    "symbol": feed.symbol,
                    "interval_ms": feed.interval_ms,
                    "market_event_topic": feed.market_event_topic(),
                    "requested_public_market_transport": feed.mode,
                }
                for feed in self.feeds
            ],
            "bus": {
                "kind": self.bus.kind,
                "redis_url": self.bus.redis_url,
                "stream_prefix": self.bus.stream_prefix,
                "consumer_group": self.bus.consumer_group,
                "consumer_name": self.bus.consumer_name,
            },
            "observability": {
                "service_name": self.observability.service_name,
                "metrics_enabled": self.observability.metrics_enabled,
                "metrics_port": self.observability.metrics_port,
            },
        }


def load_ingestor_config(path: str | Path) -> IngestorRuntimeConfig:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("ingestor config file must contain a JSON object")
    return ingestor_config_from_mapping(payload)


def load_ingestor_config_from_registry(
    path: str | Path,
    *,
    database_url: str | None = None,
) -> IngestorRuntimeConfig:
    """Load ingestor config, optionally sourcing ``feeds`` from the registry."""

    from mm.registry.feeds import merge_registry_feeds_into_mapping
    from mm.registry.store import build_registry_store, registry_database_url

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("ingestor config file must contain a JSON object")
    if payload.get("load_feeds_from_registry"):
        url = (database_url or registry_database_url() or "").strip()
        if not url:
            raise ValueError("load_feeds_from_registry requires MM_DATABASE_URL")
        store = build_registry_store(database_url=url)
        payload = merge_registry_feeds_into_mapping(payload, store)
    return ingestor_config_from_mapping(payload)


def ingestor_config_from_mapping(payload: dict[str, Any]) -> IngestorRuntimeConfig:
    feeds_payload = payload.get("feeds")
    if not isinstance(feeds_payload, list) or not feeds_payload:
        raise ValueError("ingestor config requires a non-empty feeds array")
    feeds = tuple(_feed_from_mapping(payload, row) for row in feeds_payload)
    return IngestorRuntimeConfig(
        feeds=feeds,
        bus=_bus_from_mapping(payload.get("bus")),
        observability=_observability_from_mapping(payload.get("observability")),
        scheduler=_scheduler_from_mapping(payload.get("scheduler")),
    )


def _feed_from_mapping(payload: dict[str, Any], feed_payload: Any) -> IngestorFeedConfig:
    if not isinstance(feed_payload, dict):
        raise ValueError("each feeds entry must be an object")
    exchange = str(feed_payload.get("exchange", "")).lower().strip()
    symbol = str(feed_payload.get("symbol", "")).strip()
    if not exchange:
        raise ValueError("feeds.exchange is required")
    if not symbol:
        raise ValueError("feeds.symbol is required")
    market_payload = _feed_market_defaults(payload, exchange=exchange, symbol=symbol)
    interval_ms = int(feed_payload.get("interval_ms", market_payload.get("interval_ms", 5000)))
    if interval_ms <= 0:
        raise ValueError("feeds.interval_ms must be positive")
    return IngestorFeedConfig(
        exchange=exchange,
        symbol=symbol,
        interval_ms=interval_ms,
        mode=str(feed_payload.get("mode", market_payload.get("mode", "rest"))),
        depth_limit=int(feed_payload.get("depth_limit", market_payload.get("depth_limit", 10))),
        trades_limit=int(feed_payload.get("trades_limit", market_payload.get("trades_limit", 100))),
        poll_interval_ms=int(
            feed_payload.get("poll_interval_ms", market_payload.get("poll_interval_ms", 1000))
        ),
    )


def _feed_market_defaults(payload: dict[str, Any], *, exchange: str, symbol: str) -> dict[str, Any]:
    exchanges = payload.get("exchanges")
    if isinstance(exchanges, dict) and exchange in exchanges:
        return _exchange_market_payload(payload, exchange=exchange, symbol=symbol)
    return {"exchange": exchange, "symbol": symbol}
