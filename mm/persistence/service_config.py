from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm.core.topics import RuntimeTopics
from mm.runtime.config import (
    BusRuntimeConfig,
    ObservabilityRuntimeConfig,
    PersistenceRuntimeConfig,
    _bus_from_mapping,
    _observability_from_mapping,
    _persistence_from_mapping,
)
from mm.runtime.ingestor_config import IngestorFeedConfig, _feed_from_mapping, _feed_market_defaults


FLEET_CORE_PERSISTENCE_TOPICS: tuple[str, ...] = (
    "ledger:snapshot",
    "strategy:decisions",
    "strategy:actions",
    "execution:errors",
    "risk:events",
    "admin:events",
    "portfolio:allocations",
)


@dataclass(frozen=True, slots=True)
class JointFrameSubscription:
    """One joint strategy frame stream to persist for replay lineage."""

    exchange: str
    symbol: str
    interval_ms: int

    def normalized_exchange(self) -> str:
        return self.exchange.lower().strip()

    def topic(self) -> str:
        return RuntimeTopics.joint_strategy_frame(
            self.normalized_exchange(),
            self.symbol,
            self.interval_ms,
        )


def exchange_persistence_topics(exchange: str) -> tuple[str, ...]:
    normalized = exchange.lower().strip()
    return (
        RuntimeTopics.execution_orders(normalized),
        RuntimeTopics.execution_state(normalized),
        RuntimeTopics.private_snapshot(normalized),
        f"private:state:{normalized}:fills",
        f"private:state:{normalized}:errors",
        f"ledger:diagnostics:mismatch:{normalized}",
        RuntimeTopics.ledger_snapshot_for(normalized),
    )


def actor_persistence_topics(actor_id: str) -> tuple[str, ...]:
    actor = actor_id.strip()
    return (
        f"strategy:target_books:{actor}",
        f"strategy:diagnostics:{actor}",
    )


def fleet_persistence_topics(
    *,
    feeds: tuple[IngestorFeedConfig, ...] = (),
    exchanges: tuple[str, ...] = (),
    actors: tuple[str, ...] = (),
    joint_frames: tuple[JointFrameSubscription, ...] = (),
) -> tuple[str, ...]:
    """Derive fleet-wide persistence topics when no explicit override is configured."""

    topics: list[str] = list(FLEET_CORE_PERSISTENCE_TOPICS)
    for feed in feeds:
        topics.append(feed.market_event_topic())
    resolved_joint_frames = joint_frames or tuple(
        JointFrameSubscription(
            feed.normalized_exchange(),
            feed.symbol,
            feed.interval_ms,
        )
        for feed in feeds
    )
    for joint_frame in resolved_joint_frames:
        topics.append(joint_frame.topic())
    resolved_exchanges = exchanges or tuple(
        dict.fromkeys(feed.normalized_exchange() for feed in feeds)
    )
    for exchange in resolved_exchanges:
        topics.extend(exchange_persistence_topics(exchange))
    for actor in actors:
        topics.extend(actor_persistence_topics(actor))
    return tuple(dict.fromkeys(topics))


@dataclass(frozen=True, slots=True)
class PersistenceServiceConfig:
    """Standalone persistence subscriber configuration."""

    bus: BusRuntimeConfig
    persistence: PersistenceRuntimeConfig
    observability: ObservabilityRuntimeConfig = field(default_factory=ObservabilityRuntimeConfig)
    feeds: tuple[IngestorFeedConfig, ...] = ()
    exchanges: tuple[str, ...] = ()
    actors: tuple[str, ...] = ()
    joint_frames: tuple[JointFrameSubscription, ...] = ()

    def __post_init__(self) -> None:
        if self.bus.kind != "redis_streams":
            raise ValueError("persistence service requires bus.kind='redis_streams'")
        if not self.persistence.enabled:
            raise ValueError("persistence service requires persistence.enabled=true")

    @property
    def effective_topics(self) -> tuple[str, ...]:
        if self.persistence.topics:
            return self.persistence.topics
        return fleet_persistence_topics(
            feeds=self.feeds,
            exchanges=self.exchanges,
            actors=self.actors,
            joint_frames=self.joint_frames,
        )

    def as_contract(self) -> dict[str, Any]:
        return {
            "feeds": [
                {
                    "exchange": feed.normalized_exchange(),
                    "symbol": feed.symbol,
                    "interval_ms": feed.interval_ms,
                    "market_event_topic": feed.market_event_topic(),
                }
                for feed in self.feeds
            ],
            "exchanges": list(self.exchanges),
            "actors": list(self.actors),
            "joint_frames": [
                {
                    "exchange": frame.normalized_exchange(),
                    "symbol": frame.symbol,
                    "interval_ms": frame.interval_ms,
                    "topic": frame.topic(),
                }
                for frame in self.joint_frames
            ],
            "effective_topics": list(self.effective_topics),
            "bus": {
                "kind": self.bus.kind,
                "redis_url": self.bus.redis_url,
                "stream_prefix": self.bus.stream_prefix,
                "consumer_group": self.bus.consumer_group,
                "consumer_name": self.bus.consumer_name,
            },
            "persistence": {
                "enabled": self.persistence.enabled,
                "topic_count": len(self.effective_topics),
                "batch_size": self.persistence.batch_size,
                "flush_interval_ms": self.persistence.flush_interval_ms,
            },
            "observability": {
                "service_name": self.observability.service_name,
                "metrics_enabled": self.observability.metrics_enabled,
                "metrics_port": self.observability.metrics_port,
            },
        }


def load_persistence_service_config(path: str | Path) -> PersistenceServiceConfig:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("persistence config file must contain a JSON object")
    return persistence_service_config_from_mapping(payload)


def persistence_service_config_from_mapping(payload: dict[str, Any]) -> PersistenceServiceConfig:
    persistence_payload = payload.get("persistence")
    if not isinstance(persistence_payload, dict):
        raise ValueError("persistence service requires a persistence config object")
    persistence = _persistence_from_mapping(
        {
            **persistence_payload,
            "enabled": persistence_payload.get("enabled", True),
        }
    )

    bus_payload = payload.get("bus")
    if not isinstance(bus_payload, dict):
        bus_payload = {}
    bus_payload = {
        **bus_payload,
        "kind": bus_payload.get("kind", "redis_streams"),
        "consumer_group": bus_payload.get("consumer_group", "persistence"),
        "consumer_name": bus_payload.get("consumer_name", "persistence-1"),
    }

    feeds = _feeds_from_mapping(payload)
    exchanges = _exchanges_from_mapping(payload)
    actors = _actors_from_mapping(payload)
    joint_frames = _joint_frames_from_mapping(payload)

    return PersistenceServiceConfig(
        bus=_bus_from_mapping(bus_payload),
        persistence=persistence,
        observability=_observability_from_mapping(payload.get("observability")),
        feeds=feeds,
        exchanges=exchanges,
        actors=actors,
        joint_frames=joint_frames,
    )


def _feeds_from_mapping(payload: dict[str, Any]) -> tuple[IngestorFeedConfig, ...]:
    feeds_payload = payload.get("feeds")
    if not isinstance(feeds_payload, list) or not feeds_payload:
        return ()
    return tuple(_feed_from_mapping(payload, row) for row in feeds_payload)


def _exchanges_from_mapping(payload: dict[str, Any]) -> tuple[str, ...]:
    exchanges_payload = payload.get("exchanges")
    if isinstance(exchanges_payload, str):
        return (exchanges_payload.lower().strip(),)
    if not isinstance(exchanges_payload, list):
        return ()
    return tuple(
        dict.fromkeys(str(exchange).lower().strip() for exchange in exchanges_payload if str(exchange).strip())
    )


def _actors_from_mapping(payload: dict[str, Any]) -> tuple[str, ...]:
    actors_payload = payload.get("actors")
    if isinstance(actors_payload, str):
        return (actors_payload.strip(),)
    if not isinstance(actors_payload, list):
        return ()
    return tuple(dict.fromkeys(str(actor).strip() for actor in actors_payload if str(actor).strip()))


def _joint_frames_from_mapping(payload: dict[str, Any]) -> tuple[JointFrameSubscription, ...]:
    frames_payload = payload.get("joint_frames")
    if not isinstance(frames_payload, list) or not frames_payload:
        return ()
    frames: list[JointFrameSubscription] = []
    for row in frames_payload:
        if not isinstance(row, dict):
            raise ValueError("each joint_frames entry must be an object")
        exchange = str(row.get("exchange", "")).lower().strip()
        symbol = str(row.get("symbol", "")).strip()
        if not exchange:
            raise ValueError("joint_frames.exchange is required")
        if not symbol:
            raise ValueError("joint_frames.symbol is required")
        market_payload = _feed_market_defaults(payload, exchange=exchange, symbol=symbol)
        interval_ms = int(row.get("interval_ms", market_payload.get("interval_ms", 5000)))
        if interval_ms <= 0:
            raise ValueError("joint_frames.interval_ms must be positive")
        frames.append(JointFrameSubscription(exchange=exchange, symbol=symbol, interval_ms=interval_ms))
    return tuple(frames)
