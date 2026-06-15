from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from mm.aggregation.scheduler import IntervalScheduler
from mm.bus.base import AbstractEventBus
from mm.bus.factory import build_event_bus
from mm.core.clock import WallClock
from mm.env_loader import load_mm_env
from mm.exchanges.factory import PublicMarketDataAdapterSelection, select_public_market_data_adapter
from mm.ingestion.symbol_engine import SymbolEngine
from mm.observability import configure_observability, log_event
from mm.observability.subscribers import RuntimeObservabilitySubscriber
from mm.runtime.ingestor_config import (
    IngestorFeedConfig,
    IngestorRuntimeConfig,
    group_feeds_by_poll_source,
)
from mm.runtime.pipeline import build_symbol_engine


@dataclass(slots=True)
class IngestorComponents:
    """Shared public market-data runtime: SymbolEngine feeds only."""

    config: IngestorRuntimeConfig
    bus: AbstractEventBus
    scheduler: IntervalScheduler
    engines: dict[str, SymbolEngine]
    public_market_selections: dict[str, PublicMarketDataAdapterSelection] = field(default_factory=dict)
    observability_daemons: tuple[object, ...] = ()
    _scheduler_task: asyncio.Task | None = None
    _started: bool = False

    async def start(self) -> None:
        if self._started:
            return
        for daemon in self.observability_daemons:
            start = getattr(daemon, "start", None)
            if start is not None:
                await start()
        for engine in self.engines.values():
            await engine.start()
        self._scheduler_task = asyncio.create_task(self.scheduler.run())
        self._started = True

    async def stop(self) -> None:
        if self._scheduler_task is not None:
            self.scheduler.stop()
            self._scheduler_task.cancel()
            await asyncio.gather(self._scheduler_task, return_exceptions=True)
            self._scheduler_task = None
        for daemon in reversed(self.observability_daemons):
            stop = getattr(daemon, "stop", None)
            if stop is not None:
                await stop()
        for engine in reversed(tuple(self.engines.values())):
            await engine.stop()
        await self.bus.close()
        self._started = False


async def build_ingestor_components(
    config: IngestorRuntimeConfig,
    *,
    bus: AbstractEventBus | None = None,
    enable_observability: bool = True,
) -> IngestorComponents:
    load_mm_env()
    configure_observability(config.observability)
    if config.bus.kind != "redis_streams":
        raise ValueError("ingestor requires bus.kind='redis_streams'")

    runtime_bus = bus or build_event_bus(config.bus, run_id=config.observability.run_id)
    scheduler = IntervalScheduler(WallClock(), max_concurrency=config.scheduler.max_concurrency)
    engines: dict[str, SymbolEngine] = {}
    selections: dict[str, PublicMarketDataAdapterSelection] = {}
    poll_source_groups = group_feeds_by_poll_source(config.feeds)

    for source_key, feed_group in poll_source_groups.items():
        primary = feed_group[0]
        intervals_ms = sorted({feed.interval_ms for feed in feed_group})
        selection = select_public_market_data_adapter(
            primary.normalized_exchange(),
            primary.symbol,
            mode=primary.mode,
            depth_limit=primary.depth_limit,
            trades_limit=primary.trades_limit,
            poll_interval_ms=primary.poll_interval_ms,
        )
        engine = build_symbol_engine(
            exchange=primary.normalized_exchange(),
            symbol=primary.symbol,
            adapter=selection.adapter,
            bus=runtime_bus,
        )
        await engine.publish_on_intervals(scheduler, intervals_ms)
        engines[source_key] = engine
        selections[source_key] = selection

    log_event(
        "ingestor_build_complete",
        {
            "feed_count": len(config.feeds),
            "poll_source_count": len(poll_source_groups),
            "feeds": [feed.market_event_topic() for feed in config.feeds],
            "poll_sources": list(poll_source_groups.keys()),
            "bus_kind": config.bus.kind,
            "stream_prefix": config.bus.stream_prefix,
        },
    )

    observability_daemons: tuple[object, ...] = ()
    if enable_observability:
        observability_daemons = (_build_ingestor_observability(runtime_bus, config.feeds),)

    return IngestorComponents(
        config=config,
        bus=runtime_bus,
        scheduler=scheduler,
        engines=engines,
        public_market_selections=selections,
        observability_daemons=observability_daemons,
    )


def _build_ingestor_observability(
    bus: AbstractEventBus,
    feeds: tuple[IngestorFeedConfig, ...],
) -> RuntimeObservabilitySubscriber:
    topics: list[str] = []
    for feed in feeds:
        topics.extend(
            [
                feed.market_event_topic(),
                feed.market_error_topic(),
                feed.raw_trade_topic(),
                feed.raw_orderbook_topic(),
            ]
        )
    return RuntimeObservabilitySubscriber(bus=bus, topics=tuple(dict.fromkeys(topics)))
