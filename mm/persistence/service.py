from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from mm.bus.base import AbstractEventBus
from mm.bus.factory import build_event_bus
from mm.env_loader import load_mm_env
from mm.observability import configure_observability, log_event
from mm.persistence.service_config import PersistenceServiceConfig
from mm.runtime.config import PersistenceRuntimeConfig


def build_persistence_writers(
    *,
    bus: AbstractEventBus,
    persistence: PersistenceRuntimeConfig,
    topics: tuple[str, ...],
    run_id: str = "",
) -> tuple[object, ...]:
    """Wire Postgres and Parquet writers for the given topic subscription set."""

    if not persistence.enabled:
        log_event("persistence_disabled", {"reason": "config"})
        return ()
    daemons: list[object] = []
    database_url = persistence.resolved_database_url()
    if database_url:
        from mm.persistence.postgres import PostgresEventWriter

        daemons.append(
            PostgresEventWriter(
                bus=bus,
                topics=topics,
                database_url=database_url,
                batch_size=persistence.batch_size,
                flush_interval_ms=persistence.flush_interval_ms,
                run_id=run_id,
            )
        )
    else:
        log_event(
            "persistence_sink_missing",
            {
                "sink": "postgres",
                "database_url_env": persistence.database_url_env,
                "message": "persistence.enabled is true, but no database URL resolved",
            },
        )
    parquet_root = persistence.resolved_parquet_root()
    if parquet_root:
        from mm.persistence.parquet_writer import ParquetEventWriter

        daemons.append(
            ParquetEventWriter(
                bus=bus,
                topics=topics,
                root_path=parquet_root,
                batch_size=persistence.batch_size,
                flush_interval_ms=persistence.flush_interval_ms,
                run_id=run_id,
            )
        )
    if not daemons:
        log_event(
            "persistence_no_writers_configured",
            {
                "database_url_env": persistence.database_url_env,
                "parquet_root_env": persistence.parquet_root_env,
                "topic_count": len(topics),
            },
        )
    else:
        log_event(
            "persistence_writers_configured",
            {
                "writers": [type(daemon).__name__ for daemon in daemons],
                "topic_count": len(topics),
            },
        )
    return tuple(daemons)


@dataclass(slots=True)
class PersistenceComponents:
    """Standalone persistence subscriber process."""

    config: PersistenceServiceConfig
    bus: AbstractEventBus
    writers: tuple[object, ...] = ()
    _started: bool = False

    async def start(self) -> None:
        if self._started:
            return
        for writer in self.writers:
            start = getattr(writer, "start", None)
            if start is not None:
                await start()
        self._started = True

    async def stop(self) -> None:
        for writer in reversed(self.writers):
            stop = getattr(writer, "stop", None)
            if stop is not None:
                await stop()
        await self.bus.close()
        self._started = False


async def build_persistence_service(
    config: PersistenceServiceConfig,
    *,
    bus: AbstractEventBus | None = None,
) -> PersistenceComponents:
    load_mm_env()
    configure_observability(config.observability)
    if config.bus.kind != "redis_streams":
        raise ValueError("persistence service requires bus.kind='redis_streams'")

    topics = config.effective_topics
    runtime_bus = bus or build_event_bus(config.bus, run_id=config.observability.run_id)
    writers = build_persistence_writers(
        bus=runtime_bus,
        persistence=config.persistence,
        topics=topics,
        run_id=config.observability.run_id,
    )
    log_event(
        "persistence_service_build_complete",
        {
            "writer_count": len(writers),
            "writers": [type(writer).__name__ for writer in writers],
            "topic_count": len(topics),
            "consumer_group": config.bus.consumer_group,
            "consumer_name": config.bus.consumer_name,
            "bus_kind": config.bus.kind,
            "stream_prefix": config.bus.stream_prefix,
        },
    )
    return PersistenceComponents(config=config, bus=runtime_bus, writers=writers)
