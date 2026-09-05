from __future__ import annotations

from mm.bus.base import AbstractEventBus
from mm.bus.memory import InMemoryBus
from mm.bus.redis_streams import RedisStreamsBus
from mm.runtime.config import BusRuntimeConfig


def build_event_bus(config: BusRuntimeConfig, *, run_id: str = "") -> AbstractEventBus:
    if config.kind == "memory":
        return InMemoryBus(
            max_queue_size=config.max_queue_size,
            backpressure_policy=config.backpressure_policy,
        )
    if config.kind == "redis_streams":
        return RedisStreamsBus(
            config.redis_url,
            stream_prefix=config.stream_prefix,
            maxlen=config.max_stream_length,
            envelope_size_max_bytes=config.envelope_size_max_bytes,
            read_block_ms=config.read_block_ms,
            run_id=run_id,
            max_queue_size=config.max_queue_size,
            backpressure_policy=config.backpressure_policy,
            max_connections=config.max_connections,
        )
    raise ValueError(f"unsupported bus kind: {config.kind}")
