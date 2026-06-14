"""Compare direct ingestor publish payloads with cross-process bus delivery.

Run:
  python -m mm.tools.compare_market_events --config mm/configs/market_maker.ingestor.example.json --samples 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from mm.bus.redis_streams import RedisStreamsBus
from mm.core.payload_schema import market_event_event, market_event_from_payload
from mm.ingestion.market_event_compare import compare_market_events, market_event_from_bus_payload
from mm.runtime.ingestor import build_ingestor_components
from mm.runtime.ingestor_config import IngestorFeedConfig, load_ingestor_config


async def run(args: argparse.Namespace) -> None:
    config = load_ingestor_config(args.config)
    feed = _select_feed(config.feeds, args.exchange, args.symbol, args.interval_ms)
    topic = feed.market_event_topic()

    if args.fake_redis:
        redis = _SharedFakeRedis()
        publisher_bus = RedisStreamsBus(
            stream_prefix=config.bus.stream_prefix,
            read_block_ms=config.bus.read_block_ms,
        )
        subscriber_bus = RedisStreamsBus(
            stream_prefix=config.bus.stream_prefix,
            read_block_ms=config.bus.read_block_ms,
        )
        publisher_bus._redis = redis
        subscriber_bus._redis = redis
    else:
        publisher_bus = RedisStreamsBus(
            url=config.bus.redis_url,
            stream_prefix=config.bus.stream_prefix,
            read_block_ms=config.bus.read_block_ms,
        )
        subscriber_bus = RedisStreamsBus(
            url=config.bus.redis_url,
            stream_prefix=config.bus.stream_prefix,
            read_block_ms=config.bus.read_block_ms,
        )

    published: list[dict[str, Any]] = []
    original_publish = publisher_bus.publish

    async def capture_publish(topic_name: str, event: Any) -> None:
        if topic_name == topic:
            payload = event if isinstance(event, dict) else market_event_event(event)
            if isinstance(payload, dict) and payload.get("payload_schema") == "mm.market_event":
                published.append(payload)
        await original_publish(topic_name, event)

    publisher_bus.publish = capture_publish  # type: ignore[method-assign]

    ingestor = await build_ingestor_components(config, bus=publisher_bus)
    queue = subscriber_bus.subscribe_queue(topic)
    if args.fake_redis:
        await asyncio.wait_for(redis.xread_started.wait(), timeout=1.0)

    comparisons: list[dict[str, Any]] = []
    await ingestor.start()
    try:
        for index in range(max(1, int(args.samples))):
            remote_payload = await asyncio.wait_for(queue.get(), timeout=args.timeout_s)
            remote = market_event_from_bus_payload(remote_payload)
            local_payload = published[index] if index < len(published) else None
            if local_payload is None:
                comparisons.append(
                    {
                        "index": index,
                        "equal": False,
                        "differences": ("missing_local_publish",),
                        "t_end_ms": remote.t_end_ms,
                    }
                )
                continue
            local = market_event_from_payload(local_payload)
            result = compare_market_events(local, remote)
            comparisons.append(
                {
                    "index": index,
                    "equal": result.equal,
                    "differences": list(result.differences),
                    "t_end_ms": remote.t_end_ms,
                }
            )
    finally:
        await ingestor.stop()
        await publisher_bus.close()
        await subscriber_bus.close()

    summary = {
        "type": "MarketEventComparisonSummary",
        "topic": topic,
        "samples": len(comparisons),
        "matched": sum(1 for row in comparisons if row["equal"]),
        "mismatched": sum(1 for row in comparisons if not row["equal"]),
        "comparisons": comparisons,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["mismatched"]:
        raise SystemExit(1)


def _select_feed(
    feeds: tuple[IngestorFeedConfig, ...],
    exchange: str | None,
    symbol: str | None,
    interval_ms: int | None,
) -> IngestorFeedConfig:
    if len(feeds) == 1 and not exchange and not symbol and interval_ms is None:
        return feeds[0]
    for feed in feeds:
        if exchange and feed.normalized_exchange() != exchange.lower().strip():
            continue
        if symbol and feed.symbol != symbol:
            continue
        if interval_ms is not None and feed.interval_ms != interval_ms:
            continue
        return feed
    raise ValueError("no ingestor feed matched the selector arguments")


class _SharedFakeRedis:
    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._counter = 0
        self._condition = asyncio.Condition()
        self.xread_started = asyncio.Event()

    async def xadd(self, stream: str, fields: dict[str, str], **kwargs: Any) -> str:
        async with self._condition:
            self._counter += 1
            message_id = f"{self._counter}-0"
            self.streams.setdefault(stream, []).append((message_id, fields))
            self._condition.notify_all()
            return message_id

    async def xread(self, streams: dict[str, str], *, block: int | None = None, count: int | None = None):
        async with self._condition:
            start_indexes = {
                stream: self._start_index(stream, last_id)
                for stream, last_id in streams.items()
            }
            self.xread_started.set()
            while True:
                for stream, start_index in start_indexes.items():
                    messages = self.streams.get(stream, [])
                    if len(messages) > start_index:
                        limit = count or len(messages)
                        return [(stream, messages[start_index : start_index + limit])]
                timeout = None if block is None else max(0.001, float(block) / 1000.0)
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    return []

    async def aclose(self) -> None:
        return None

    def _start_index(self, stream: str, last_id: str) -> int:
        messages = self.streams.get(stream, [])
        if last_id == "$":
            return len(messages)
        for index, (message_id, _) in enumerate(messages):
            if message_id == last_id:
                return index + 1
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ingestor publish vs Redis bus delivery.")
    parser.add_argument("--config", required=True, help="Ingestor JSON config path.")
    parser.add_argument("--samples", type=int, default=3, help="Number of MarketEvent samples to compare.")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Per-sample wait timeout.")
    parser.add_argument("--exchange", default=None, help="Optional feed exchange selector.")
    parser.add_argument("--symbol", default=None, help="Optional feed symbol selector.")
    parser.add_argument("--interval-ms", type=int, default=None, help="Optional feed interval selector.")
    parser.add_argument(
        "--fake-redis",
        action="store_true",
        help="Use in-process fake Redis for contract tests without a live Redis server.",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
