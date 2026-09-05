"""Deterministic Redis-stream trim and retention canary.

This is a defensive backup to the per-XADD ``MAXLEN ~`` cap: it walks every
stream under the configured prefix and trims each one to a hard ``MAXLEN``,
records the size in bytes, and writes a fresh canary entry.

The canary is a small envelope published to ``<prefix>._retention.canary`` with
a fixed TTL. If the canary is missing on the next run, the bus is silently
failing — exactly the failure mode that caused the Sep-2026 strategy stall.

Run:

    python -m mm.tools.bus_trim \
        --redis-url "$MM_BUS_REDIS_URL" \
        --stream-prefix mm \
        --max-len 5000

Cron suggestion: every 5 minutes, in the same namespace as the ingestor.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from typing import Any

from mm.observability import counter_inc, gauge, log_event
from mm.observability.schema import (
    MM_BUS_ENVELOPE_BYTES,
    MM_RETENTION_BYTES_DELETED_TOTAL,
    MM_RETENTION_CANARY_AGE_MS,
    MM_RETENTION_ROWS_DELETED_TOTAL,
)

CANARY_TOPIC = "_retention.canary"
CANARY_TTL_SECONDS = 600
CANARY_VALUE = "ok"


async def _scan_and_trim(
    redis_client: Any,
    *,
    stream_prefix: str,
    max_len: int,
) -> dict[str, dict[str, int]]:
    """Trim every stream under ``<stream_prefix>.*`` to ``max_len`` entries.

    Returns a per-stream map of ``{stream: {"before": N, "after": M, "removed": K}}``.
    """
    pattern = f"{stream_prefix}.*"
    results: dict[str, dict[str, int]] = {}
    async for key in redis_client.scan_iter(match=pattern, count=500):
        key_str = key.decode() if isinstance(key, (bytes, bytearray)) else str(key)
        kind = await redis_client.type(key_str)
        if isinstance(kind, (bytes, bytearray)):
            kind = kind.decode()
        if kind != "stream":
            continue
        try:
            before = int(await redis_client.xlen(key_str))
        except Exception as exc:  # noqa: BLE001
            log_event(
                "bus_trim_xlen_failed",
                {"stream": key_str, "error": type(exc).__name__, "message": str(exc)},
            )
            continue
        if before <= max_len:
            results[key_str] = {"before": before, "after": before, "removed": 0}
            continue
        try:
            removed = int(
                await redis_client.xtrim(
                    key_str,
                    maxlen=max_len,
                    approximate=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log_event(
                "bus_trim_xtrim_failed",
                {"stream": key_str, "error": type(exc).__name__, "message": str(exc)},
            )
            continue
        after = before - removed
        results[key_str] = {"before": before, "after": after, "removed": removed}
        counter_inc(
            MM_RETENTION_ROWS_DELETED_TOTAL,
            inc=float(removed),
            labels={"store": "redis", "stream": key_str},
        )
    return results


async def _write_canary(redis_client: Any, *, stream_prefix: str) -> int:
    """Write a fresh canary envelope and return the age of the previous one in ms.

    A new canary is published to ``<stream_prefix>._retention.canary`` with a
    short TTL. The function looks up the previous canary's age so we can flag a
    silent bus failure.
    """
    canary_stream = f"{stream_prefix}.{CANARY_TOPIC}"
    run_id = f"bus-trim-{uuid.uuid4().hex[:8]}"
    payload = {
        "value": CANARY_VALUE,
        "ts_ms": int(time.time() * 1000),
        "run_id": run_id,
    }
    encoded = json.dumps(payload, separators=(",", ":"))
    await redis_client.xadd(
        canary_stream,
        {"data": encoded},
        maxlen=10,
        approximate=True,
    )
    await redis_client.expire(canary_stream, CANARY_TTL_SECONDS)

    # Find age of the most recent canary (excluding the one we just wrote).
    try:
        rows = await redis_client.xrevrange(canary_stream, count=2)
    except Exception:
        rows = []
    if not rows:
        return 0
    last_id, _fields = rows[0]
    try:
        last_ms = int(last_id.decode().split("-", 1)[0]) if isinstance(last_id, (bytes, bytearray)) else int(str(last_id).split("-", 1)[0])
    except Exception:
        return 0
    return int(time.time() * 1000) - last_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default=os.environ.get("MM_BUS_REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--stream-prefix", default=os.environ.get("MM_BUS_STREAM_PREFIX", "mm"))
    parser.add_argument("--max-len", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    max_len = max(1, int(args.max_len))
    stream_prefix = args.stream_prefix.strip(":") or "mm"

    async def _run() -> dict[str, Any]:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            args.redis_url,
            decode_responses=True,
            socket_connect_timeout=5.0,
            socket_timeout=5.0,
        )
        try:
            canary_age_ms = await _write_canary(
                client,
                stream_prefix=stream_prefix,
            )
            gauge(MM_RETENTION_CANARY_AGE_MS, float(canary_age_ms), labels={"store": "redis"})
            if args.dry_run:
                trim_results: dict[str, dict[str, int]] = {}
            else:
                trim_results = await _scan_and_trim(
                    client,
                    stream_prefix=stream_prefix,
                    max_len=max_len,
                )
            info_mem = await client.info("memory")
            used = int(info_mem.get("used_memory", 0))
            gauge(MM_BUS_ENVELOPE_BYTES, float(used), labels={"topic": "redis:memory"})
            return {
                "dry_run": bool(args.dry_run),
                "stream_prefix": stream_prefix,
                "max_len": max_len,
                "canary_age_ms": canary_age_ms,
                "trim": trim_results,
                "used_memory_bytes": used,
            }
        finally:
            await client.aclose()

    result = asyncio.run(_run())
    total_removed = sum(v.get("removed", 0) for v in result["trim"].values())
    counter_inc(
        MM_RETENTION_ROWS_DELETED_TOTAL,
        inc=float(total_removed),
        labels={"store": "redis", "stream": "*"},
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
