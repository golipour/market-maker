"""Load interval steps from Postgres."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from mm.env_loader import load_mm_env

from research.model_lab.core.types import IntervalStep, orderbook_from_levels


def resolve_database_url(explicit: str | None = None) -> str:
    load_mm_env()
    url = explicit or os.environ.get("MM_DATABASE_URL") or os.environ.get("MM_STEP_ECONOMICS_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "No database URL. Set MM_DATABASE_URL in .env or pass --database-url."
        )
    return url


def parse_since(value: str | None, *, until: datetime) -> datetime | None:
    if value is None or value.lower() in {"", "all"}:
        return None
    value = value.strip().lower()
    if value.endswith("h"):
        return until - timedelta(hours=float(value[:-1]))
    if value.endswith("d"):
        return until - timedelta(days=float(value[:-1]))
    if value.endswith("m"):
        return until - timedelta(minutes=float(value[:-1]))
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_levels(orderbook: object, side: str) -> list[tuple[float, float]]:
    if orderbook is None:
        return []
    if isinstance(orderbook, str):
        try:
            orderbook = json.loads(orderbook)
        except json.JSONDecodeError:
            return []
    if not isinstance(orderbook, dict):
        return []
    levels: list[tuple[float, float]] = []
    for level in orderbook.get(side) or []:
        if not isinstance(level, dict):
            continue
        price = _safe_float(level.get("price"))
        qty = _safe_float(level.get("qty"))
        if math.isfinite(price) and math.isfinite(qty) and price > 0 and qty >= 0:
            levels.append((price, qty))
    if side == "bids":
        return sorted(levels, key=lambda item: item[0], reverse=True)
    return sorted(levels, key=lambda item: item[0])


def load_interval_steps(
    *,
    database_url: str | None = None,
    exchange: str,
    symbol: str,
    interval_ms: int,
    since: datetime | None = None,
    until: datetime | None = None,
    source: str = "auto",
) -> list[IntervalStep]:
    """Load time-ordered interval steps.

    ``source``:
    - ``auto``: prefer ``mm_events`` MarketEvent (full orderbook), else ``mm_market_facts``.
    - ``events``: ``mm_events`` only.
    - ``market_facts``: shallow book (best bid/ask only) from ``mm_market_facts``.
    """

    import psycopg2

    url = resolve_database_url(database_url)
    until = until or datetime.now(timezone.utc)
    exchange = exchange.lower().strip()
    symbol = symbol.upper().strip()
    topic = f"market:events:{exchange}:{symbol}:{int(interval_ms)}"

    if source in {"auto", "events"}:
        steps = _load_from_events(url, exchange, symbol, interval_ms, topic, since, until)
        if steps or source == "events":
            return steps
    return _load_from_market_facts(url, exchange, symbol, interval_ms, since, until)


def _load_from_events(
    database_url: str,
    exchange: str,
    symbol: str,
    interval_ms: int,
    topic: str,
    since: datetime | None,
    until: datetime,
) -> list[IntervalStep]:
    import psycopg2

    query = """
        SELECT
            me.ts,
            COALESCE((me.payload->>'t_end_ms')::bigint, me.timestamp_ms) AS t_end_ms,
            COALESCE((me.payload->>'buy_volume')::double precision, 0.0),
            COALESCE((me.payload->>'sell_volume')::double precision, 0.0),
            COALESCE(
                (me.payload->'orderbook'->'bids'->0->>'price')::double precision,
                (me.payload->>'microprice')::double precision
            ),
            COALESCE(
                (me.payload->'orderbook'->'asks'->0->>'price')::double precision,
                (me.payload->>'microprice')::double precision
            ),
            COALESCE((me.payload->>'microprice')::double precision, 0.0),
            me.payload->'orderbook'
        FROM mm_events me
        WHERE me.event_type = 'MarketEvent'
          AND me.exchange = %s
          AND me.symbol = %s
          AND (me.topic = %s OR (me.payload->>'interval_ms')::bigint = %s)
          AND me.ts <= %s
          AND (%s::timestamptz IS NULL OR me.ts >= %s)
        ORDER BY me.ts ASC
    """
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (exchange, symbol, topic, interval_ms, until, since, since))
            rows = cur.fetchall()
    return _rows_to_steps(rows, exchange, symbol, interval_ms)


def _load_from_market_facts(
    database_url: str,
    exchange: str,
    symbol: str,
    interval_ms: int,
    since: datetime | None,
    until: datetime,
) -> list[IntervalStep]:
    import psycopg2

    query = """
        SELECT
            ts,
            t_end_ms,
            COALESCE(buy_volume, 0.0),
            COALESCE(sell_volume, 0.0),
            best_bid,
            best_ask,
            COALESCE(microprice, mid, 0.0)
        FROM mm_market_facts
        WHERE exchange = %s
          AND symbol = %s
          AND interval_ms = %s
          AND ts <= %s
          AND (%s::timestamptz IS NULL OR ts >= %s)
        ORDER BY ts ASC
    """
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (exchange, symbol, interval_ms, until, since, since))
            rows = cur.fetchall()

    steps: list[IntervalStep] = []
    for row in rows:
        best_bid = _safe_float(row[4])
        best_ask = _safe_float(row[5])
        if not (math.isfinite(best_bid) and math.isfinite(best_ask) and best_bid > 0 and best_ask > 0):
            continue
        mid = 0.5 * (best_bid + best_ask)
        micro = _safe_float(row[6])
        if not math.isfinite(micro):
            micro = mid
        bids = [(best_bid, 1.0)]
        asks = [(best_ask, 1.0)]
        steps.append(
            IntervalStep(
                ts=row[0],
                t_end_ms=int(row[1]),
                exchange=exchange,
                symbol=symbol,
                interval_ms=interval_ms,
                buy_volume=_safe_float(row[2]),
                sell_volume=_safe_float(row[3]),
                best_bid=best_bid,
                best_ask=best_ask,
                microprice=micro,
                spread=best_ask - best_bid,
                mid=mid,
                orderbook=orderbook_from_levels(bids, asks, mid=mid),
            )
        )
    return steps


def _rows_to_steps(
    rows: list[tuple[Any, ...]],
    exchange: str,
    symbol: str,
    interval_ms: int,
) -> list[IntervalStep]:
    steps: list[IntervalStep] = []
    for row in rows:
        best_bid = _safe_float(row[4])
        best_ask = _safe_float(row[5])
        if not (math.isfinite(best_bid) and math.isfinite(best_ask) and best_bid > 0 and best_ask > 0):
            continue
        bids = _parse_levels(row[7], "bids")
        asks = _parse_levels(row[7], "asks")
        if not bids:
            bids = [(best_bid, 1.0)]
        if not asks:
            asks = [(best_ask, 1.0)]
        mid = 0.5 * (best_bid + best_ask)
        micro = _safe_float(row[6])
        if not math.isfinite(micro):
            micro = mid
        steps.append(
            IntervalStep(
                ts=row[0],
                t_end_ms=int(row[1]),
                exchange=exchange,
                symbol=symbol,
                interval_ms=interval_ms,
                buy_volume=_safe_float(row[2]),
                sell_volume=_safe_float(row[3]),
                best_bid=best_bid,
                best_ask=best_ask,
                microprice=micro,
                spread=best_ask - best_bid,
                mid=mid,
                orderbook=orderbook_from_levels(bids, asks, mid=mid),
            )
        )
    return steps
