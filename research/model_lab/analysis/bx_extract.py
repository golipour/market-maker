"""Extract MarketEvent price/flow series for Bitwana + Nobitex into npz caches.

Uses the (exchange, symbol, ts DESC) index and streams rows so we never hold
the full payload set in memory.
"""
import gzip
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import psycopg2

sys.path.insert(0, "/Users/teanab/Documents/projects/market-maker")
from research.model_lab.data.loader import resolve_database_url  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CACHE = os.path.join(OUT, "bx_series")

SQL = """
    SELECT me.ts,
           me.payload
    FROM mm_events me
    WHERE me.exchange = %s
      AND me.symbol = %s
      AND me.event_type = 'MarketEvent'
      AND me.ts >= %s
      AND me.ts <= %s
    ORDER BY me.ts ASC
"""


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def extract(cur, exchange, symbol, since, until, topic=None):
    cur.execute(SQL, (exchange, symbol, since, until))
    cols = {k: [] for k in (
        "ts_ms", "t_end_ms", "buy_volume", "sell_volume",
        "best_bid", "best_bid_qty", "best_ask", "best_ask_qty",
        "mid", "microprice", "spread", "stale_ms", "interval_ms",
    )}
    n = 0
    while True:
        rows = cur.fetchmany(2000)
        if not rows:
            break
        for ts, payload in rows:
            p = payload if isinstance(payload, dict) else json.loads(payload)
            ob = p.get("orderbook") or {}
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []
            bid = _f(bids[0]["price"]) if bids else _f(p.get("bid"))
            bidq = _f(bids[0]["qty"]) if bids else np.nan
            ask = _f(asks[0]["price"]) if asks else _f(p.get("ask"))
            askq = _f(asks[0]["qty"]) if asks else np.nan
            micro = _f(p.get("microprice"))
            if not np.isfinite(bid) or not np.isfinite(ask) or bid <= 0 or ask <= 0:
                continue
            mid = 0.5 * (bid + ask)
            if not np.isfinite(micro):
                micro = mid
            t_end = p.get("t_end_ms")
            obs = p.get("orderbook_timestamp_ms") or p.get("t")
            cols["ts_ms"].append(ts.timestamp() * 1000.0)
            cols["t_end_ms"].append(float(t_end) if t_end is not None else ts.timestamp() * 1000.0)
            cols["buy_volume"].append(_f(p.get("buy_volume")))
            cols["sell_volume"].append(_f(p.get("sell_volume")))
            cols["best_bid"].append(bid)
            cols["best_bid_qty"].append(bidq)
            cols["best_ask"].append(ask)
            cols["best_ask_qty"].append(askq)
            cols["mid"].append(mid)
            cols["microprice"].append(micro)
            cols["spread"].append(ask - bid)
            cols["stale_ms"].append(
                float(cols["t_end_ms"][-1] - obs) if obs is not None else np.nan)
            cols["interval_ms"].append(float(p.get("interval_ms") or 0.0))
            n += 1
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}, n


def main() -> int:
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
    url = resolve_database_url() if os.environ.get("MM_DATABASE_URL") else \
        "host=127.0.0.1 port=15432 user=mm password=mm dbname=postgres"
    os.makedirs(CACHE, exist_ok=True)
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=days)
    targets = [
        ("nobitex", "USDT_IRT"),
        ("nobitex", "BTC_IRT"),
        ("bitwana", "USDT_IRT"),
    ]
    with psycopg2.connect(url) as conn:
        with conn.cursor() as boot:
            boot.execute("SET statement_timeout = '600s'")
        with conn.cursor() as boot:
            boot.execute("SET statement_timeout = '600s'")
        for ex, sym in targets:
            print(f"extracting {ex}/{sym} since {since:%Y-%m-%d %H:%M} ...", flush=True)
            cur = conn.cursor(name=f"bx_{ex}_{sym}")
            cur.itersize = 5000
            try:
                d, n = extract(cur, ex, sym, since, until)
            except Exception as exc:  # noqa: BLE001
                print("  ERR", exc, flush=True)
                conn.rollback()
                cur.close()
                continue
            cur.close()
            if not n:
                print("  no rows", flush=True)
                continue
            path = os.path.join(CACHE, f"{ex}_{sym}.npz")
            np.savez_compressed(path, **d)
            print(f"  n={n}  {path}", flush=True)
            print(f"  span {d['ts_ms'][0]:.0f} .. {d['ts_ms'][-1]:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
