"""Load the Postgres-extracted L2 book series into a compact causal frame.

Each row is one 10s bucket with full 24-level depth.  We need, per transition
t -> t+1, exactly what TorchConditionalMechanicalMixturePriceModel sees:

  prev_ob  (bids/asks price->qty),  cur_ob,  buy_vol,  sell_vol

and we reproduce its features verbatim (price_torch.py:500-540):
  mech_bid/mech_ask from _consume_depth walking the book with the aggressor
  volume, then mid/spread/buy_shape/sell_shape, all in log-difference form.
"""
import csv
import gzip
import json
import os

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK_CSV = os.path.join(LAB, "dataset", "nobitex_usdt_irt_10s_book.csv")
CACHE = os.path.join(LAB, "analysis", "out", "book_cache.npz")


def load(path=None, limit=None):
    if path is None:
        path = CACHE if os.path.exists(CACHE) else BOOK_CSV
    if path == CACHE:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    csv.field_size_limit(10 ** 9)
    t_end = []
    bids = []
    asks = []
    bv = []
    sv = []
    imb = []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        for i, r in enumerate(rd):
            if limit is not None and i >= limit:
                break
            try:
                b = json.loads(r["bids"])
                a = json.loads(r["asks"])
            except (ValueError, TypeError):
                continue
            if not b or not a:
                continue
            t_end.append(int(r["t_end"]))
            bids.append({float(l["price"]): float(l["qty"]) for l in b})
            asks.append({float(l["price"]): float(l["qty"]) for l in a})
            bv.append(float(r["buy_vol"] or 0.0))
            sv.append(float(r["sell_vol"] or 0.0))
            imb.append(float(r["imbalance"] or 0.0))
    out = {
        "t_end": np.asarray(t_end, dtype=np.int64),
        "bids": np.asarray(bids, dtype=object),
        "asks": np.asarray(asks, dtype=object),
        "buy_vol": np.asarray(bv, dtype=np.float64),
        "sell_vol": np.asarray(sv, dtype=np.float64),
        "imbalance": np.asarray(imb, dtype=np.float64),
        "N": len(t_end),
    }
    return out


def consume_depth(depth, vol, best_first_asc):
    """Port of mm/pricing/reference_prices_ticks.py:25 -- identical logic."""
    out = dict(depth)
    remaining = max(0.0, float(vol))
    for p in sorted(list(out.keys()), reverse=not best_first_asc):
        if remaining <= 0:
            break
        q = float(out.get(p, 0.0))
        take = min(q, remaining)
        q_new = q - take
        remaining -= take
        if q_new > 0:
            out[p] = q_new
        elif p in out:
            del out[p]
    return out


def features(mid, spread, buy_vwap, sell_vwap, eps=1e-9):
    """Port of _features_from_prices (price_torch.py:457)."""
    half = 0.5 * max(eps, float(spread))
    return {
        "mid": float(mid),
        "spread": max(eps, float(spread)),
        "buy_shape": max(eps, float(buy_vwap) - float(mid) - half),
        "sell_shape": max(eps, float(mid) - float(sell_vwap) - half),
    }


def vwap_to_price(depth, lots, best_first_asc):
    """Port of _vwap_to_depth (reference_prices_ticks.py:46)."""
    remaining = float(lots)
    px_qty = 0.0
    qty = 0.0
    for p in sorted(depth.keys(), reverse=not best_first_asc):
        if remaining <= 0:
            break
        q = max(0.0, float(depth.get(p, 0.0)))
        if q <= 0:
            continue
        take = min(q, remaining)
        px_qty += p * take
        qty += take
        remaining -= take
    if qty <= 0:
        return float("nan")
    return px_qty / qty
