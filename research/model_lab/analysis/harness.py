"""Analysis harness for the 10s Nobitex USDT_IRT interval series.

Loads the parquet export, builds a *causal* row-per-decision dataset, and exposes
clean arrays for the diagnostics in the roadmap.

Filtration contract (see roadmap S1):
  row t  <-  observation available at t_end_ms
  feature at row t may use only rows <= t
  target  at row t is realised during [t_end_ms, t_end_ms + 10s)
"""

from __future__ import annotations

import os

import csv
import gzip

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LAB = os.path.join(REPO, "research", "model_lab")
CSV = os.path.join(LAB, "dataset", "nobitex_USDT_IRT_10000ms.csv")

NUM_COLS = [
    "t_start_ms", "t_end_ms", "interval_ms",
    "best_bid", "best_bid_qty", "best_ask", "best_ask_qty",
    "mid", "spread", "microprice", "vwap",
    "buy_volume", "sell_volume", "total_trade_volume", "trade_count",
    "imbalance", "bid_depth_levels", "ask_depth_levels",
    "orderbook_timestamp_ms",
]


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "cache.npz")


def load(path: str | None = None) -> dict:
    """Stream the gz export; float64 throughout, no object arrays."""
    if path is None:
        for cand in (CACHE, CSV + ".gz", CSV):
            if os.path.exists(cand):
                path = cand
                break
        else:
            raise FileNotFoundError(
                "no source found; expected one of " + ", ".join((CACHE, CSV + ".gz", CSV)))
    if path == CACHE:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    opener = gzip.open if path.endswith(".gz") else open
    cols: dict[str, list] = {c: [] for c in NUM_COLS}
    ts: list = []
    with opener(path, "rt", newline="") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        pos = {c: head.index(c) for c in NUM_COLS}
        ts_pos = head.index("ts")
        for row in rd:
            for c in NUM_COLS:
                cols[c].append(row[pos[c]])
            ts.append(row[ts_pos])
    out = {c: np.asarray(v, dtype=np.float64) for c, v in cols.items()}
    out["ts"] = np.asarray(ts, dtype=object)
    order = np.argsort(out["t_end_ms"], kind="stable")
    for k in list(out):
        out[k] = out[k][order]
    out["t_end"] = out["t_end_ms"].astype(np.int64)
    if path in (CSV, CSV + ".gz"):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, **out)
    return out


def continuity(d: dict) -> dict:
    """Flag non-contiguous transitions (a gap is NOT a quiet bucket)."""
    dt = np.diff(d["t_end"])
    ok = dt == 10_000
    return {"dt": dt, "contiguous": ok}


def core_frame(d: dict) -> dict:
    """Continuous-run arrays with all strictly-past-differenced features."""
    c = continuity(d)
    idx = np.flatnonzero(c["contiguous"]) + 1  # rows whose predecessor is exactly 10s earlier
    f = {}
    for k in ["best_bid", "best_bid_qty", "best_ask", "best_ask_qty", "mid",
              "spread", "microprice", "vwap", "buy_volume", "sell_volume",
              "total_trade_volume", "trade_count", "imbalance",
              "bid_depth_levels", "ask_depth_levels", "t_end", "orderbook_timestamp_ms"]:
        f[k] = d[k][idx].astype(np.float64)
    f["ts"] = d["ts"][idx]
    f["n_gaps"] = int((~c["contiguous"]).sum())
    f["max_gap_ms"] = int(c["dt"].max())
    return f


def derived(f: dict) -> dict:
    """Causal feature/derivations on ONE uniform time axis.

    ALIGNMENT CONTRACT
    ------------------
    Every array in the returned dict has the SAME length N and is indexed by the
    decision instant t (the end of bucket t).  A quantity whose natural
    definition needs a neighbouring bucket is padded with NaN at the
    corresponding end, so index arithmetic and fancy-indexing are always safe:

      * level at t            -> y['mid'], y['spread'], ...
      * change over [t-1, t]  -> y['dmid']        (NaN at t = 0)
      * change over [t, t+1]  -> y['dmid_n1']     (NaN at t = N-1)

    Predicting means: use columns WITHOUT the _n1 suffix at time t to produce a
    density for the columns WITH the _n1 suffix at time t.
    """
    N = len(f["mid"])
    NaN = np.full(N, np.nan)
    mid = f["mid"]
    bb, ba = f["best_bid"], f["best_ask"]

    def diff(a):
        return np.concatenate([[np.nan], np.diff(a)])

    def fwd(a):
        return np.concatenate([a[1:], [np.nan]])

    y = {"N": N}
    # --- levels at t
    y["mid"] = mid
    y["best_bid"] = bb
    y["best_ask"] = ba
    y["spread"] = f["spread"]
    y["lspread"] = np.log(f["spread"])
    y["rel_spread_bps"] = f["spread"] / mid * 1e4
    y["stale_ms"] = f["t_end"] - f["orderbook_timestamp_ms"]
    y["micro_minus_mid"] = f["microprice"] - mid
    y["imb"] = f["imbalance"]
    y["queue_imb"] = (f["best_bid_qty"] - f["best_ask_qty"]) / np.maximum(
        1.0, f["best_bid_qty"] + f["best_ask_qty"])
    y["log_ask_over_bid"] = np.log((f["best_ask_qty"] + 1.0) / (f["best_bid_qty"] + 1.0))
    y["bid_qty"] = f["best_bid_qty"]
    y["ask_qty"] = f["best_ask_qty"]
    # --- flow realised IN bucket t
    y["bv"] = f["buy_volume"]
    y["sv"] = f["sell_volume"]
    y["tot"] = f["buy_volume"] + f["sell_volume"]
    y["ocur"] = (y["tot"] > 0).astype(float)
    y["sflow"] = f["buy_volume"] - f["sell_volume"]
    y["tc"] = f["trade_count"]
    y["dlogtot"] = np.diff(np.log1p(y["tot"]), prepend=np.nan)
    # --- changes over [t-1, t]
    y["dmid"] = diff(mid)
    y["dbid"] = diff(bb)
    y["dask"] = diff(ba)
    y["dspread"] = diff(f["spread"])
    y["dlogmid"] = diff(np.log(mid))
    y["ret_bps"] = y["dlogmid"] * 1e4
    y["dbid_step"] = np.abs(y["dbid"])
    y["dmid_step"] = np.abs(y["dmid"])

    # --- forward targets over [t, t+1]
    for k in ["dmid", "dbid", "dask", "dspread", "dlogmid", "tot", "bv", "sv",
              "sflow", "tc", "ocur", "queue_imb", "imb", "stale_ms", "spread",
              "lspread", "rel_spread_bps", "micro_minus_mid", "log_ask_over_bid",
              "dlogtot", "mid", "best_bid", "best_ask"]:
        y[k + "_n1"] = fwd(y[k])
    return y


def lag(a, k):
    """Value of `a` k steps back (NaN-padded), same length as `a`."""
    k = int(k)
    a = np.asarray(a, dtype=float)
    if k <= 0:
        return a
    return np.concatenate([np.full(k, np.nan), a[:-k]])


def ewm(a, lam):
    """Causal exponentially-weighted mean ending at each t (EWMA)."""
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    s = 0.0
    w = 0.0
    for i in range(len(a)):
        v = a[i]
        if np.isfinite(v):
            s += v
            w += 1.0
        out[i] = s / w if w > 0 else np.nan
        # decay applied to the accumulated state
        m = w
        s *= lam
        w *= lam
        w = max(w, 1e-12)
        del m
    return out


def causal_stack(y, names, horizon=1):
    """Design matrix X (state at t) and target array T (realised at t+horizon)."""
    X = np.column_stack([np.asarray(y[n], dtype=float) for n in names])
    T = np.asarray(y["dmid"], dtype=float) if horizon == 0 else None
    return X, T


def horizon_target(y, key="dmid", h=1):
    """The value of `key` realised h steps after the decision instant t."""
    return lag(np.asarray(y[key], dtype=float), h)


def shifted(y: dict, keys, k=1) -> dict:
    """Shift selected series forward k steps (the k-step-ahead target)."""
    n = len(next(iter(y.values())))
    out = {}
    for key in keys:
        a = y[key]
        out[key + f"_fwd{k}"] = np.concatenate([np.full(k, np.nan), a[:-k]]) if k > 0 else a
    return out
