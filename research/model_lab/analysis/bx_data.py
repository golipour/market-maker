"""Causal decision-frame construction for the Bitwana/Nobitex backtest.

Alignment contract: row t is the decision instant, features use rows <= t,
and targets are realised over [t, t+1).

Gaps where dt != interval_ms start a new run segment; targets bridging across
runs (where run[t+1] != run[t]) are strictly excluded from prequential scoring.
"""
from __future__ import annotations

import os

import numpy as np

SERIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "bx_series")


def load_series(exchange: str, symbol: str, series_dir: str | None = None) -> dict:
    path = os.path.join(series_dir or SERIES_DIR, f"{exchange}_{symbol}.npz")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def detect_tick(mid: np.ndarray, default: float = 1.0) -> float:
    """Smallest positive spacing on the observed mid grid."""
    d = np.diff(np.asarray(mid, dtype=np.float64))
    d = d[np.isfinite(d) & (d != 0.0)]
    if d.size == 0:
        return float(default)
    g = np.abs(d)
    for cand in (1.0, 0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        if np.all(np.abs(g / cand - np.round(g / cand)) < 1e-6):
            return cand
    return float(np.min(g))


def build(d: dict, *, interval_ms: float = 10000.0) -> dict:
    """Filter to target interval stream and split into contiguous runs."""
    if "interval_ms" in d:
        mask = d["interval_ms"] == float(interval_ms)
        d_filt = {k: d[k][mask] for k in d}
    else:
        d_filt = d
    te = d_filt["t_end_ms"]
    step = float(interval_ms)
    dt = np.diff(te)
    # A new run segment starts whenever dt != step
    run = np.concatenate([[0], np.cumsum(dt != step)])
    f = {k: d_filt[k].astype(np.float64) for k in d_filt}
    f["run"] = run
    f["step_ms"] = step
    f["tick"] = detect_tick(f["mid"])
    return f


def derived(f: dict, symbol: str = "USDT_IRT") -> dict:
    """All arrays length N, indexed by decision instant t."""
    N = len(f["mid"])

    def diff(a):
        return np.concatenate([[np.nan], np.diff(a)])

    def fwd(a):
        return np.concatenate([a[1:], [np.nan]])

    mid = f["mid"]
    bb, ba = f["best_bid"], f["best_ask"]
    tot = f["buy_volume"] + f["sell_volume"]

    # Lot size scaling for flow: BTC is small fractions; USDT is lot=1.0
    lot_scale = 1e-4 if "BTC" in symbol.upper() else 1.0

    y = {
        "N": N,
        "run": f["run"],
        "step_ms": f["step_ms"],
        "ts_ms": f["ts_ms"],
        "tick": f["tick"],
        "lot_scale": lot_scale,
        "symbol": symbol,
    }
    y["mid"] = mid
    y["best_bid"] = bb
    y["best_ask"] = ba
    y["spread"] = f["spread"]
    y["stale_ms"] = f["stale_ms"]
    y["micro_minus_mid"] = f["microprice"] - mid
    y["queue_imb"] = (
        (f["best_bid_qty"] - f["best_ask_qty"])
        / np.maximum(1.0, f["best_bid_qty"] + f["best_ask_qty"])
    )
    y["log_ask_over_bid"] = np.log((f["best_ask_qty"] + 1.0) / (f["best_bid_qty"] + 1.0))
    y["bid_qty"] = f["best_bid_qty"]
    y["ask_qty"] = f["best_ask_qty"]
    y["bv"] = f["buy_volume"]
    y["sv"] = f["sell_volume"]
    y["tot"] = tot
    y["ocur"] = (tot > 0).astype(float)
    y["sflow"] = f["buy_volume"] - f["sell_volume"]
    y["dmid"] = diff(mid)
    y["dbid"] = diff(bb)
    y["dask"] = diff(ba)
    y["dlogtot"] = diff(np.log1p(np.maximum(tot, 0.0)))
    y["logtot"] = np.log1p(np.maximum(tot, 0.0))

    # Next-step targets realized over [t, t+1)
    y["dmid_n1"] = fwd(y["dmid"])
    y["tot_n1"] = fwd(y["tot"])
    y["bv_n1"] = fwd(y["bv"])
    y["sv_n1"] = fwd(y["sv"])

    # Discretized integer lot targets
    y["bv_lots_n1"] = np.maximum(0, np.round(y["bv_n1"] / lot_scale)).astype(np.int64)
    y["sv_lots_n1"] = np.maximum(0, np.round(y["sv_n1"] / lot_scale)).astype(np.int64)

    # Valid step mask: target exists AND t -> t+1 is within the same contiguous run
    valid_step = np.zeros(N, dtype=bool)
    for t in range(N - 1):
        if (
            f["run"][t + 1] == f["run"][t]
            and np.isfinite(y["dmid_n1"][t])
            and np.isfinite(y["bv_n1"][t])
            and np.isfinite(y["sv_n1"][t])
        ):
            valid_step[t] = True
    y["valid_step"] = valid_step

    return y
