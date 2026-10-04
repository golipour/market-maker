"""Build causal MPC ``State`` objects from the cached Nobitex 10s series.

The early-stopping study that was previously rejected was measured on the
*degenerate* warm-start manifold (all components initialized at the same
depth), so its conclusion has to be re-measured on the symmetry-broken
baseline. This module is the deterministic frame source both measurements
share, so before/after numbers are directly comparable.

Filtration contract (mirrors ``harness.py``):
  row t <- observation available at t_end_ms
  no feature at row t uses rows > t
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import CACHE, NUM_COLS, core_frame  # noqa: E402

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from mm.mpc.mpc_state import OrderbookRelative, State  # noqa: E402


def _book_from_row(
    f: dict,
    i: int,
    n_levels: int,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Synthesize a causal level ladder from best quote + aggregate depth.

    The cached series carries top-of-book and total depth, not a full ladder.
    We build a geometrically decaying ladder anchored on the best quote and
    the aggregate visible size. This is a *stand-in* book: it reproduces the
    shape statistics the depth decoder sees (monotone decreasing size away
    from the touch, spread-scaled level spacing) without inventing levels
    that would not be reconstructable from the export.
    """
    bb = f["best_bid"][i]
    ba = f["best_ask"][i]
    bid_sz = max(f["best_bid_qty"][i], 1e-6)
    ask_sz = max(f["best_ask_qty"][i], 1e-6)
    bid_depth = max(f["bid_depth_levels"][i], 1.0)
    ask_depth = max(f["ask_depth_levels"][i], 1.0)

    spread_ticks = max(ba - bb, 1e-6)
    tick = spread_ticks

    decay_b = bid_sz / bid_depth
    decay_a = ask_sz / ask_depth

    bids: list[tuple[float, float]] = []
    asks: list[tuple[float, float]] = []
    for k in range(n_levels):
        # Levels fan out on the tick grid; sizes decay geometrically so the
        # cumulative curve has the concave shape the decoder expects.
        bids.append((bb - k * tick, bid_sz * (decay_b**k) if k < 12 else decay_b))
        asks.append((ba + k * tick, ask_sz * (decay_a**k) if k < 12 else decay_a))
    return tuple(bids), tuple(asks)


def build_states(
    *,
    n_states: int = 40,
    n_levels: int = 20,
    stride: int = 37,
    start: int = 500,
) -> list[tuple[State, dict]]:
    """Deterministic, evenly-spaced replay corpus of causal MPC states."""
    if not os.path.exists(CACHE):
        raise FileNotFoundError(f"missing dataset cache: {CACHE}")
    d = _load_cache()
    f = core_frame(d)
    total = len(f["mid"])
    if total <= start + stride * n_states:
        raise ValueError(f"corpus too small: {total} rows for {n_states} states")

    out: list[tuple[State, dict]] = []
    for s in range(n_states):
        i = start + s * stride
        bids, asks = _book_from_row(f, i, n_levels)
        mid = float(f["mid"][i])
        buy_vwap = mid - 0.5 * abs(mid - float(f["best_bid"][i])) - 0.01 * float(f["spread"][i])
        sell_vwap = mid + 0.5 * abs(float(f["best_ask"][i]) - mid) + 0.01 * float(f["spread"][i])
        st = State(
            mid=mid,
            ob_rel=OrderbookRelative(bids=bids, asks=asks),
            buy_vol=float(f["buy_volume"][i]),
            sell_vol=float(f["sell_volume"][i]),
            inventory=0.0,
            cash=10_000.0,
            buy_vwap=buy_vwap,
            sell_vwap=sell_vwap,
        )
        meta = {
            "index": int(i),
            "t_end_ms": int(f["t_end"][i]),
            "spread": float(f["spread"][i]),
            "imbalance": float(f["imbalance"][i]),
            "buy_vol": float(f["buy_volume"][i]),
            "sell_vol": float(f["sell_volume"][i]),
        }
        out.append((st, meta))
    return out


_CACHE_D: dict | None = None


def _load_cache() -> dict:
    global _CACHE_D
    if _CACHE_D is None:
        z = np.load(CACHE, allow_pickle=True)
        _CACHE_D = {k: z[k] for k in z.files}
    return _CACHE_D


if __name__ == "__main__":
    corpus = build_states(n_states=5)
    for st, meta in corpus:
        print(meta)
        print(f"  mid={st.mid:.2f} best_bid={st.ob_rel.bids[0][0]:.2f} best_ask={st.ob_rel.asks[0][0]:.2f}")
        print(f"  bid levels={len(st.ob_rel.bids)} ask levels={len(st.ob_rel.asks)}")
        print(f"  top bid qty={st.ob_rel.bids[0][1]:.4f} top ask qty={st.ob_rel.asks[0][1]:.4f}")
