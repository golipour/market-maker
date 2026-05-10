"""Observed cumulative limit-order flow vs distance from mid (LOF diagnostic)."""

from __future__ import annotations

import numpy as np
from mm.integrations.trade_stats import Orderbook


def limit_order_flow_observed(
    prev_ob: Orderbook,
    cur_ob: Orderbook,
    buy_vol: float,
    sell_vol: float,
    spread_max: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Observed cumulative limit order quantity change from mid out to each tick distance.
    Orderbooks aligned by absolute price; prev OB reduced by fills.
    Returns (x, cum_bid, cum_ask) with x = price − prev_mid in [-spread_max, spread_max].
    """
    prev_mid = int(round(float(prev_ob.mid)))
    spread_max = int(spread_max)
    price_lo = prev_mid - spread_max
    price_hi = prev_mid + spread_max

    def depth_by_price(ob: Orderbook, side: str) -> dict[int, float]:
        levels = ob.bids if side == "bid" else ob.asks
        out: dict[int, float] = {}
        for lev in levels:
            p = lev.price
            if price_lo <= p <= price_hi:
                out[p] = out.get(p, 0.0) + float(lev.qty)
        return out

    def reduce_by_flow_at_price(
        depth: dict[int, float], flow_vol: float, best_first_asc: bool
    ) -> dict[int, float]:
        remaining = float(flow_vol)
        out = dict(depth)
        prices = sorted(out.keys(), reverse=not best_first_asc)
        for p in prices:
            if remaining <= 0:
                break
            q = out[p]
            take = min(q, max(0, remaining))
            new_q = q - take
            remaining -= take
            if new_q > 0:
                out[p] = new_q
            else:
                del out[p]
        return out

    prev_bid = depth_by_price(prev_ob, "bid")
    prev_ask = depth_by_price(prev_ob, "ask")
    prev_bid_reduced = reduce_by_flow_at_price(prev_bid, sell_vol, best_first_asc=False)
    prev_ask_reduced = reduce_by_flow_at_price(prev_ask, buy_vol, best_first_asc=True)
    cur_bid = depth_by_price(cur_ob, "bid")
    cur_ask = depth_by_price(cur_ob, "ask")

    x = np.arange(-spread_max, spread_max + 1, dtype=np.float64)
    order_change_bid = np.zeros_like(x)
    order_change_ask = np.zeros_like(x)
    for i, xi in enumerate(x):
        P = prev_mid + int(xi)
        if xi <= 0:
            order_change_bid[i] = cur_bid.get(P, 0.0) - prev_bid_reduced.get(P, 0.0)
        if xi >= 0:
            order_change_ask[i] = cur_ask.get(P, 0.0) - prev_ask_reduced.get(P, 0.0)

    mid_ix = int(spread_max)
    cum_bid = np.zeros_like(order_change_bid)
    cum_ask = np.zeros_like(order_change_ask)
    cum_bid[mid_ix::-1] = np.cumsum(order_change_bid[mid_ix::-1])
    cum_ask[mid_ix:] = np.cumsum(order_change_ask[mid_ix:])
    return x, cum_bid, cum_ask
