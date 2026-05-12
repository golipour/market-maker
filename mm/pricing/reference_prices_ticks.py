"""
Reference prices (ticks) from orderbook and market order flow (MOF-only).

Tick VWAP helpers for MOF-only reference pricing (standalone; ``mm.mpc`` has no legacy imports).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mm.integrations.trade_stats import Orderbook


def _depth_by_price(ob: "Orderbook", side: str) -> dict[int, float]:
    levels = ob.bids if side == "bid" else ob.asks
    out: dict[int, float] = {}
    for lev in levels:
        p = int(lev.price)
        out[p] = out.get(p, 0.0) + float(lev.qty)
    return out


def _consume_depth(depth: dict[int, float], vol: float, best_first_asc: bool) -> dict[int, float]:
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


def _total_depth(depth: dict[int, float]) -> float:
    return float(sum(max(0.0, q) for q in depth.values()))


def _vwap_to_depth(depth: dict[int, float], depth_lots: float, best_first_asc: bool) -> float:
    if depth_lots <= 0:
        return float("nan")
    remaining = float(depth_lots)
    px_qty = 0.0
    for p in sorted(depth.keys(), reverse=not best_first_asc):
        if remaining <= 0:
            break
        q = max(0.0, float(depth.get(p, 0.0)))
        if q <= 0:
            continue
        take = min(q, remaining)
        px_qty += float(p) * take
        remaining -= take
    if remaining > 1e-9:
        return float("nan")
    return float(px_qty / depth_lots)


def reference_prices_ticks_from_orderbook(
    ob: "Orderbook",
    target_depth_lots: float = 1000.0,
) -> tuple[float, float]:
    """
    Buy ref = VWAP on ask side, sell ref = VWAP on bid side, both to ``target_depth_lots``
    (capped by displayed depth). Falls back to ``ob.mid`` when too thin.
    """
    mid = float(ob.mid)
    bid = _depth_by_price(ob, "bid")
    ask = _depth_by_price(ob, "ask")
    d_b = _total_depth(bid)
    d_a = _total_depth(ask)
    td = max(0.0, float(target_depth_lots))
    depth_used = min(td, d_b, d_a)
    if depth_used <= 1e-12:
        return mid, mid
    sell_ref = _vwap_to_depth(bid, depth_used, best_first_asc=False)
    buy_ref = _vwap_to_depth(ask, depth_used, best_first_asc=True)
    if not math.isfinite(sell_ref):
        sell_ref = mid
    if not math.isfinite(buy_ref):
        buy_ref = mid
    return float(buy_ref), float(sell_ref)


def reference_prices_from_transition(
    prev_ob: "Orderbook",
    cur_ob: "Orderbook",
    buy_vol: float,
    sell_vol: float,
    target_depth_lots: float,
) -> tuple[float, float, float, float]:
    """
    Instantaneous VWAPs: realized on ``cur_ob`` vs MOF-only post-flow book from ``prev_ob``.
    Returns (buy_real, sell_real, buy_cmp, sell_cmp) in ticks.
    """
    bid_prev = _depth_by_price(prev_ob, "bid")
    ask_prev = _depth_by_price(prev_ob, "ask")
    bid_after = _consume_depth(bid_prev, float(sell_vol), best_first_asc=False)
    ask_after = _consume_depth(ask_prev, float(buy_vol), best_first_asc=True)
    bid_cur = _depth_by_price(cur_ob, "bid")
    ask_cur = _depth_by_price(cur_ob, "ask")
    td = max(0.0, float(target_depth_lots))
    depth_cur = min(_total_depth(bid_cur), _total_depth(ask_cur))
    depth_cmp = min(_total_depth(bid_after), _total_depth(ask_after))
    depth_used = float(min(td, depth_cur, depth_cmp))
    mid = float(cur_ob.mid)
    if depth_used <= 1e-12:
        return mid, mid, mid, mid
    buy_real = _vwap_to_depth(ask_cur, depth_used, best_first_asc=True)
    sell_real = _vwap_to_depth(bid_cur, depth_used, best_first_asc=False)
    buy_cmp = _vwap_to_depth(ask_after, depth_used, best_first_asc=True)
    sell_cmp = _vwap_to_depth(bid_after, depth_used, best_first_asc=False)
    br = float(buy_real) if math.isfinite(buy_real) else mid
    sr = float(sell_real) if math.isfinite(sell_real) else mid
    bc = float(buy_cmp) if math.isfinite(buy_cmp) else mid
    sc = float(sell_cmp) if math.isfinite(sell_cmp) else mid
    return br, sr, bc, sc


# Alias for ``mm.mpc.price_torch`` call sites.
reference_prices_from_transition_ticks = reference_prices_from_transition


def mid_ticks_after_market_order_flow(
    prev_ob: "Orderbook",
    buy_vol: float,
    sell_vol: float,
    spread_max: float = 100.0,
) -> float:
    """Mid (ticks) from best bid/ask after consuming MOF on ``prev_ob`` only (no LOF replenishment)."""
    prev_mid = int(round(float(prev_ob.mid)))
    sm = int(max(1, round(float(spread_max))))
    price_lo = prev_mid - sm
    price_hi = prev_mid + sm

    def depth_by_price(ob: "Orderbook", side: str) -> dict[int, float]:
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
    bid_r = reduce_by_flow_at_price(prev_bid, sell_vol, best_first_asc=False)
    ask_r = reduce_by_flow_at_price(prev_ask, buy_vol, best_first_asc=True)
    best_bid_cands = [p for p, q in bid_r.items() if q > 0]
    best_ask_cands = [p for p, q in ask_r.items() if q > 0]
    bb = max(best_bid_cands) if best_bid_cands else prev_mid
    ba = min(best_ask_cands) if best_ask_cands else prev_mid
    return (float(bb) + float(ba)) / 2.0
