"""VWAP from a live ``RawOrderbook`` (float quote-per-base, base qty)."""

from __future__ import annotations

import math

from mm.core.models.market import RawOrderbook


def mid_raw_orderbook(book: RawOrderbook) -> float:
    """Best bid/ask mid; single-sided fallbacks; raises if book is empty."""

    if book.bids and book.asks:
        return 0.5 * (float(book.bids[0].price) + float(book.asks[0].price))
    if book.bids:
        return float(book.bids[0].price)
    if book.asks:
        return float(book.asks[0].price)
    raise ValueError("cannot compute mid from empty orderbook")


def vwap_from_raw_orderbook(book: RawOrderbook, *, target_depth: float) -> tuple[float, float]:
    """Return ``(buy_vwap, sell_vwap)`` from native bid/ask levels.

    Walk **asks** (buy) and **bids** (sell) from best prices. Uses
    ``depth_used = min(target_depth, total_bid_qty, total_ask_qty)`` (base units),
    matching the depth cap in ``vwap_prices_batched`` — which applies the same
    cap to **grid-aligned** qty tensors inside MPC, not to raw level prices here.

    When ``depth_used`` is negligible, returns ``(mid, mid)`` from ``mid_raw_orderbook``.
    """

    eps = 1e-12
    mid = mid_raw_orderbook(book)
    total_bid = sum(max(0.0, float(l.qty)) for l in book.bids)
    total_ask = sum(max(0.0, float(l.qty)) for l in book.asks)
    cap = max(0.0, float(target_depth))
    depth_used = min(cap, total_bid, total_ask)
    if depth_used <= eps:
        return mid, mid

    remaining = depth_used
    notional_buy = 0.0
    for lev in book.asks:
        if remaining <= eps:
            break
        q = max(0.0, float(lev.qty))
        if q <= eps:
            continue
        take = min(q, remaining)
        notional_buy += float(lev.price) * take
        remaining -= take
    buy_vwap = notional_buy / depth_used

    remaining = depth_used
    notional_sell = 0.0
    for lev in book.bids:
        if remaining <= eps:
            break
        q = max(0.0, float(lev.qty))
        if q <= eps:
            continue
        take = min(q, remaining)
        notional_sell += float(lev.price) * take
        remaining -= take
    sell_vwap = notional_sell / depth_used

    if not math.isfinite(buy_vwap):
        buy_vwap = mid
    if not math.isfinite(sell_vwap):
        sell_vwap = mid
    return float(buy_vwap), float(sell_vwap)
