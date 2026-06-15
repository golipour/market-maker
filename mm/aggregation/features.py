from __future__ import annotations

from mm.core.models.market import RawOrderbook, RawTrade
from mm.pricing.orderbook_ref import vwap_from_raw_orderbook


def volume_by_side(trades: tuple[RawTrade, ...]) -> tuple[float, float]:
    buy_volume = sum(t.qty for t in trades if t.side == "buy")
    sell_volume = sum(t.qty for t in trades if t.side == "sell")
    return float(buy_volume), float(sell_volume)


def vwap(trades: tuple[RawTrade, ...]) -> float:
    total_qty = sum(t.qty for t in trades)
    if total_qty <= 0:
        return 0.0
    return float(sum(t.price * t.qty for t in trades) / total_qty)


def top_imbalance(book: RawOrderbook) -> float:
    bid_qty = book.bids[0].qty if book.bids else 0.0
    ask_qty = book.asks[0].qty if book.asks else 0.0
    denom = bid_qty + ask_qty
    if denom <= 0:
        return 0.0
    return float((bid_qty - ask_qty) / denom)


def microprice(book: RawOrderbook) -> float:
    if not book.bids or not book.asks:
        return 0.0
    bid = book.bids[0]
    ask = book.asks[0]
    denom = bid.qty + ask.qty
    if denom <= 0:
        return float((bid.price + ask.price) / 2.0)
    return float((bid.price * ask.qty + ask.price * bid.qty) / denom)


def book_vwap_depth(book: RawOrderbook) -> float:
    """Base qty used when walking the snapshot for book buy/sell VWAP."""

    total_bid = sum(max(0.0, float(level.qty)) for level in book.bids)
    total_ask = sum(max(0.0, float(level.qty)) for level in book.asks)
    return min(total_bid, total_ask)


def book_vwaps(book: RawOrderbook) -> tuple[float, float, float]:
    """Return ``(buy_vwap, sell_vwap, depth_base)`` from the stored orderbook."""

    depth = book_vwap_depth(book)
    if depth <= 0.0 or not book.bids or not book.asks:
        return 0.0, 0.0, 0.0
    buy_vwap, sell_vwap = vwap_from_raw_orderbook(book, target_depth=depth)
    return float(buy_vwap), float(sell_vwap), float(depth)

