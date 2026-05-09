from __future__ import annotations

from mm.core.models.market import RawOrderbook, RawTrade


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

