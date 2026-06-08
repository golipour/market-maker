"""Quote-anchor prices from one or two clean orderbooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mm.core.models.market import RawOrderbook

QuoteAnchorMode = Literal["quoting_mid", "liquidity_mid", "liquidity_weighted_mid"]


@dataclass(frozen=True, slots=True)
class SideWindowVwap:
    price: float
    qty: float


@dataclass(frozen=True, slots=True)
class AnchorPriceResult:
    price: float
    mode: QuoteAnchorMode
    anchor_vwap_bps: float
    bid_price: float | None = None
    ask_price: float | None = None
    quoting_bid_qty: float = 0.0
    quoting_ask_qty: float = 0.0
    liquidity_bid_qty: float = 0.0
    liquidity_ask_qty: float = 0.0


def two_sided_mid(book: RawOrderbook) -> float:
    """Best bid/ask mid; quote anchors require both sides for direct mid modes."""

    if not book.bids or not book.asks:
        raise ValueError("cannot compute two-sided mid from one-sided or empty orderbook")
    return 0.5 * (float(book.bids[0].price) + float(book.asks[0].price))


def quote_anchor_price(
    *,
    quoting_book: RawOrderbook,
    liquidity_book: RawOrderbook,
    mode: QuoteAnchorMode,
    anchor_vwap_bps: float,
) -> AnchorPriceResult:
    """Compute the configured quote anchor.

    ``liquidity_weighted_mid`` uses side VWAPs inside ``anchor_vwap_bps`` from
    each venue's best bid/ask. The side weights are exactly the base volume
    found in that bps window.
    """

    mode = str(mode).strip().lower()
    if mode == "quoting_mid":
        return AnchorPriceResult(
            price=two_sided_mid(quoting_book),
            mode="quoting_mid",
            anchor_vwap_bps=float(anchor_vwap_bps),
        )
    if mode == "liquidity_mid":
        return AnchorPriceResult(
            price=two_sided_mid(liquidity_book),
            mode="liquidity_mid",
            anchor_vwap_bps=float(anchor_vwap_bps),
        )
    if mode != "liquidity_weighted_mid":
        raise ValueError(
            "quote_anchor.mode must be 'quoting_mid', 'liquidity_mid', or 'liquidity_weighted_mid'"
        )

    bps = max(0.0, float(anchor_vwap_bps))
    q_bid = side_window_vwap(quoting_book, side="bid", window_bps=bps)
    q_ask = side_window_vwap(quoting_book, side="ask", window_bps=bps)
    l_bid = side_window_vwap(liquidity_book, side="bid", window_bps=bps)
    l_ask = side_window_vwap(liquidity_book, side="ask", window_bps=bps)
    bid_price = _weighted_side_price(q_bid, l_bid)
    ask_price = _weighted_side_price(q_ask, l_ask)
    if bid_price is None or ask_price is None:
        raise ValueError("liquidity_weighted_mid requires at least one valid bid window and ask window")
    return AnchorPriceResult(
        price=0.5 * (bid_price + ask_price),
        mode="liquidity_weighted_mid",
        anchor_vwap_bps=bps,
        bid_price=bid_price,
        ask_price=ask_price,
        quoting_bid_qty=q_bid.qty if q_bid is not None else 0.0,
        quoting_ask_qty=q_ask.qty if q_ask is not None else 0.0,
        liquidity_bid_qty=l_bid.qty if l_bid is not None else 0.0,
        liquidity_ask_qty=l_ask.qty if l_ask is not None else 0.0,
    )


def side_window_vwap(
    book: RawOrderbook,
    *,
    side: Literal["bid", "ask"],
    window_bps: float,
) -> SideWindowVwap | None:
    levels = book.bids if side == "bid" else book.asks
    if not levels:
        return None
    best = float(levels[0].price)
    if best <= 0.0:
        return None
    band = max(0.0, float(window_bps)) / 10_000.0
    min_bid = best * (1.0 - band)
    max_ask = best * (1.0 + band)
    qty = 0.0
    notional = 0.0
    for level in levels:
        price = float(level.price)
        if side == "bid" and price < min_bid:
            break
        if side == "ask" and price > max_ask:
            break
        level_qty = max(0.0, float(level.qty))
        if level_qty <= 0.0:
            continue
        qty += level_qty
        notional += price * level_qty
    if qty <= 0.0:
        return None
    return SideWindowVwap(price=notional / qty, qty=qty)


def _weighted_side_price(
    quoting: SideWindowVwap | None,
    liquidity: SideWindowVwap | None,
) -> float | None:
    total_qty = (quoting.qty if quoting is not None else 0.0) + (
        liquidity.qty if liquidity is not None else 0.0
    )
    if total_qty <= 0.0:
        return None
    notional = (quoting.price * quoting.qty if quoting is not None else 0.0) + (
        liquidity.price * liquidity.qty if liquidity is not None else 0.0
    )
    return float(notional / total_qty)
