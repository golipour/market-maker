"""Tests for ``mm.pricing.orderbook_ref`` VWAP helpers."""

from __future__ import annotations

import pytest

from mm.core.models.market import PriceLevel, RawOrderbook
from mm.pricing.orderbook_ref import mid_raw_orderbook, reference_vwaps_raw_orderbook


def _book() -> RawOrderbook:
    # Bids best-first (high price), asks best-first (low price).
    bids = (PriceLevel(99.0, 20.0), PriceLevel(98.0, 20.0))
    asks = (PriceLevel(101.0, 10.0), PriceLevel(102.0, 10.0))
    return RawOrderbook(
        exchange="x",
        symbol="y",
        timestamp_ms=0,
        bids=bids,
        asks=asks,
        is_snapshot=True,
    )


def test_reference_vwaps_depth_min_of_sides_and_cap() -> None:
    book = _book()
    # total_bid=40, total_ask=20 -> depth_used = min(15, 40, 20) = 15
    buy_vwap, sell_vwap = reference_vwaps_raw_orderbook(book, target_depth=15.0)
    # Buy: 10 @ 101 + 5 @ 102
    assert buy_vwap == pytest.approx((10 * 101 + 5 * 102) / 15.0)
    # Sell: 15 @ 99
    assert sell_vwap == pytest.approx(99.0)


def test_reference_vwaps_falls_back_to_mid_when_thin() -> None:
    book = RawOrderbook(
        exchange="x",
        symbol="y",
        timestamp_ms=0,
        bids=(PriceLevel(100.0, 0.0),),
        asks=(PriceLevel(101.0, 0.0),),
        is_snapshot=True,
    )
    mid = mid_raw_orderbook(book)
    b, s = reference_vwaps_raw_orderbook(book, target_depth=100.0)
    assert b == pytest.approx(mid)
    assert s == pytest.approx(mid)
