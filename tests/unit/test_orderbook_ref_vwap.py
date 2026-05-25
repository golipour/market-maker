"""Tests for ``mm.pricing.orderbook_ref`` VWAP helpers."""

from __future__ import annotations

import pytest
import torch

from mm.core.models.market import PriceLevel, RawOrderbook
from mm.pricing.orderbook_ref import mid_raw_orderbook, vwap_from_raw_orderbook
from mm.pricing.vwap_batched import vwap_prices_batched


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


def test_vwap_from_raw_orderbook_depth_min_of_sides_and_cap() -> None:
    book = _book()
    # total_bid=40, total_ask=20 -> depth_used = min(15, 40, 20) = 15
    buy_vwap, sell_vwap = vwap_from_raw_orderbook(book, target_depth=15.0)
    # Buy: 10 @ 101 + 5 @ 102
    assert buy_vwap == pytest.approx((10 * 101 + 5 * 102) / 15.0)
    # Sell: 15 @ 99
    assert sell_vwap == pytest.approx(99.0)


def test_vwap_from_raw_orderbook_falls_back_to_mid_when_thin() -> None:
    book = RawOrderbook(
        exchange="x",
        symbol="y",
        timestamp_ms=0,
        bids=(PriceLevel(100.0, 0.0),),
        asks=(PriceLevel(101.0, 0.0),),
        is_snapshot=True,
    )
    mid = mid_raw_orderbook(book)
    b, s = vwap_from_raw_orderbook(book, target_depth=100.0)
    assert b == pytest.approx(mid)
    assert s == pytest.approx(mid)


def test_batched_vwap_accepts_per_sample_depth() -> None:
    ob_bids = torch.tensor([[5.0, 10.0], [5.0, 10.0]], dtype=torch.float64)
    ob_asks = torch.tensor([[5.0, 10.0], [5.0, 10.0]], dtype=torch.float64)
    mid = torch.tensor([[100.0], [100.0]], dtype=torch.float64)
    spread = torch.tensor([1.0, 2.0], dtype=torch.float64)
    depth = torch.tensor([[5.0], [15.0]], dtype=torch.float64)

    buy_vwap, sell_vwap = vwap_prices_batched(
        ob_bids,
        ob_asks,
        mid,
        spread,
        target_depth=depth,
    )

    assert float(buy_vwap[0, 0]) == pytest.approx(101.0)
    assert float(sell_vwap[0, 0]) == pytest.approx(99.0)
    assert float(buy_vwap[1, 0]) == pytest.approx((5.0 * 101.0 + 10.0 * 102.0) / 15.0)
    assert float(sell_vwap[1, 0]) == pytest.approx((5.0 * 99.0 + 10.0 * 98.0) / 15.0)
