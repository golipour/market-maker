from __future__ import annotations

import pytest

from mm.core.models.market import PriceLevel, RawOrderbook
from mm.pricing.anchor import quote_anchor_price, two_sided_mid


def _book(
    *,
    exchange: str,
    bids: tuple[tuple[float, float], ...],
    asks: tuple[tuple[float, float], ...],
) -> RawOrderbook:
    return RawOrderbook(
        exchange=exchange,
        symbol="USDT_IRT",
        timestamp_ms=1,
        bids=tuple(PriceLevel(price, qty) for price, qty in bids),
        asks=tuple(PriceLevel(price, qty) for price, qty in asks),
        is_snapshot=True,
    )


def test_liquidity_weighted_anchor_uses_side_volume_inside_bps_window() -> None:
    quoting = _book(exchange="nobitex", bids=((99.0, 10.0),), asks=())
    liquidity = _book(exchange="tabdeal", bids=((98.0, 5.0),), asks=((102.0, 20.0),))

    anchor = quote_anchor_price(
        quoting_book=quoting,
        liquidity_book=liquidity,
        mode="liquidity_weighted_mid",
        anchor_vwap_bps=20.0,
    )

    expected_bid = (99.0 * 10.0 + 98.0 * 5.0) / 15.0
    expected_ask = 102.0
    assert anchor.bid_price == pytest.approx(expected_bid)
    assert anchor.ask_price == pytest.approx(expected_ask)
    assert anchor.price == pytest.approx((expected_bid + expected_ask) / 2.0)
    assert anchor.quoting_bid_qty == pytest.approx(10.0)
    assert anchor.quoting_ask_qty == pytest.approx(0.0)
    assert anchor.liquidity_bid_qty == pytest.approx(5.0)
    assert anchor.liquidity_ask_qty == pytest.approx(20.0)


def test_direct_mid_anchor_requires_two_sided_book() -> None:
    one_sided = _book(exchange="nobitex", bids=((99.0, 10.0),), asks=())

    with pytest.raises(ValueError, match="two-sided mid"):
        two_sided_mid(one_sided)
