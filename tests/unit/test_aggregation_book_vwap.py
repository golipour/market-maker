from __future__ import annotations

from mm.aggregation.features import book_vwaps
from mm.core.models.market import PriceLevel, RawOrderbook


def test_book_vwaps_use_min_total_side_qty_as_depth() -> None:
    book = RawOrderbook(
        exchange="nobitex",
        symbol="USDT_IRT",
        timestamp_ms=1_000,
        bids=(
            PriceLevel(99.0, 5.0),
            PriceLevel(98.0, 10.0),
        ),
        asks=(
            PriceLevel(101.0, 3.0),
            PriceLevel(102.0, 20.0),
        ),
        is_snapshot=True,
    )

    buy_vwap, sell_vwap, depth = book_vwaps(book)

    assert depth == 15.0
    assert buy_vwap == (3.0 * 101.0 + 12.0 * 102.0) / 15.0
    assert sell_vwap == (5.0 * 99.0 + 10.0 * 98.0) / 15.0


def test_book_vwaps_return_zero_when_book_is_one_sided() -> None:
    book = RawOrderbook(
        exchange="nobitex",
        symbol="USDT_IRT",
        timestamp_ms=1_000,
        bids=(PriceLevel(99.0, 5.0),),
        asks=(),
        is_snapshot=True,
    )

    buy_vwap, sell_vwap, depth = book_vwaps(book)

    assert depth == 0.0
    assert buy_vwap == 0.0
    assert sell_vwap == 0.0
