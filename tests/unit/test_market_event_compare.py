from __future__ import annotations

import pytest

from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.ingestion.market_event_compare import compare_market_events


def _event(*, t_end_ms: int, microprice: float = 100.5) -> MarketEvent:
    book = RawOrderbook(
        exchange="wallex",
        symbol="BTC_IRT",
        timestamp_ms=t_end_ms,
        bids=(PriceLevel(100.0, 5.0),),
        asks=(PriceLevel(101.0, 4.0),),
        is_snapshot=True,
    )
    return MarketEvent(
        exchange="wallex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=t_end_ms - 1000,
        t_end_ms=t_end_ms,
        orderbook=book,
        trades=(),
        buy_volume=1.0,
        sell_volume=0.5,
        vwap=100.5,
        imbalance=0.1,
        microprice=microprice,
    )


def test_compare_market_events_reports_no_differences_for_identical_events() -> None:
    event = _event(t_end_ms=5000)
    result = compare_market_events(event, event)
    assert result.equal
    assert result.differences == ()


def test_compare_market_events_reports_field_mismatch() -> None:
    left = _event(t_end_ms=5000)
    right = _event(t_end_ms=5000, microprice=101.0)
    result = compare_market_events(left, right)
    assert not result.equal
    assert any(diff.startswith("microprice:") for diff in result.differences)
