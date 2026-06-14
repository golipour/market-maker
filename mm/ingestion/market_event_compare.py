"""Compare local and bus-delivered ``MarketEvent`` payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mm.core.models.market import MarketEvent
from mm.core.payload_schema import market_event_from_payload


@dataclass(frozen=True, slots=True)
class MarketEventComparison:
    equal: bool
    differences: tuple[str, ...]


def market_event_from_bus_payload(payload: Any) -> MarketEvent:
    parsed = market_event_from_payload(payload)
    if parsed is None:
        raise TypeError(f"expected versioned MarketEvent payload, got {type(payload)!r}")
    return parsed


def compare_market_events(
    left: MarketEvent,
    right: MarketEvent,
    *,
    price_tol: float = 1e-9,
    qty_tol: float = 1e-9,
    volume_tol: float = 1e-9,
) -> MarketEventComparison:
    diffs: list[str] = []

    def _check(label: str, a: Any, b: Any) -> None:
        if a != b:
            diffs.append(f"{label}: {a!r} != {b!r}")

    def _check_float(label: str, a: float, b: float, tol: float) -> None:
        if abs(float(a) - float(b)) > tol:
            diffs.append(f"{label}: {a!r} != {b!r} (tol={tol})")

    _check("exchange", left.exchange, right.exchange)
    _check("symbol", left.symbol, right.symbol)
    _check("interval_ms", left.interval_ms, right.interval_ms)
    _check("t_start_ms", left.t_start_ms, right.t_start_ms)
    _check("t_end_ms", left.t_end_ms, right.t_end_ms)
    _check("trade_count", len(left.trades), len(right.trades))
    _check_float("buy_volume", left.buy_volume, right.buy_volume, volume_tol)
    _check_float("sell_volume", left.sell_volume, right.sell_volume, volume_tol)
    _check_float("vwap", left.vwap, right.vwap, price_tol)
    _check_float("imbalance", left.imbalance, right.imbalance, volume_tol)
    _check_float("microprice", left.microprice, right.microprice, price_tol)

    if left.orderbook.bids and right.orderbook.bids:
        _check_float("best_bid", left.orderbook.bids[0].price, right.orderbook.bids[0].price, price_tol)
        _check_float("best_bid_qty", left.orderbook.bids[0].qty, right.orderbook.bids[0].qty, qty_tol)
    else:
        _check("best_bid_present", bool(left.orderbook.bids), bool(right.orderbook.bids))

    if left.orderbook.asks and right.orderbook.asks:
        _check_float("best_ask", left.orderbook.asks[0].price, right.orderbook.asks[0].price, price_tol)
        _check_float("best_ask_qty", left.orderbook.asks[0].qty, right.orderbook.asks[0].qty, qty_tol)
    else:
        _check("best_ask_present", bool(left.orderbook.asks), bool(right.orderbook.asks))

    return MarketEventComparison(equal=not diffs, differences=tuple(diffs))
