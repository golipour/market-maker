from __future__ import annotations

from mm.core.models.execution import Order
from mm.core.models.market import PriceLevel, RawOrderbook


DEFAULT_PRICE_TOLERANCE_BPS = 1e-6


def subtract_own_orders(
    book: RawOrderbook,
    open_orders: dict[str, Order],
    *,
    price_tolerance_bps: float = DEFAULT_PRICE_TOLERANCE_BPS,
) -> RawOrderbook:
    bids = [(float(level.price), float(level.qty)) for level in book.bids]
    asks = [(float(level.price), float(level.qty)) for level in book.asks]

    for order in open_orders.values():
        if order.exchange != book.exchange or order.symbol != book.symbol:
            continue
        side = bids if order.side == "buy" else asks
        _subtract_at_nearest_price(
            side,
            price=float(order.price),
            qty=float(order.qty),
            tolerance_bps=price_tolerance_bps,
        )

    clean_bids = tuple(PriceLevel(price, qty) for price, qty in sorted(bids, reverse=True))
    clean_asks = tuple(PriceLevel(price, qty) for price, qty in sorted(asks))
    return RawOrderbook(
        exchange=book.exchange,
        symbol=book.symbol,
        timestamp_ms=book.timestamp_ms,
        bids=clean_bids,
        asks=clean_asks,
        is_snapshot=True,
    )


def _subtract_at_nearest_price(
    levels: list[tuple[float, float]],
    *,
    price: float,
    qty: float,
    tolerance_bps: float,
) -> None:
    if not levels:
        return
    idx = min(range(len(levels)), key=lambda i: abs(levels[i][0] - price))
    level_price, level_qty = levels[idx]
    tolerance = _price_tolerance(level_price, price, tolerance_bps)
    if abs(level_price - price) > tolerance:
        return
    remaining = level_qty - qty
    if remaining <= 0:
        del levels[idx]
    else:
        levels[idx] = (level_price, remaining)


def _price_tolerance(level_price: float, order_price: float, tolerance_bps: float) -> float:
    base = max(abs(level_price), abs(order_price), 1.0)
    rel_tol = max(0.0, float(tolerance_bps)) / 10_000.0
    return max(1e-12, base * rel_tol)
