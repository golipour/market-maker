from __future__ import annotations

from mm.core.models.execution import Order
from mm.core.models.market import PriceLevel, RawOrderbook


def subtract_own_orders(book: RawOrderbook, open_orders: dict[str, Order]) -> RawOrderbook:
    bids = {level.price: level.qty for level in book.bids}
    asks = {level.price: level.qty for level in book.asks}

    for order in open_orders.values():
        if order.exchange != book.exchange or order.symbol != book.symbol:
            continue
        side = bids if order.side == "buy" else asks
        side[order.price] = side.get(order.price, 0.0) - order.qty
        if side[order.price] <= 0:
            del side[order.price]

    clean_bids = tuple(PriceLevel(price, qty) for price, qty in sorted(bids.items(), reverse=True))
    clean_asks = tuple(PriceLevel(price, qty) for price, qty in sorted(asks.items()))
    return RawOrderbook(
        exchange=book.exchange,
        symbol=book.symbol,
        timestamp_ms=book.timestamp_ms,
        bids=clean_bids,
        asks=clean_asks,
        is_snapshot=True,
    )

