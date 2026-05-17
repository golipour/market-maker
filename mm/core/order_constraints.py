"""Order pipeline: MPC ladder → venue rules (``InstrumentSpec``) → strategy caps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mm.core.models.execution import Order
from mm.core.models.instrument import InstrumentSpec
from mm.core.models.market import MarketEvent, RawOrderbook


@dataclass(frozen=True, slots=True)
class OrderConstraints:
    """Resolved venue + strategy limits for one trading symbol."""

    price_step: float | None
    qty_step: float | None
    min_qty: float | None
    min_notional: float | None
    max_orders_per_side: int

    @classmethod
    def from_spec(cls, spec: InstrumentSpec, *, max_orders_per_side: int) -> OrderConstraints:
        return cls(
            price_step=spec.price_step,
            qty_step=spec.qty_step,
            min_qty=spec.min_qty,
            min_notional=spec.min_notional,
            max_orders_per_side=max(0, int(max_orders_per_side)),
        )

def mid_from_orderbook(book: RawOrderbook) -> float:
    if book.bids and book.asks:
        return 0.5 * (book.bids[0].price + book.asks[0].price)
    if book.bids:
        return book.bids[0].price
    if book.asks:
        return book.asks[0].price
    raise ValueError("cannot build mid from empty orderbook")


def ladder_orders_from_sigmoid(
    *,
    actor_id: str,
    event: MarketEvent,
    sigmoid_action: Any,
    orderbook_grid: Any,
) -> list[Order]:
    """Marginal ladder on ``orderbook_grid.spread_ladder()`` (no venue or cap rules)."""
    mid = mid_from_orderbook(event.orderbook)
    levels = list(orderbook_grid.spread_ladder())
    orders: list[Order] = []
    orders.extend(
        _ladder_side_orders(
            event=event,
            actor_id=actor_id,
            side="buy",
            mid=mid,
            spreads=levels,
            depth_fn=sigmoid_action.depth_bid,
        )
    )
    orders.extend(
        _ladder_side_orders(
            event=event,
            actor_id=actor_id,
            side="sell",
            mid=mid,
            spreads=levels,
            depth_fn=sigmoid_action.depth_ask,
        )
    )
    return orders


def apply_venue_constraints(orders: list[Order], constraints: OrderConstraints) -> list[Order]:
    """Round to tick/lot and drop orders below exchange minimums."""
    out: list[Order] = []
    for order in orders:
        price = _quantize_price(order.price, constraints.price_step)
        qty = _quantize_qty(order.qty, constraints.qty_step)
        if qty <= 0.0 or price <= 0.0 or not math.isfinite(price) or not math.isfinite(qty):
            continue
        if constraints.min_qty is not None and qty < float(constraints.min_qty):
            continue
        if constraints.min_notional is not None and price * qty < float(constraints.min_notional):
            continue
        out.append(
            Order(
                exchange=order.exchange,
                symbol=order.symbol,
                side=order.side,
                price=float(price),
                qty=float(qty),
                client_order_id=order.client_order_id,
            )
        )
    return out


def cap_orders_per_side(orders: list[Order], max_per_side: int) -> list[Order]:
    """Keep the largest-qty orders per side (strategy cap)."""
    cap = max(0, int(max_per_side))
    if cap == 0:
        return []
    buys = sorted((o for o in orders if o.side.lower().strip() == "buy"), key=lambda o: o.qty, reverse=True)
    sells = sorted((o for o in orders if o.side.lower().strip() == "sell"), key=lambda o: o.qty, reverse=True)
    return list(buys[:cap]) + list(sells[:cap])


def build_orders_from_sigmoid(
    *,
    actor_id: str,
    event: MarketEvent,
    sigmoid_action: Any,
    orderbook_grid: Any,
    constraints: OrderConstraints,
) -> list[Order]:
    ladder = ladder_orders_from_sigmoid(
        actor_id=actor_id,
        event=event,
        sigmoid_action=sigmoid_action,
        orderbook_grid=orderbook_grid,
    )
    venued = apply_venue_constraints(ladder, constraints)
    return cap_orders_per_side(venued, constraints.max_orders_per_side)


def _ladder_side_orders(
    *,
    event: MarketEvent,
    actor_id: str,
    side: str,
    mid: float,
    spreads: list[float],
    depth_fn: Any,
) -> list[Order]:
    out: list[Order] = []
    prev_cumulative = float(depth_fn(0.0))
    for idx, spread in enumerate(spreads):
        cumulative = float(depth_fn(spread))
        qty = max(0.0, cumulative - prev_cumulative)
        prev_cumulative = cumulative
        if qty <= 0.0:
            continue
        price = mid - spread if side == "buy" else mid + spread
        if price <= 0 or not math.isfinite(price):
            continue
        out.append(
            Order(
                exchange=event.exchange,
                symbol=event.symbol,
                side=side,
                price=float(price),
                qty=float(qty),
                client_order_id=f"{actor_id}:{side}:{event.t_end_ms}:{idx}",
            )
        )
    return out


def _quantize_qty(qty: float, qty_step: float | None) -> float:
    if qty_step is None or float(qty_step) <= 0.0:
        return float(qty)
    step = float(qty_step)
    return math.floor(float(qty) / step) * step


def _quantize_price(price: float, price_step: float | None) -> float:
    if price_step is None or float(price_step) <= 0.0:
        return float(price)
    step = float(price_step)
    return math.floor(float(price) / step) * step
