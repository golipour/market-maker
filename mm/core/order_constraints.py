"""Order pipeline helpers for strategy-owned ladder construction and caps.

Venue tick/lot quantization and min-size validation belong to the execution
boundary. Strategy construction remains float-first so the target book shows
what the strategy decided before exchange formatting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mm.core.models.execution import Order
from mm.core.models.instrument import InstrumentSpec
from mm.core.models.market import MarketEvent, RawOrderbook
from mm.execution.plan import ActionPipelineReport, ActionStageReport, stage_report_with_rejections


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
    anchor_price: float | None = None,
) -> list[Order]:
    """Marginal ladder on ``orderbook_grid.spread_ladder()`` (no venue or cap rules)."""
    orders, _report = ladder_orders_from_sigmoid_with_report(
        actor_id=actor_id,
        event=event,
        sigmoid_action=sigmoid_action,
        orderbook_grid=orderbook_grid,
        anchor_price=anchor_price,
    )
    return orders


def ladder_orders_from_sigmoid_with_report(
    *,
    actor_id: str,
    event: MarketEvent,
    sigmoid_action: Any,
    orderbook_grid: Any,
    anchor_price: float | None = None,
) -> tuple[list[Order], ActionStageReport]:
    """Marginal ladder with a stage summary for diagnostics."""
    mid = float(anchor_price) if anchor_price is not None else mid_from_orderbook(event.orderbook)
    levels = list(orderbook_grid.spread_ladder())
    rejection_counts: dict[str, int] = {}
    orders: list[Order] = []
    orders.extend(
        _ladder_side_orders(
            event=event,
            actor_id=actor_id,
            side="buy",
            mid=mid,
            spreads=levels,
            depth_fn=_side_depth_fn(sigmoid_action, "buy"),
            rejection_counts=rejection_counts,
        )
    )
    orders.extend(
        _ladder_side_orders(
            event=event,
            actor_id=actor_id,
            side="sell",
            mid=mid,
            spreads=levels,
            depth_fn=_side_depth_fn(sigmoid_action, "sell"),
            rejection_counts=rejection_counts,
        )
    )
    return orders, stage_report_with_rejections(orders, rejection_counts)


def apply_venue_constraints(orders: list[Order], constraints: OrderConstraints) -> list[Order]:
    """Round to tick/lot and drop orders below exchange minimums."""
    out, _report = apply_venue_constraints_with_report(orders, constraints)
    return out


def apply_venue_constraints_with_report(
    orders: list[Order], constraints: OrderConstraints
) -> tuple[list[Order], ActionStageReport]:
    """Round to tick/lot; return kept orders and per-reason rejection counts."""
    out: list[Order] = []
    rejection_counts: dict[str, int] = {}
    for order in orders:
        price = _quantize_price(order.price, constraints.price_step)
        qty = _quantize_qty(order.qty, constraints.qty_step)
        if qty <= 0.0 or price <= 0.0 or not math.isfinite(price) or not math.isfinite(qty):
            rejection_counts["invalid_price_or_qty"] = rejection_counts.get("invalid_price_or_qty", 0) + 1
            continue
        if constraints.min_qty is not None and qty < float(constraints.min_qty):
            rejection_counts["below_min_qty"] = rejection_counts.get("below_min_qty", 0) + 1
            continue
        if constraints.min_notional is not None and price * qty < float(constraints.min_notional):
            rejection_counts["below_min_notional"] = rejection_counts.get("below_min_notional", 0) + 1
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
    return out, stage_report_with_rejections(out, rejection_counts)


def cap_orders_per_side(orders: list[Order], max_per_side: int) -> list[Order]:
    """Keep the largest-qty orders per side (strategy cap)."""
    out, _report = cap_orders_per_side_with_report(orders, max_per_side)
    return out


def cap_orders_per_side_with_report(
    orders: list[Order], max_per_side: int
) -> tuple[list[Order], ActionStageReport]:
    """Keep largest-qty orders per side; report how many were dropped per side."""
    cap = max(0, int(max_per_side))
    rejection_counts: dict[str, int] = {}
    if cap == 0:
        if orders:
            rejection_counts["max_orders_per_side_zero"] = len(orders)
        return [], stage_report_with_rejections([], rejection_counts)
    buys = sorted((o for o in orders if o.side.lower().strip() == "buy"), key=lambda o: o.qty, reverse=True)
    sells = sorted((o for o in orders if o.side.lower().strip() == "sell"), key=lambda o: o.qty, reverse=True)
    dropped_buys = max(0, len(buys) - cap)
    dropped_sells = max(0, len(sells) - cap)
    if dropped_buys:
        rejection_counts["capped_buy"] = dropped_buys
    if dropped_sells:
        rejection_counts["capped_sell"] = dropped_sells
    kept = list(buys[:cap]) + list(sells[:cap])
    return kept, stage_report_with_rejections(kept, rejection_counts)


def build_orders_from_sigmoid(
    *,
    actor_id: str,
    event: MarketEvent,
    sigmoid_action: Any,
    orderbook_grid: Any,
    constraints: OrderConstraints,
    anchor_price: float | None = None,
) -> list[Order]:
    orders, _pipeline, _ladder = build_orders_from_sigmoid_with_report(
        actor_id=actor_id,
        event=event,
        sigmoid_action=sigmoid_action,
        orderbook_grid=orderbook_grid,
        constraints=constraints,
        anchor_price=anchor_price,
    )
    return orders


def build_orders_from_sigmoid_with_report(
    *,
    actor_id: str,
    event: MarketEvent,
    sigmoid_action: Any,
    orderbook_grid: Any,
    constraints: OrderConstraints,
    continuous: dict[str, Any] | None = None,
    anchor_price: float | None = None,
) -> tuple[list[Order], ActionPipelineReport, list[Order]]:
    ladder, ladder_report = ladder_orders_from_sigmoid_with_report(
        actor_id=actor_id,
        event=event,
        sigmoid_action=sigmoid_action,
        orderbook_grid=orderbook_grid,
        anchor_price=anchor_price,
    )
    capped, capped_report = cap_orders_per_side_with_report(ladder, constraints.max_orders_per_side)
    pipeline = ActionPipelineReport(
        continuous=dict(continuous or {}),
        ladder=ladder_report,
        capped=capped_report,
    )
    return capped, pipeline, ladder


def _ladder_side_orders(
    *,
    event: MarketEvent,
    actor_id: str,
    side: str,
    mid: float,
    spreads: list[float],
    depth_fn: Any,
    rejection_counts: dict[str, int] | None = None,
) -> list[Order]:
    out: list[Order] = []
    prev_cumulative = 0.0
    rejects = rejection_counts if rejection_counts is not None else {}
    for idx, spread in enumerate(spreads):
        cumulative = float(depth_fn(spread))
        qty = max(0.0, cumulative - prev_cumulative)
        prev_cumulative = cumulative
        if qty <= 0.0:
            rejects["zero_marginal_qty"] = rejects.get("zero_marginal_qty", 0) + 1
            continue
        price = mid - spread if side == "buy" else mid + spread
        if price <= 0 or not math.isfinite(price):
            rejects["invalid_price"] = rejects.get("invalid_price", 0) + 1
            continue
        out.append(
            Order(
                exchange=event.exchange,
                symbol=event.symbol,
                side=side,
                price=float(price),
                qty=float(qty),
                client_order_id=f"{actor_id}:{side}:{idx}",
            )
        )
    return out


def _side_depth_fn(sigmoid_action: Any, side: str) -> Any:
    if side == "buy":
        effective = getattr(sigmoid_action, "effective_depth_bid", None)
        return effective if callable(effective) else _baseline_shifted_depth(getattr(sigmoid_action, "depth_bid"))
    effective = getattr(sigmoid_action, "effective_depth_ask", None)
    return effective if callable(effective) else _baseline_shifted_depth(getattr(sigmoid_action, "depth_ask"))


def _baseline_shifted_depth(raw_depth_fn: Any) -> Any:
    base = float(raw_depth_fn(0.0))

    def _depth(spread: float) -> float:
        return max(0.0, float(raw_depth_fn(spread)) - base)

    return _depth


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
