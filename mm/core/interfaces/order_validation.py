"""Pluggable venue limits for pre-flight order validation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from mm.core.models.execution import Order
from mm.core.models.venue_order_limits import VenueOrderLimits


class AbstractVenueOrderLimitSource(ABC):
    """Exchange-specific provider of ``VenueOrderLimits`` keyed by trading symbol."""

    @abstractmethod
    def limits_for(self, symbol: str) -> VenueOrderLimits:
        """Return limits for ``symbol`` (cached inside implementations as needed)."""
        ...


def order_passes_limits(order: Order, limits: VenueOrderLimits, *, rel_tol: float = 1e-9) -> tuple[bool, str]:
    """Return (True, \"\") if ``order`` satisfies ``limits``; else (False, short reason)."""

    price = float(order.price)
    qty = float(order.qty)
    if price <= 0.0 or qty <= 0.0 or not math.isfinite(price) or not math.isfinite(qty):
        return False, "rules_invalid_price_or_qty"
    if limits.min_base_qty > 0.0 and qty + rel_tol < float(limits.min_base_qty):
        return False, "rules_min_qty"
    notional = price * qty
    if limits.min_notional_quote > 0.0 and notional + rel_tol < float(limits.min_notional_quote):
        return False, "rules_min_notional"
    return True, ""


def normalize_order_to_limits(order: Order, limits: VenueOrderLimits) -> Order:
    """Round an order down to venue tick/lot steps at the execution boundary."""

    price = _quantize_down(float(order.price), float(getattr(limits, "price_step", 0.0)))
    qty = _quantize_down(float(order.qty), float(getattr(limits, "qty_step", 0.0)))
    return Order(
        exchange=order.exchange,
        symbol=order.symbol,
        side=order.side,
        price=price,
        qty=qty,
        order_type=order.order_type,
        client_order_id=order.client_order_id,
    )


def _quantize_down(value: float, step: float) -> float:
    if step <= 0.0:
        return float(value)
    if value <= 0.0 or not math.isfinite(value):
        return float(value)
    return math.floor(value / step) * step
