"""Hard risk checks for hedge market orders."""

from __future__ import annotations

from mm.core.models.execution import Order
from mm.core.risk_limits import HedgeVenueRiskLimits
from mm.exchanges.capabilities import supports_market_orders


def validate_hedge_market_order(
    order: Order,
    *,
    limits: HedgeVenueRiskLimits,
    hedge_inventory: float,
    hedge_cash: float,
) -> tuple[bool, str | None]:
    if not supports_market_orders(order.exchange):
        return False, "market_orders_unsupported"

    qty = float(order.qty)
    if qty <= 0.0:
        return False, "invalid_qty"

    notional = float(order.price) * qty if float(order.price) > 0.0 else qty
    if notional > limits.max_market_order_notional:
        return False, "max_market_order_notional"

    side = order.side.lower().strip()
    if side == "sell":
        if qty > hedge_inventory:
            return False, "insufficient_hedge_inventory"
        if hedge_inventory - qty < limits.min_inventory:
            return False, "min_hedge_inventory"
    elif side == "buy":
        if hedge_cash - notional < limits.min_cash:
            return False, "min_hedge_cash"
        if hedge_inventory + qty > limits.max_inventory:
            return False, "max_hedge_inventory"
    else:
        return False, "invalid_side"
    return True, None
