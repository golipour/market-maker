"""Trading snapshot for risk checks and own-order subtraction."""

from __future__ import annotations

from collections.abc import Mapping

from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Order, TradingState
from mm.core.models.instrument import SymbolBinding
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame


def live_orders_from_managers(order_managers: Mapping[str, AbstractOrderManager]) -> dict[str, Order]:
    """Aggregate exchange live orders from all ``OrderManager`` instances."""

    out: dict[str, Order] = {}
    for manager in order_managers.values():
        for exchange_order_id, open_order in manager.live_orders.items():
            out[exchange_order_id] = open_order.order
    return out


def trading_state_from_frame(
    frame: StrategyInputFrame,
    binding: SymbolBinding,
    *,
    actor_id: str,
    open_orders: Mapping[str, Order],
) -> TradingState:
    """Build risk state from the latest frame private snapshot and live orders."""

    inventory, cash = _inventory_cash(
        frame.primary_private,
        binding.primary.base_asset,
        binding.primary.quote_asset,
    )
    return TradingState(
        actor_id=actor_id,
        cash=cash,
        inventory=inventory,
        open_orders=dict(open_orders),
    )


def _inventory_cash(
    snapshot: PrivateStateSnapshot | None,
    base_asset: str,
    quote_asset: str,
) -> tuple[float, float]:
    if snapshot is None:
        return 0.0, 0.0
    base = snapshot.balances.get(base_asset)
    quote = snapshot.balances.get(quote_asset)
    inventory = (base.free + base.locked) if base is not None else 0.0
    cash = (quote.free + quote.locked) if quote is not None else 0.0
    return float(inventory), float(cash)
