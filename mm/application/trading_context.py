"""Trading snapshot for risk checks and own-order subtraction."""

from __future__ import annotations

from collections.abc import Mapping

from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Order, TradingState
from mm.core.models.instrument import SymbolBinding
from mm.core.models.ledger import JointLedgerSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.core.wallet import inventory_cash_from_snapshot, inventory_cash_from_venue_ledger
from mm.portfolio.wallet import StrategyWallet


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

    inventory, cash = inventory_cash_from_snapshot(
        frame.local_private,
        binding.local.base_asset,
        binding.local.quote_asset,
        trading_symbol=binding.local.symbol,
    )
    return TradingState(
        actor_id=actor_id,
        cash=cash,
        inventory=inventory,
        open_orders=dict(open_orders),
    )


def trading_state_from_ledger(
    ledger: JointLedgerSnapshot,
    *,
    actor_id: str,
    open_orders: Mapping[str, Order],
) -> TradingState:
    """Build risk state from the ledger-owned local wallet and live orders."""

    inventory, cash = inventory_cash_from_venue_ledger(ledger.local)
    return TradingState(
        actor_id=actor_id,
        cash=cash,
        inventory=inventory,
        open_orders=dict(open_orders),
    )


def trading_state_from_wallet(
    wallet: StrategyWallet,
    *,
    actor_id: str,
    open_orders: Mapping[str, Order],
) -> TradingState:
    """Build MPC/risk state from the strategy virtual wallet (not the exchange ledger)."""

    return TradingState(
        actor_id=actor_id,
        cash=max(0.0, wallet.free_quote),
        inventory=max(0.0, wallet.free_base),
        open_orders=dict(open_orders),
    )
