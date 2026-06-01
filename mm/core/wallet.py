"""Normalized wallet leg totals: available (free), locked, and total."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mm.core.models.execution import Order
from mm.core.models.ledger import VenueLedger
from mm.core.models.private import Balance, OpenOrder, PrivateStateSnapshot

_BALANCE_EPS = 1e-12


def make_balance(
    exchange: str,
    asset: str,
    free: float,
    locked: float,
    timestamp_ms: int,
    *,
    total: float | None = None,
) -> Balance:
    """Build a ``Balance`` with ``total`` defaulting to ``free + locked``."""

    free_f = float(free)
    locked_f = float(locked)
    total_f = float(total) if total is not None else free_f + locked_f
    return Balance(
        exchange=exchange,
        asset=asset,
        free=free_f,
        locked=locked_f,
        total=total_f,
        timestamp_ms=int(timestamp_ms),
    )


def reported_wallet_total(balance: Balance | None) -> float:
    """Exchange-reported total for one asset leg (prefers ``total``, else ``free + locked``)."""

    if balance is None:
        return 0.0
    total = float(balance.total)
    if total > _BALANCE_EPS:
        return total
    return float(balance.free) + float(balance.locked)


def order_collateral_from_open_orders(
    open_orders: Mapping[str, OpenOrder] | None,
    *,
    trading_symbol: str,
    base_asset: str,
    quote_asset: str,
) -> tuple[float, float]:
    """
    Collateral reserved in resting orders when the venue does not reflect it in ``locked``.

    Sell orders lock base ``remaining_qty``; buy orders lock quote ``price * remaining_qty``.
    """

    if not open_orders:
        return 0.0, 0.0
    symbol = trading_symbol.strip()
    base_key = base_asset.strip()
    quote_key = quote_asset.strip()
    base_locked = 0.0
    quote_locked = 0.0
    for open_order in open_orders.values():
        order = open_order.order
        if symbol and order.symbol.strip() != symbol:
            continue
        remaining = max(0.0, float(open_order.remaining_qty))
        if remaining <= 0.0:
            continue
        side = str(order.side).lower().strip()
        if side == "sell":
            base_locked += remaining
        elif side == "buy":
            quote_locked += max(0.0, float(order.price)) * remaining
    return base_locked, quote_locked


def normalize_balance_lock(
    balance: Balance | None,
    *,
    exchange: str,
    asset: str,
    timestamp_ms: int,
    required_locked: float = 0.0,
) -> Balance:
    """Return a balance where ``locked`` covers live-order collateral and ``free + locked == total``."""

    required_locked_f = max(0.0, float(required_locked))
    if balance is None:
        total = required_locked_f
        locked = required_locked_f
        free = 0.0
    else:
        total = max(reported_wallet_total(balance), required_locked_f)
        locked = min(total, max(0.0, float(balance.locked), required_locked_f))
        free = max(0.0, total - locked)
    return make_balance(exchange, asset, free, locked, timestamp_ms, total=total)


def normalize_snapshot_balances_for_open_orders(
    snapshot: PrivateStateSnapshot,
    *,
    base_asset: str,
    quote_asset: str,
    trading_symbol: str,
) -> PrivateStateSnapshot:
    """Normalize a private snapshot so wallet locked balances reflect live open-order collateral."""

    base_order_locked, quote_order_locked = order_collateral_from_open_orders(
        snapshot.open_orders,
        trading_symbol=trading_symbol,
        base_asset=base_asset,
        quote_asset=quote_asset,
    )
    balances = dict(snapshot.balances)
    exchange = snapshot.exchange
    timestamp_ms = int(snapshot.timestamp_ms)
    balances[base_asset] = normalize_balance_lock(
        balances.get(base_asset),
        exchange=exchange,
        asset=base_asset,
        timestamp_ms=timestamp_ms,
        required_locked=base_order_locked,
    )
    balances[quote_asset] = normalize_balance_lock(
        balances.get(quote_asset),
        exchange=exchange,
        asset=quote_asset,
        timestamp_ms=timestamp_ms,
        required_locked=quote_order_locked,
    )
    return PrivateStateSnapshot(
        exchange=snapshot.exchange,
        timestamp_ms=snapshot.timestamp_ms,
        balances=balances,
        open_orders=dict(snapshot.open_orders),
    )


@dataclass(frozen=True, slots=True)
class WalletLegView:
    asset: str
    free: float
    locked: float
    total: float
    order_locked: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "asset": self.asset,
            "free": self.free,
            "locked": self.locked,
            "total": self.total,
            "order_locked": self.order_locked,
        }


def wallet_leg_view(
    balance: Balance | None,
    *,
    asset: str,
    order_locked: float = 0.0,
) -> WalletLegView:
    free = float(balance.free) if balance is not None else 0.0
    locked = float(balance.locked) if balance is not None else 0.0
    total = reported_wallet_total(balance)
    return WalletLegView(
        asset=asset,
        free=free,
        locked=locked,
        total=total,
        order_locked=max(0.0, float(order_locked)),
    )


def inventory_cash_from_snapshot(
    snapshot: PrivateStateSnapshot | None,
    base_asset: str,
    quote_asset: str,
    *,
    trading_symbol: str = "",
) -> tuple[float, float]:
    """Portfolio inventory/cash using normalized wallet totals: available + locked."""

    if snapshot is None:
        return 0.0, 0.0
    del trading_symbol
    base = snapshot.balances.get(base_asset)
    quote = snapshot.balances.get(quote_asset)
    inventory = reported_wallet_total(base)
    cash = reported_wallet_total(quote)
    return float(inventory), float(cash)


def private_snapshot_from_venue_ledger(ledger: VenueLedger) -> PrivateStateSnapshot:
    return PrivateStateSnapshot(
        exchange=ledger.exchange,
        timestamp_ms=int(ledger.last_snapshot_ms or ledger.last_private_event_ms or 0),
        balances=dict(ledger.balances),
        open_orders=dict(ledger.live_orders),
    )


def inventory_cash_from_venue_ledger(ledger: VenueLedger) -> tuple[float, float]:
    return inventory_cash_from_snapshot(
        private_snapshot_from_venue_ledger(ledger),
        ledger.base_asset,
        ledger.quote_asset,
        trading_symbol=ledger.symbol,
    )


def wallet_pair_from_snapshot(
    snapshot: PrivateStateSnapshot | None,
    base_asset: str,
    quote_asset: str,
    *,
    trading_symbol: str = "",
) -> tuple[WalletLegView, WalletLegView]:
    if snapshot is None:
        empty_base = wallet_leg_view(None, asset=base_asset)
        empty_quote = wallet_leg_view(None, asset=quote_asset)
        return empty_base, empty_quote
    base_order_locked, quote_order_locked = order_collateral_from_open_orders(
        snapshot.open_orders,
        trading_symbol=trading_symbol,
        base_asset=base_asset,
        quote_asset=quote_asset,
    )
    base = wallet_leg_view(
        snapshot.balances.get(base_asset),
        asset=base_asset,
        order_locked=base_order_locked,
    )
    quote = wallet_leg_view(
        snapshot.balances.get(quote_asset),
        asset=quote_asset,
        order_locked=quote_order_locked,
    )
    return base, quote


def wallet_pair_from_venue_ledger(ledger: VenueLedger) -> tuple[WalletLegView, WalletLegView]:
    return wallet_pair_from_snapshot(
        private_snapshot_from_venue_ledger(ledger),
        ledger.base_asset,
        ledger.quote_asset,
        trading_symbol=ledger.symbol,
    )


def portfolio_wealth(
    cash: float,
    inventory: float,
    mark_price: float,
    *,
    min_wealth: float = 1e-8,
) -> float:
    """Mark portfolio wealth for return normalization: cash + inventory * mark."""

    wealth = float(cash) + float(inventory) * float(mark_price)
    return max(float(min_wealth), wealth)


def balance_event_payload(balance: Balance) -> dict[str, float | str | int]:
    return {
        "exchange": balance.exchange,
        "asset": balance.asset,
        "free": float(balance.free),
        "locked": float(balance.locked),
        "total": float(balance.total),
        "timestamp_ms": int(balance.timestamp_ms),
    }
