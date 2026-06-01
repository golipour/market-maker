from __future__ import annotations

from mm.core.models.execution import Order
from mm.core.models.private import Balance, OpenOrder, PrivateStateSnapshot
from mm.core.wallet import (
    inventory_cash_from_snapshot,
    make_balance,
    normalize_snapshot_balances_for_open_orders,
    order_collateral_from_open_orders,
    portfolio_wealth,
    wallet_pair_from_snapshot,
)


def test_make_balance_defaults_total_to_free_plus_locked() -> None:
    balance = make_balance("wallex", "USDT", 1.0, 2.0, 1)
    assert balance.total == 3.0


def test_order_collateral_from_open_orders() -> None:
    open_orders = {
        "sell-1": OpenOrder(
            exchange_order_id="sell-1",
            actor_id=None,
            order=Order(
                exchange="nobitex",
                symbol="USDT_IRT",
                side="sell",
                price=100.0,
                qty=2.0,
            ),
            remaining_qty=1.5,
            filled_qty=0.5,
            status="open",
            timestamp_ms=1,
        ),
        "buy-1": OpenOrder(
            exchange_order_id="buy-1",
            actor_id=None,
            order=Order(
                exchange="nobitex",
                symbol="USDT_IRT",
                side="buy",
                price=100.0,
                qty=1.0,
            ),
            remaining_qty=1.0,
            filled_qty=0.0,
            status="open",
            timestamp_ms=1,
        ),
    }
    base_locked, quote_locked = order_collateral_from_open_orders(
        open_orders,
        trading_symbol="USDT_IRT",
        base_asset="USDT",
        quote_asset="IRT",
    )
    assert base_locked == 1.5
    assert quote_locked == 100.0


def test_normalize_snapshot_balances_for_open_orders_lifts_locked_without_inflating_total() -> None:
    snapshot = PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=1,
        balances={
            "USDT": make_balance("nobitex", "USDT", 23.0, 0.0, 1, total=23.0),
            "IRT": make_balance("nobitex", "IRT", 10.0, 0.0, 1),
        },
        open_orders={
            "sell-1": OpenOrder(
                exchange_order_id="sell-1",
                actor_id=None,
                order=Order(
                    exchange="nobitex",
                    symbol="USDT_IRT",
                    side="sell",
                    price=100.0,
                    qty=3.0,
                ),
                remaining_qty=3.0,
                filled_qty=0.0,
                status="open",
                timestamp_ms=1,
            ),
        },
    )
    normalized = normalize_snapshot_balances_for_open_orders(
        snapshot,
        base_asset="USDT",
        quote_asset="IRT",
        trading_symbol="USDT_IRT",
    )

    assert normalized.balances["USDT"].free == 20.0
    assert normalized.balances["USDT"].locked == 3.0
    assert normalized.balances["USDT"].total == 23.0

    inventory, cash = inventory_cash_from_snapshot(
        normalized,
        "USDT",
        "IRT",
        trading_symbol="USDT_IRT",
    )
    assert inventory == 23.0
    assert cash == 10.0


def test_wallet_pair_from_snapshot_exposes_breakdown() -> None:
    snapshot = PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=1,
        balances={"IRT": make_balance("nobitex", "IRT", 4.0, 1.0, 1, total=5.0)},
    )
    base, quote = wallet_pair_from_snapshot(snapshot, "USDT", "IRT", trading_symbol="USDT_IRT")
    assert quote.free == 4.0
    assert quote.locked == 1.0
    assert quote.total == 5.0


def test_portfolio_wealth_enforces_floor() -> None:
    assert portfolio_wealth(0.0, 0.0, 100.0) == 1e-8
