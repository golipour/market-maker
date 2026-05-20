from __future__ import annotations

import pytest

from mm.core.models.execution import Action, Order, TradingState
from mm.core.models.market import MarketEvent, RawOrderbook
from mm.core.models.private import OpenOrder, PrivateStateSnapshot
from mm.core.order_constraints import (
    OrderConstraints,
    apply_venue_constraints_with_report,
    cap_orders_per_side_with_report,
)
from mm.execution.order_manager import OrderManager
from mm.risk.engine import RiskLimits, RuntimeRiskEngine
from mm.core.risk_limits import VenueRiskLimits
from mm.core.models.strategy import StrategyInputFrame


def _order(side: str, qty: float, price: float = 100.0, *, client_order_id: str | None = None) -> Order:
    return Order("nobitex", "BTC_IRT", side, price, qty, client_order_id=client_order_id)


def test_venue_constraints_report_rejection_reasons() -> None:
    constraints = OrderConstraints(
        price_step=None,
        qty_step=0.01,
        min_qty=0.05,
        min_notional=100.0,
        max_orders_per_side=10,
    )
    orders = [_order("buy", 0.01, 200.0), _order("sell", 0.5, 250.0)]
    kept, report = apply_venue_constraints_with_report(orders, constraints)
    assert len(kept) == 1
    assert kept[0].side == "sell"
    assert report.rejection_counts.get("below_min_qty", 0) == 1


def test_cap_orders_per_side_reports_capped_counts() -> None:
    orders = [_order("buy", 0.1), _order("buy", 0.5), _order("sell", 0.3)]
    kept, report = cap_orders_per_side_with_report(orders, 1)
    assert len(kept) == 2
    assert report.rejection_counts.get("capped_buy") == 1


def test_runtime_risk_validate_with_result_counts_inventory_rejects() -> None:
    risk = RuntimeRiskEngine(
        limits=RiskLimits(primary=VenueRiskLimits(max_inventory=1.0, max_private_state_age_ms=60_000)),
        clock_ms=lambda: 1000,
    )
    action = Action(actor_id="a", orders=(_order("buy", 5.0),))
    event = MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=0,
        t_end_ms=1000,
        orderbook=RawOrderbook("nobitex", "BTC_IRT", 1000, (), (), True),
        trades=(),
        buy_volume=0.0,
        sell_volume=0.0,
        vwap=0.0,
        imbalance=0.0,
        microprice=0.0,
    )
    frame = StrategyInputFrame(
        primary=event,
        primary_private=PrivateStateSnapshot(exchange="nobitex", timestamp_ms=1000, balances={}),
    )
    result = risk.validate_frame_with_result(
        [action], TradingState(actor_id="a", inventory=0.0, cash=1_000_000.0), frame
    )
    assert result.approved == ()
    assert result.rejection_counts.get("max_inventory", 0) >= 1


def test_order_manager_preview_reconcile_keeps_and_places() -> None:
    class _Limits:
        min_base_qty = 0.01
        min_notional_quote = 1.0

    class _LimitSource:
        def limits_for(self, symbol: str) -> _Limits:
            return _Limits()

    class _Adapter:
        async def place_orders(self, orders):
            return [f"id-{i}" for i in range(len(orders))]

        async def cancel_orders(self, order_ids):
            return None

    manager = OrderManager("nobitex", _Adapter(), limit_source=_LimitSource())
    existing = _order("buy", 1.0, 100.0, client_order_id="level-0")
    manager._live_orders = {
        "live-1": OpenOrder(
            exchange_order_id="live-1",
            actor_id="a",
            order=existing,
            remaining_qty=1.0,
            filled_qty=0.0,
            status="open",
            timestamp_ms=1000,
        )
    }
    action = Action(
        actor_id="a",
        orders=(
            _order("buy", 1.0, 100.0, client_order_id="level-0"),
            _order("sell", 2.0, 101.0, client_order_id="level-0"),
        ),
    )
    report = manager.preview_placement([action])
    assert report.kept.order_count >= 1
    assert report.to_place.order_count >= 1
    assert report.limit_valid.order_count >= 1


@pytest.mark.asyncio
async def test_order_manager_sync_cancels_orphan_and_places_new() -> None:
    class _Limits:
        min_base_qty = 0.01
        min_notional_quote = 1.0

    class _LimitSource:
        def limits_for(self, symbol: str) -> _Limits:
            return _Limits()

    class _Adapter:
        def __init__(self) -> None:
            self.placed: list[Order] = []
            self.cancelled: list[str] = []

        async def place_orders(self, orders: list[Order]) -> list[str]:
            self.placed.extend(orders)
            return [f"new-{i}" for i in range(len(orders))]

        async def cancel_orders(self, order_ids: list[str]) -> None:
            self.cancelled.extend(order_ids)

    adapter = _Adapter()
    manager = OrderManager("nobitex", adapter, limit_source=_LimitSource())
    manager._live_orders = {
        "orphan": OpenOrder(
            exchange_order_id="orphan",
            actor_id="a",
            order=_order("sell", 1.0, 105.0),
            remaining_qty=1.0,
            filled_qty=0.0,
            status="open",
            timestamp_ms=1000,
        ),
    }
    action = Action(actor_id="a", orders=(_order("buy", 1.0, 100.0),))
    report = await manager.sync_with_report([action])
    assert "orphan" in adapter.cancelled
    assert len(adapter.placed) == 1
    assert report.canceled.order_count >= 1
    assert report.accepted.order_count == 1
