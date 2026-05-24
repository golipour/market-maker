"""Phase 0 regression tests: document current OrderManager reconcile counts and rejection labels."""

from __future__ import annotations

import pytest

from mm.core.interfaces.order_validation import AbstractVenueOrderLimitSource
from mm.core.models.execution import Action, Order
from mm.core.models.private import OpenOrder
from mm.core.models.venue_order_limits import VenueOrderLimits
from mm.execution.order_manager import OrderManager
from mm.execution.reconcile import ReconcileConfig
from tests.unit.test_execution_manager import FakeOrderCommandAdapter


class StaticVenueOrderLimitSource(AbstractVenueOrderLimitSource):
    def __init__(self, limits: VenueOrderLimits) -> None:
        self._limits = limits

    def limits_for(self, symbol: str) -> VenueOrderLimits:
        return self._limits


def _open_order(
    order_id: str,
    order: Order,
    *,
    actor_id: str = "actor-1",
    remaining_qty: float | None = None,
) -> OpenOrder:
    qty = remaining_qty if remaining_qty is not None else order.qty
    return OpenOrder(
        exchange_order_id=order_id,
        actor_id=actor_id,
        order=order,
        remaining_qty=qty,
        filled_qty=0.0,
        status="open",
        timestamp_ms=1000,
    )


@pytest.mark.asyncio
async def test_order_manager_sync_report_reconcile_counts() -> None:
    manager = OrderManager(
        "nobitex",
        FakeOrderCommandAdapter(),
        clock_ms=lambda: 1000,
        reconcile_config=ReconcileConfig(price_tol_bps=0.0, qty_tol_rel=0.0),
    )
    manager._live_orders = {
        "live-keep": _open_order(
            "live-keep",
            Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0),
        ),
        "live-cancel": _open_order(
            "live-cancel",
            Order("nobitex", "BTC_IRT", "sell", 105.0, 0.5),
        ),
    }

    report = await manager.sync_with_report(
        [
            Action(
                actor_id="actor-1",
                orders=(
                    Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0, client_order_id="target-buy"),
                    Order("nobitex", "BTC_IRT", "sell", 101.0, 2.0, client_order_id="target-sell"),
                ),
            )
        ]
    )

    assert report.target.order_count == 2
    assert report.kept.order_count == 1
    assert report.kept.buy_qty == 1.0
    assert report.canceled.order_count == 1
    assert report.canceled.sell_qty == 0.5
    assert report.to_place.order_count == 1
    assert report.to_place.sell_qty == 2.0
    assert report.cancel_count == 1
    assert report.placed is True
    assert report.accepted.order_count == 1


@pytest.mark.asyncio
async def test_order_manager_preview_placement_matches_execute_plan_counts() -> None:
    manager = OrderManager(
        "nobitex",
        FakeOrderCommandAdapter(),
        clock_ms=lambda: 1000,
        reconcile_config=ReconcileConfig(price_tol_bps=0.0, qty_tol_rel=0.0),
    )
    manager._live_orders = {
        "live-1": _open_order(
            "live-1",
            Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0),
        ),
    }
    actions = [
        Action(
            actor_id="actor-1",
            orders=(
                Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0),
                Order("nobitex", "BTC_IRT", "sell", 101.0, 1.0),
            ),
        )
    ]

    preview = manager.preview_placement(actions)
    executed = await manager.sync_with_report(actions)

    assert preview.target.order_count == executed.target.order_count
    assert preview.kept.order_count == executed.kept.order_count
    assert preview.canceled.order_count == executed.canceled.order_count
    assert preview.to_place.order_count == executed.to_place.order_count
    assert preview.cancel_count == executed.cancel_count
    assert preview.placed is False


@pytest.mark.asyncio
async def test_order_manager_rejection_label_below_min_base_qty() -> None:
    limits = VenueOrderLimits(min_base_qty=0.5, min_notional_quote=0.0)
    commands = FakeOrderCommandAdapter()
    manager = OrderManager(
        "nobitex",
        commands,
        clock_ms=lambda: 1000,
        limit_source=StaticVenueOrderLimitSource(limits),
    )

    report = await manager.sync_with_report(
        [
            Action(
                actor_id="actor-1",
                orders=(Order("nobitex", "BTC_IRT", "buy", 100.0, 0.1),),
            )
        ]
    )

    assert report.venue_rejection_counts == {"below_min_base_qty": 1}
    assert report.to_place.order_count == 1
    assert report.limit_valid.order_count == 0
    assert report.accepted.order_count == 0
    assert commands.placed == []


@pytest.mark.asyncio
async def test_order_manager_rejection_label_below_min_notional_quote() -> None:
    limits = VenueOrderLimits(min_base_qty=0.0, min_notional_quote=500.0)
    commands = FakeOrderCommandAdapter()
    manager = OrderManager(
        "nobitex",
        commands,
        clock_ms=lambda: 1000,
        limit_source=StaticVenueOrderLimitSource(limits),
    )

    report = await manager.sync_with_report(
        [
            Action(
                actor_id="actor-1",
                orders=(Order("nobitex", "BTC_IRT", "buy", 10.0, 1.0),),
            )
        ]
    )

    assert report.venue_rejection_counts == {"below_min_notional_quote": 1}
    assert report.limit_valid.order_count == 0
    assert report.accepted.order_count == 0
    assert commands.placed == []


@pytest.mark.asyncio
async def test_order_manager_empty_target_cancels_all_scoped_orders() -> None:
    commands = FakeOrderCommandAdapter()
    manager = OrderManager("nobitex", commands, clock_ms=lambda: 1000)
    manager._live_orders = {
        "live-1": _open_order(
            "live-1",
            Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0),
        ),
    }

    report = await manager.sync_with_report([Action(actor_id="actor-1", orders=())])

    assert report.target.order_count == 0
    assert report.kept.order_count == 0
    assert report.canceled.order_count == 1
    assert report.to_place.order_count == 0
    assert report.cancel_count == 1
    assert commands.cancelled == ["live-1"]
