"""Phase 6 tests: execution planner, rules gateway, and stateless router."""

from __future__ import annotations

import asyncio

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.interfaces.order_validation import AbstractVenueOrderLimitSource
from mm.core.models.execution import Order
from mm.core.models.ledger import VenueLedger
from mm.core.models.private import OpenOrder
from mm.core.models.target_book import TargetOrder, VenueTargetBook
from mm.core.models.venue_order_limits import VenueOrderLimits
from mm.execution.planner import AtomicReplacePlanner, ToleranceReconcilePlanner
from mm.execution.rules_gateway import RULES_MIN_NOTIONAL, RULES_MIN_QTY, ExchangeRulesGateway
from mm.execution.router import OrderRouter
from mm.execution.reconcile import ReconcileConfig
from mm.execution.target_book import CancelCommand, ExecutionPlan, PlaceCommand
from tests.unit.test_execution_manager import FakeOrderCommandAdapter


def _open_order(
    order_id: str,
    order: Order,
    *,
    actor_id: str = "actor-1",
) -> OpenOrder:
    return OpenOrder(
        exchange_order_id=order_id,
        actor_id=actor_id,
        order=order,
        remaining_qty=order.qty,
        filled_qty=0.0,
        status="open",
        timestamp_ms=1000,
    )


def _ledger(*live: OpenOrder) -> VenueLedger:
    return VenueLedger(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
        live_orders={order.exchange_order_id: order for order in live},
    )


def _target(*orders: TargetOrder) -> VenueTargetBook:
    return VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=orders,
        policy="reconcile",
    )


def test_tolerance_reconcile_planner_keeps_match_and_places_gap() -> None:
    live = _open_order("live-1", Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0))
    target = _target(
        TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 1.0),
        TargetOrder("nobitex", "BTC_IRT", "sell", 101.0, 2.0),
    )
    plan = ToleranceReconcilePlanner().plan(target, _ledger(live))

    assert len(plan.keep) == 1
    assert plan.keep[0].exchange_order_id == "live-1"
    assert plan.cancel_ids == ()
    assert len(plan.place) == 1
    assert plan.place[0].order.side == "sell"


def test_tolerance_reconcile_planner_cancels_orphans() -> None:
    live_keep = _open_order("live-1", Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0))
    live_orphan = _open_order("live-2", Order("nobitex", "BTC_IRT", "sell", 105.0, 0.5))
    target = _target(TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 1.0))
    plan = ToleranceReconcilePlanner().plan(target, _ledger(live_keep, live_orphan))

    assert plan.cancel_ids == ("live-2",)
    assert plan.place == ()


def test_tolerance_reconcile_planner_respects_tolerance() -> None:
    live = _open_order("live-1", Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0))
    target = _target(TargetOrder("nobitex", "BTC_IRT", "buy", 100.05, 1.005))
    strict = ToleranceReconcilePlanner().plan(
        target,
        _ledger(live),
        config=ReconcileConfig(price_tol_bps=0.0, qty_tol_rel=0.0),
    )
    loose = ToleranceReconcilePlanner().plan(
        target,
        _ledger(live),
        config=ReconcileConfig(price_tol_bps=10.0, qty_tol_rel=0.01),
    )

    assert strict.cancel_ids == ("live-1",)
    assert len(strict.place) == 1
    assert loose.cancel_ids == ()
    assert loose.place == ()


def test_atomic_replace_planner_cancels_all_before_place() -> None:
    live = _open_order("live-1", Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0))
    target = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=(TargetOrder("nobitex", "BTC_IRT", "sell", 101.0, 1.0),),
        policy="replace",
    )
    plan = AtomicReplacePlanner().plan(target, _ledger(live))

    assert plan.policy == "replace"
    assert plan.keep == ()
    assert plan.cancel_ids == ("live-1",)
    assert len(plan.place) == 1


def test_atomic_replace_planner_cancels_unattributed_same_symbol_orders() -> None:
    live = _open_order(
        "live-1",
        Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0),
        actor_id=None,
    )
    other_actor = _open_order(
        "other-actor",
        Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0),
        actor_id="other",
    )
    other_symbol = _open_order(
        "other-symbol",
        Order("nobitex", "ETH_IRT", "buy", 100.0, 1.0),
        actor_id=None,
    )
    target = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        orders=(),
        policy="replace",
    )
    plan = AtomicReplacePlanner().plan(target, _ledger(live, other_actor, other_symbol))

    assert plan.cancel_ids == ("live-1",)
    assert plan.place == ()


class StaticVenueOrderLimitSource(AbstractVenueOrderLimitSource):
    def __init__(self, limits: VenueOrderLimits) -> None:
        self._limits = limits

    def limits_for(self, symbol: str) -> VenueOrderLimits:
        return self._limits


def test_exchange_rules_gateway_rejects_with_stable_labels() -> None:
    gateway = ExchangeRulesGateway(
        StaticVenueOrderLimitSource(VenueOrderLimits(min_base_qty=0.5, min_notional_quote=500.0))
    )
    plan = ExecutionPlan(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        policy="reconcile",
        place=(
            PlaceCommand("actor-1", Order("nobitex", "BTC_IRT", "buy", 10.0, 0.1)),
            PlaceCommand("actor-1", Order("nobitex", "BTC_IRT", "buy", 10.0, 1.0)),
        ),
    )

    validated = gateway.validate(plan)

    assert validated.place == ()
    reasons = {rejection.reason for rejection in validated.rejected}
    assert RULES_MIN_QTY in reasons
    assert RULES_MIN_NOTIONAL in reasons


def test_exchange_rules_gateway_rounds_to_venue_steps() -> None:
    gateway = ExchangeRulesGateway(
        StaticVenueOrderLimitSource(
            VenueOrderLimits(
                min_base_qty=0.0,
                min_notional_quote=0.0,
                price_step=0.5,
                qty_step=0.01,
            )
        )
    )
    plan = ExecutionPlan(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        policy="reconcile",
        place=(PlaceCommand("actor-1", Order("nobitex", "BTC_IRT", "buy", 10.74, 1.239)),),
    )

    validated = gateway.validate(plan)

    assert validated.rejected == ()
    assert len(validated.place) == 1
    assert validated.place[0].order.price == 10.5
    assert validated.place[0].order.qty == 1.23


@pytest.mark.asyncio
async def test_order_router_cancel_before_place_for_replace_policy() -> None:
    adapter = FakeOrderCommandAdapter()
    bus = InMemoryBus()
    events = bus.subscribe_queue("execution:orders:nobitex")
    router = OrderRouter("nobitex", adapter, bus=bus, clock_ms=lambda: 1000)
    plan = ExecutionPlan(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        policy="replace",
        cancel=(CancelCommand("live-1", "actor-1", "replace"),),
        place=(PlaceCommand("actor-1", Order("nobitex", "BTC_IRT", "sell", 101.0, 1.0)),),
    )

    result = await router.execute(plan)

    assert adapter.cancelled == ["live-1"]
    assert len(adapter.placed) == 1
    assert result.cancelled_ids == ("live-1",)
    assert len(result.placed) == 1
    assert result.cancel_ms >= 0.0
    assert result.place_ms >= 0.0

    event_types = []
    while not events.empty():
        event_types.append((await events.get())["type"])
    assert event_types.index("CancelOrdersRequested") < event_types.index("PlaceOrdersRequested")
    assert "PlaceOrdersAccepted" in event_types


@pytest.mark.asyncio
async def test_order_router_aborts_placement_on_cancel_failure_for_reconcile_policy() -> None:
    class FailingCancelAdapter(FakeOrderCommandAdapter):
        async def cancel_orders(self, order_ids: list[str]) -> None:
            raise RuntimeError("Nobitex cancel failed: invalid order id")

    adapter = FailingCancelAdapter()
    bus = InMemoryBus()
    router = OrderRouter("nobitex", adapter, bus=bus, clock_ms=lambda: 1000)
    plan = ExecutionPlan(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        policy="reconcile",
        cancel=(CancelCommand("live-1", "actor-1", "orphan"),),
        place=(PlaceCommand("actor-1", Order("nobitex", "BTC_IRT", "sell", 101.0, 1.0)),),
    )

    result = await router.execute(plan)

    # Cancel failed: MUST abort placements so orders do not accumulate on the exchange!
    assert result.failed_cancel_ids == ("live-1",)
    assert result.placed == ()
    assert adapter.placed == []


@pytest.mark.asyncio
async def test_order_router_does_not_mutate_external_live_order_state() -> None:
    adapter = FakeOrderCommandAdapter()
    router = OrderRouter("nobitex", adapter, clock_ms=lambda: 1000)
    external_live: dict[str, OpenOrder] = {
        "live-1": _open_order("live-1", Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0)),
    }
    before = dict(external_live)

    plan = ExecutionPlan(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="actor-1",
        policy="replace",
        cancel=(CancelCommand("live-1", "actor-1", "replace"),),
        place=(PlaceCommand("actor-1", Order("nobitex", "BTC_IRT", "sell", 101.0, 1.0)),),
    )
    await router.execute(plan)

    assert external_live == before


@pytest.mark.asyncio
async def test_planner_output_deterministic_from_target_and_ledger() -> None:
    live = _open_order("live-1", Order("nobitex", "BTC_IRT", "buy", 100.0, 1.0))
    target = _target(
        TargetOrder("nobitex", "BTC_IRT", "buy", 100.0, 1.0),
        TargetOrder("nobitex", "BTC_IRT", "sell", 101.0, 2.0),
    )
    ledger = _ledger(live)
    planner = ToleranceReconcilePlanner()
    config = ReconcileConfig(price_tol_bps=10.0, qty_tol_rel=0.01)

    first = planner.plan(target, ledger, config=config)
    second = planner.plan(target, ledger, config=config)

    assert first == second
