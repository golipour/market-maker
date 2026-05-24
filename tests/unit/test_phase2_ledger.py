"""Phase 2 tests: portfolio ledger state machine, replay, service, and consistency."""

from __future__ import annotations

import asyncio

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.models.execution import Fill, Order
from mm.core.models.instrument import InstrumentSpec
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.core.wallet import make_balance
from mm.execution.order_manager import OrderManager
from mm.execution.reconciler import ExecutionStateReconciler
from mm.ledger.models import LedgerConsistencyDiagnostic
from mm.ledger.replay import apply_ledger_event, replay_venue_ledger
from mm.ledger.service import PortfolioLedgerService
from mm.ledger.state_machine import (
    VenueLedgerStateMachine,
    build_joint_ledger_snapshot,
    compare_ledger_to_private_snapshot,
)
from mm.private_state.store import PrivateStateStore
from mm.runtime.config import MarketDataSourceConfig, MarketMakerRuntimeConfig, PrivateRuntimeConfig
from tests.unit.test_execution_manager import FakeOrderCommandAdapter
from tests.unit.test_private_state_reconciler import FakePrivateStateAdapter
from mm.private_state.reconciler import PrivateStateReconciler


def _nobitex_spec() -> InstrumentSpec:
    return InstrumentSpec(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
    )


def _buy_order(order_id: str = "o1", *, qty: float = 0.5, price: float = 100.0) -> OpenOrder:
    return OpenOrder(
        exchange_order_id=order_id,
        actor_id="actor-1",
        order=Order("nobitex", "BTC_IRT", "buy", price, qty),
        remaining_qty=qty,
        filled_qty=0.0,
        status="open",
        timestamp_ms=1100,
    )


def test_venue_ledger_state_machine_matches_private_store_snapshot() -> None:
    spec = _nobitex_spec()
    machine = VenueLedgerStateMachine("nobitex", spec)
    store = PrivateStateStore("nobitex", trading_symbol="BTC_IRT")

    snapshot = PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=1000,
        balances={
            "BTC": make_balance("nobitex", "BTC", 1.0, 0.25, 1000),
            "IRT": make_balance("nobitex", "IRT", 5000.0, 0.0, 1000),
        },
        open_orders={"o1": _buy_order()},
    )
    store.apply_snapshot(snapshot)
    ledger = machine.apply_snapshot(snapshot)

    assert compare_ledger_to_private_snapshot(ledger, store.snapshot())
    assert ledger.balances["BTC"].free == pytest.approx(1.0)
    assert ledger.live_orders["o1"].remaining_qty == pytest.approx(0.5)
    assert ledger.available_base == pytest.approx(1.0)
    assert ledger.locked_quote == pytest.approx(50.0)


def test_replay_rebuilds_balances_open_orders_and_terminal_fill_state() -> None:
    spec = _nobitex_spec()
    events = [
        PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=1000,
            balances={"BTC": make_balance("nobitex", "BTC", 2.0, 0.0, 1000)},
            open_orders={"o1": _buy_order()},
        ),
        Balance("nobitex", "BTC", 1.5, 0.0, 1.5, 1200),
        Fill(
            exchange_order_id="o1",
            actor_id="actor-1",
            exchange="nobitex",
            symbol="BTC_IRT",
            side="buy",
            price=100.0,
            qty=0.5,
            fee=0.0,
            timestamp_ms=1300,
        ),
        OrderUpdate(
            exchange="nobitex",
            symbol="BTC_IRT",
            exchange_order_id="o1",
            actor_id="actor-1",
            status="filled",
            remaining_qty=0.0,
            filled_qty=0.5,
            timestamp_ms=1300,
        ),
    ]

    ledger = replay_venue_ledger(events, exchange="nobitex", spec=spec)

    assert ledger.balances["BTC"].free == pytest.approx(1.5)
    assert "o1" not in ledger.live_orders


def test_apply_ledger_event_sequence_matches_store_replay() -> None:
    spec = _nobitex_spec()
    machine = VenueLedgerStateMachine("nobitex", spec)
    store = PrivateStateStore("nobitex", trading_symbol="BTC_IRT")

    events = [
        PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=1000,
            balances={"BTC": make_balance("nobitex", "BTC", 1.0, 0.0, 1000)},
            open_orders={"o1": _buy_order(), "o2": _buy_order("o2", qty=0.25, price=101.0)},
        ),
        OpenOrder(
            exchange_order_id="o2",
            actor_id="actor-1",
            order=Order("nobitex", "BTC_IRT", "buy", 101.0, 0.25),
            remaining_qty=0.0,
            filled_qty=0.25,
            status="filled",
            timestamp_ms=1400,
        ),
    ]
    ledger = machine.snapshot()
    private = store.snapshot()
    for event in events:
        ledger = apply_ledger_event(machine, event)
        if isinstance(event, PrivateStateSnapshot):
            store.apply_snapshot(event)
        elif isinstance(event, OpenOrder):
            store.apply_open_order(event)
        private = store.snapshot()

    assert compare_ledger_to_private_snapshot(ledger, private)
    assert "o2" not in ledger.live_orders
    assert "o1" in ledger.live_orders


def test_build_joint_ledger_snapshot_consolidates_local_and_global() -> None:
    config = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        hedge=MarketDataSourceConfig(role="global", exchange="wallex", symbol="BTC_IRT"),
    )
    local = VenueLedgerStateMachine("nobitex", config.binding.local).apply_snapshot(
        PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=1000,
            balances={
                "BTC": make_balance("nobitex", "BTC", 1.0, 0.0, 1000),
                "IRT": make_balance("nobitex", "IRT", 1000.0, 0.0, 1000),
            },
        )
    )
    global_ = VenueLedgerStateMachine("wallex", config.binding.global_).apply_snapshot(
        PrivateStateSnapshot(
            exchange="wallex",
            timestamp_ms=1000,
            balances={
                "BTC": make_balance("wallex", "BTC", 0.5, 0.0, 1000),
                "IRT": make_balance("wallex", "IRT", 500.0, 0.0, 1000),
            },
        )
    )

    joint = build_joint_ledger_snapshot(
        local,
        global_,
        config.binding,
        reference_price=100.0,
        created_at_ms=2000,
    )

    assert joint.local.exchange == "nobitex"
    assert joint.global_ is not None
    assert joint.global_.exchange == "wallex"
    assert joint.consolidated_base_inventory == pytest.approx(1.5)
    assert joint.consolidated_quote_wealth == pytest.approx(1500.0 + 1.5 * 100.0)


@pytest.mark.asyncio
async def test_portfolio_ledger_service_publishes_snapshots_from_private_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOBITEX_API_TOKEN", "test-token")
    config = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        local_private=PrivateRuntimeConfig(api_token_env="NOBITEX_API_TOKEN"),
    )
    bus = InMemoryBus()
    venue_queue = bus.subscribe_queue("ledger:snapshot:nobitex")
    joint_queue = bus.subscribe_queue("ledger:snapshot")

    service = PortfolioLedgerService(config, config.binding, bus)
    adapter = FakePrivateStateAdapter()
    reconciler = PrivateStateReconciler("nobitex", adapter, bus)

    try:
        await service.start()
        await reconciler.start()
        await adapter.open_orders.put(_buy_order())

        venue_payload = await asyncio.wait_for(venue_queue.get(), timeout=1.0)
        joint_payload = await asyncio.wait_for(joint_queue.get(), timeout=1.0)

        assert venue_payload["exchange"] == "nobitex"
        assert "o1" in venue_payload["live_orders"]
        assert joint_payload["local"]["exchange"] == "nobitex"
        assert joint_payload["global"] is None
    finally:
        await reconciler.stop()
        await service.stop()


@pytest.mark.asyncio
async def test_portfolio_ledger_service_consistency_diagnostic_detects_divergence() -> None:
    config = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        local_private=PrivateRuntimeConfig(api_token_env="NOBITEX_API_TOKEN"),
    )
    bus = InMemoryBus()
    manager = OrderManager("nobitex", FakeOrderCommandAdapter(), clock_ms=lambda: 1000)
    manager._live_orders = {
        "o1": _buy_order(qty=0.4),
    }

    service = PortfolioLedgerService(
        config,
        config.binding,
        bus,
        order_managers={"nobitex": manager},
        clock_ms=lambda: 1500,
    )
    service._local.apply_snapshot(
        PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=1000,
            balances={"BTC": make_balance("nobitex", "BTC", 1.0, 0.0, 1000)},
            open_orders={"o1": _buy_order(qty=0.5)},
        )
    )

    diagnostic = service.consistency_diagnostic("nobitex")

    assert isinstance(diagnostic, LedgerConsistencyDiagnostic)
    assert diagnostic.matches is False
    assert diagnostic.ledger_order_count == 1
    assert diagnostic.manager_order_count == 1
    assert diagnostic.mismatches[0].reason == "remaining_qty"
    assert diagnostic.mismatches[0].ledger_value == pytest.approx(0.5)
    assert diagnostic.mismatches[0].manager_value == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_portfolio_ledger_service_publishes_consistency_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOBITEX_API_TOKEN", "test-token")
    config = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        local_private=PrivateRuntimeConfig(
            api_token_env="NOBITEX_API_TOKEN",
            enable_execution=True,
        ),
    )
    bus = InMemoryBus()
    manager = OrderManager("nobitex", FakeOrderCommandAdapter(), clock_ms=lambda: 1000)
    consistency_queue = bus.subscribe_queue("ledger:diagnostics:consistency:nobitex")

    service = PortfolioLedgerService(
        config,
        config.binding,
        bus,
        order_managers={"nobitex": manager},
        clock_ms=lambda: 2000,
    )
    adapter = FakePrivateStateAdapter()
    reconciler = PrivateStateReconciler("nobitex", adapter, bus)
    exec_reconciler = ExecutionStateReconciler("nobitex", manager, bus)

    try:
        await service.start()
        await exec_reconciler.start()
        await reconciler.start()
        await adapter.open_orders.put(_buy_order())

        payload = None
        for _ in range(6):
            payload = await asyncio.wait_for(consistency_queue.get(), timeout=1.0)
            if payload["ledger_order_count"] == 1 and payload["matches"] is True:
                break

        assert payload is not None
        assert payload["type"] == "LedgerConsistencyDiagnostic"
        assert payload["exchange"] == "nobitex"
        assert payload["matches"] is True
    finally:
        await reconciler.stop()
        await exec_reconciler.stop()
        await service.stop()
