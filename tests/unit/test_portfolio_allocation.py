from __future__ import annotations

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.models.execution import Fill
from mm.core.risk_limits import VenueRiskLimits
from mm.portfolio.allocation_service import ActorAllocationTracker, PortfolioAllocationService
from mm.portfolio.limits import AllocationCaps, merge_allocation_into_venue_limits
from mm.registry.memory import MemoryRegistryStore
from mm.registry.models import AllocationRecord
from mm.risk.net_delta import evaluate_venue_target_transition
from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.config import ControllerRuntimeConfig, MarketMakerRuntimeConfig
from mm.runtime.control import RuntimeControlPlane
from mm.worker.worker import LoadedStrategy, build_worker_components, load_worker_allocations
from mm.worker.worker_config import worker_config_from_mapping
from tests.unit.test_multi_strategy_worker import _seed_worker_registry


def test_merge_allocation_into_venue_limits_caps_inventory_and_notional() -> None:
    record = AllocationRecord(
        allocation_id="a1",
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        allocated_quote=1_000_000.0,
        allocated_base=2.5,
    )
    merged = merge_allocation_into_venue_limits(
        VenueRiskLimits(max_inventory=10_000.0, max_open_notional=5_000_000.0),
        record,
    )
    assert merged.max_inventory == 2.5
    assert merged.max_open_notional == 1_000_000.0


def test_actor_allocation_tracker_attributes_fills() -> None:
    from mm.runtime.config import MarketDataSourceConfig

    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        controller=ControllerRuntimeConfig(enabled=True, actor_id="btc-maker-actor"),
    )
    record = AllocationRecord(
        allocation_id="a1",
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        allocated_quote=1_000_000.0,
        allocated_base=2.0,
    )
    tracker = ActorAllocationTracker(record=record, binding=config.binding)
    tracker.apply_fill(
        Fill(
            exchange_order_id="1",
            actor_id="btc-maker-actor",
            exchange="nobitex",
            symbol="BTC_IRT",
            side="buy",
            price=100.0,
            qty=1.0,
            fee=0.1,
            timestamp_ms=1,
        )
    )
    assert tracker.usage.attributed_base == 1.0
    assert tracker.usage.attributed_quote == pytest.approx(-100.1)


def test_evaluate_venue_target_transition_rejects_allocation_quote_breach() -> None:
    from mm.core.models.ledger import VenueLedger
    from mm.core.models.target_book import TargetOrder, VenueTargetBook

    ledger = VenueLedger(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
        available_base=10.0,
        available_quote=1_000_000.0,
        locked_base=0.0,
        locked_quote=0.0,
        balances={},
        live_orders={},
        last_snapshot_ms=1,
        last_private_event_ms=1,
    )
    target = VenueTargetBook(
        exchange="nobitex",
        symbol="BTC_IRT",
        actor_id="btc-maker-actor",
        orders=(
            TargetOrder(exchange="nobitex", symbol="BTC_IRT", side="buy", price=100.0, qty=1.0),
        ),
        policy="merge",
    )
    ok, reasons, _ = evaluate_venue_target_transition(
        ledger=ledger,
        target=target,
        limits=VenueRiskLimits(),
        allocation=AllocationCaps(
            max_quote_exposure=50.0,
            attributed_quote_used=40.0,
        ),
    )
    assert not ok
    assert "allocation_quote_exceeded" in reasons


@pytest.mark.asyncio
async def test_portfolio_allocation_service_halts_actor_on_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    from mm.runtime.config import MarketDataSourceConfig

    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    bus = InMemoryBus()
    control_plane = RuntimeControlPlane(bus=bus)
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        controller=ControllerRuntimeConfig(enabled=True, actor_id="btc-maker-actor"),
    )
    record = AllocationRecord(
        allocation_id="a1",
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        allocated_quote=100.0,
        allocated_base=1.0,
        halt_on_breach=True,
    )
    tracker = ActorAllocationTracker(record=record, binding=config.binding)
    tracker.usage.locked_quote = 150.0
    service = PortfolioAllocationService(
        bus=bus,
        allocations=(record,),
        bindings={record.actor_id: config.binding},
        order_managers={},
        control_plane=control_plane,
    )
    service._trackers[record.actor_id] = tracker
    await service._enforce_breach(tracker)
    assert "btc-maker-actor" in control_plane.state.allocation_breached_actors
    assert "btc-maker-actor" in control_plane.state.paused_actors


def test_registry_allocation_crud() -> None:
    store = MemoryRegistryStore()
    _seed_worker_registry(store)
    record = store.upsert_allocation(
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        allocated_quote=5_000_000.0,
        allocated_base=10.0,
    )
    assert store.get_allocation_for_strategy("btc-maker").allocation_id == record.allocation_id
    assert store.get_allocation_for_actor("btc-maker-actor").allocated_base == 10.0


@pytest.mark.asyncio
async def test_worker_loads_allocations_from_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    store = MemoryRegistryStore()
    assignment_id, _ = _seed_worker_registry(store)
    store.upsert_allocation(
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        allocated_quote=1_000_000.0,
        allocated_base=2.0,
    )
    store.upsert_allocation(
        strategy_id="usdt-maker",
        actor_id="usdt-maker-actor",
        account_id="nobitex-main",
        allocated_quote=500_000.0,
        allocated_base=5.0,
    )
    worker = await build_worker_components(
        worker_config_from_mapping({"worker_id": "worker-1", "assignment_id": assignment_id}),
        store=store,
        bus=InMemoryBus(),
    )
    assert worker.allocation_service is not None
    assert len(worker.allocation_service._trackers) == 2
