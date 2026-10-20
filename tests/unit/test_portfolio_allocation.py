from __future__ import annotations

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.models.execution import Fill
from mm.core.risk_limits import VenueRiskLimits
from mm.core.models.target_book import TargetOrder, VenueTargetBook
from mm.portfolio.virtual_ledger_service import ActorWalletTracker, VirtualLedgerService
from mm.portfolio.limits import merge_allocation_into_venue_limits
from mm.portfolio.wallet import StrategyWallet, joint_ledger_with_virtual_local, venue_ledger_from_strategy_wallet
from mm.registry.memory import MemoryRegistryStore
from mm.registry.models import AllocationRecord
from mm.risk.net_delta import evaluate_venue_target_transition
from mm.core.models.strategy import StrategyInputFrame
from mm.runtime.config import ControllerRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig
from mm.runtime.control import RuntimeControlPlane
from mm.worker.worker import build_worker_components
from mm.worker.worker_config import worker_config_from_mapping
from tests.unit.test_multi_strategy_worker import _seed_worker_registry


def test_merge_allocation_into_venue_limits_is_noop_for_virtual_wallets() -> None:
    record = AllocationRecord(
        allocation_id="a1",
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        allocated_quote=1_000_000.0,
        allocated_base=2.5,
    )
    limits = VenueRiskLimits(max_inventory=10_000.0, max_open_notional=5_000_000.0)
    merged = merge_allocation_into_venue_limits(limits, record)
    assert merged.max_inventory == 10_000.0
    assert merged.max_open_notional == 5_000_000.0


def test_strategy_wallet_tracks_fills_and_locks() -> None:
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
    tracker = ActorWalletTracker(record=record, binding=config.binding)
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
    assert tracker.wallet is not None
    assert tracker.wallet.held_base == 3.0
    assert tracker.wallet.held_quote == pytest.approx(999_899.9)


def test_evaluate_venue_target_transition_rejects_insufficient_wallet_quote() -> None:
    wallet = StrategyWallet(
        actor_id="btc-maker-actor",
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
        initial_base=0.0,
        initial_quote=50.0,
        held_base=0.0,
        held_quote=50.0,
    )
    ledger = venue_ledger_from_strategy_wallet(wallet)
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
        allocation=None,
    )
    assert not ok
    assert "insufficient_quote_for_bid_orders" in reasons


@pytest.mark.asyncio
async def test_portfolio_allocation_service_halts_actor_on_wallet_breach(monkeypatch: pytest.MonkeyPatch) -> None:
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
    tracker = ActorWalletTracker(record=record, binding=config.binding)
    assert tracker.wallet is not None
    tracker.wallet.held_quote = -1.0
    service = VirtualLedgerService(
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
async def test_worker_physical_ledger_skips_virtual_wallet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mm.worker.worker.load_mm_env", lambda: False)
    store = MemoryRegistryStore()
    assignment_id, _ = _seed_worker_registry(store)
    wallet_id = store.upsert_allocation(
        account_id="nobitex-main",
        allocated_quote=1_000_000.0,
        allocated_base=2.0,
    ).allocation_id
    store.upsert_strategy(
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        symbol="BTC_IRT",
        preset_id="test_preset",
        quoting_exchange="nobitex",
        liquidity_exchange="nobitex",
        status="active",
        wallet_id=wallet_id,
        ledger_source="physical",
    )
    worker = await build_worker_components(
        worker_config_from_mapping({"worker_id": "worker-1", "assignment_id": assignment_id}),
        store=store,
        bus=InMemoryBus(),
    )
    supervisor = next(item for item in worker.supervisors if item.slot.strategy_id == "btc-maker")
    btc_slot = supervisor._rebuild_slot()
    try:
        assert btc_slot.strategy_runner is not None
        assert btc_slot.strategy_runner.strategy_wallet_provider is None
    finally:
        await btc_slot.stop()


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
    assert worker.virtual_ledger_service is not None
    assert len(worker.virtual_ledger_service._trackers) == 2
    wallet = worker.virtual_ledger_service.wallet_for("btc-maker-actor")
    assert wallet is not None
    assert wallet.initial_base == 2.0
    assert wallet.held_base == 2.0


def test_joint_ledger_with_virtual_local_uses_wallet_not_physical_balances() -> None:
    from mm.core.models.ledger import JointLedgerSnapshot, VenueLedger
    from mm.core.wallet import make_balance

    physical = VenueLedger(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
        available_base=100.0,
        available_quote=50_000_000.0,
        locked_base=0.0,
        locked_quote=0.0,
        balances={
            "BTC": make_balance("nobitex", "BTC", 100.0, 0.0, 1),
            "IRT": make_balance("nobitex", "IRT", 50_000_000.0, 0.0, 1),
        },
    )
    joint = JointLedgerSnapshot(local=physical)
    wallet = StrategyWallet(
        actor_id="btc-maker-actor",
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
        initial_base=1.0,
        initial_quote=1_000_000.0,
        held_base=1.0,
        held_quote=1_000_000.0,
    )
    virtual_joint = joint_ledger_with_virtual_local(joint, wallet)
    assert virtual_joint.local.available_base == pytest.approx(1.0)
    assert virtual_joint.local.available_quote == pytest.approx(1_000_000.0)
    assert virtual_joint.consolidated_base_inventory == pytest.approx(1.0)
    assert virtual_joint.consolidated_quote_wealth == pytest.approx(1_000_000.0)


def test_mpc_state_from_frame_uses_virtual_ledger_balances() -> None:
    from mm.core.models.ledger import JointLedgerSnapshot, VenueLedger
    from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
    from mm.core.wallet import make_balance
    from mm.strategies.mpc_engine import _state_from_frame

    physical = VenueLedger(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
        available_base=100.0,
        available_quote=50_000_000.0,
        balances={
            "BTC": make_balance("nobitex", "BTC", 100.0, 0.0, 1),
            "IRT": make_balance("nobitex", "IRT", 50_000_000.0, 0.0, 1),
        },
        last_snapshot_ms=1,
    )
    wallet = StrategyWallet(
        actor_id="btc-maker-actor",
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
        initial_base=2.0,
        initial_quote=500_000.0,
        held_base=2.0,
        held_quote=500_000.0,
    )
    ledger = joint_ledger_with_virtual_local(JointLedgerSnapshot(local=physical), wallet)
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        controller=ControllerRuntimeConfig(enabled=True, actor_id="btc-maker-actor"),
    )
    frame = StrategyInputFrame(
        local=MarketEvent(
            exchange="nobitex",
            symbol="BTC_IRT",
            interval_ms=10_000,
            t_start_ms=0,
            t_end_ms=10_000,
            orderbook=RawOrderbook(
                exchange="nobitex",
                symbol="BTC_IRT",
                timestamp_ms=10_000,
                bids=(PriceLevel(100.0, 1.0),),
                asks=(PriceLevel(101.0, 1.0),),
                is_snapshot=True,
            ),
            trades=(),
            buy_volume=0.0,
            sell_volume=0.0,
            vwap=100.5,
            imbalance=0.0,
            microprice=100.5,
        ),
    )
    state = _state_from_frame(
        frame,
        config.binding,
        spread_max=200.0,
        ledger=ledger,
    )
    assert state.inventory == pytest.approx(2.0)
    assert state.cash == pytest.approx(500_000.0)


def test_risk_rejects_on_virtual_ledger_without_allocation_cap_double_count() -> None:
    wallet = StrategyWallet(
        actor_id="btc-maker-actor",
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
        initial_base=0.0,
        initial_quote=50.0,
        held_base=0.0,
        held_quote=50.0,
    )
    ledger = venue_ledger_from_strategy_wallet(wallet)
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
        allocation=None,
    )
    assert not ok
    assert "insufficient_quote_for_bid_orders" in reasons
    assert "allocation_quote_exceeded" not in reasons


def test_pending_place_locks_virtual_quote_before_the_order_is_live() -> None:
    from mm.core.models.execution import Order
    from mm.core.models.private import OpenOrder

    wallet = StrategyWallet(
        actor_id="btc-maker-actor",
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
        initial_base=1.0,
        initial_quote=50.0,
        held_base=1.0,
        held_quote=50.0,
    )
    pending = OpenOrder(
        exchange_order_id="pending-1",
        actor_id="btc-maker-actor",
        order=Order("nobitex", "BTC_IRT", "buy", 100.0, 2.0),
        remaining_qty=2.0,
        filled_qty=0.0,
        status="pending",
        timestamp_ms=1,
    )

    class _Manager:
        live_orders: dict = {}

        def pending_place_orders(self):
            return {"pending-1": pending}

    wallet.refresh_open_order_locks({"nobitex": _Manager()})
    assert wallet.locked_quote == pytest.approx(200.0)
    assert "wallet_quote_locked_exceeds_held" in wallet.breach_reasons()


@pytest.mark.asyncio
async def test_wallet_held_restores_when_grant_matches_and_resets_when_it_changes() -> None:
    from mm.portfolio.wallet_store import MemoryWalletStore

    store = MemoryWalletStore()
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        controller=ControllerRuntimeConfig(enabled=True, actor_id="btc-maker-actor"),
    )
    record = AllocationRecord(
        allocation_id="a1",
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        allocated_quote=1_000.0,
        allocated_base=2.0,
    )
    service = VirtualLedgerService(
        bus=InMemoryBus(),
        allocations=(record,),
        bindings={record.actor_id: config.binding},
        order_managers={},
        wallet_store=store,
    )
    wallet = service.wallet_for("btc-maker-actor")
    assert wallet is not None
    wallet.held_base = 4.0
    wallet.held_quote = 250.0
    await service._persist_wallets()

    restored = VirtualLedgerService(
        bus=InMemoryBus(),
        allocations=(record,),
        bindings={record.actor_id: config.binding},
        order_managers={},
        wallet_store=store,
    )
    await restored._restore_wallets()
    kept = restored.wallet_for("btc-maker-actor")
    assert kept is not None
    assert kept.held_base == pytest.approx(4.0)
    assert kept.held_quote == pytest.approx(250.0)

    changed = AllocationRecord(
        allocation_id="a1",
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        allocated_quote=1_000.0,
        allocated_base=9.0,
    )
    reset = VirtualLedgerService(
        bus=InMemoryBus(),
        allocations=(changed,),
        bindings={record.actor_id: config.binding},
        order_managers={},
        wallet_store=store,
    )
    await reset._restore_wallets()
    fresh = reset.wallet_for("btc-maker-actor")
    assert fresh is not None
    assert fresh.held_base == pytest.approx(9.0)


def test_shared_wallets_breach_when_held_exceeds_physical_balances() -> None:
    from mm.core.models.private import PrivateStateSnapshot
    from mm.core.wallet import make_balance

    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        controller=ControllerRuntimeConfig(enabled=True, actor_id="btc-maker-actor"),
    )
    first = AllocationRecord(
        allocation_id="a1",
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        allocated_quote=100.0,
        allocated_base=2.0,
    )
    second = AllocationRecord(
        allocation_id="a2",
        strategy_id="btc-maker-2",
        actor_id="btc-maker-actor-2",
        account_id="nobitex-main",
        allocated_quote=100.0,
        allocated_base=2.0,
    )
    service = VirtualLedgerService(
        bus=InMemoryBus(),
        allocations=(first, second),
        bindings={
            first.actor_id: config.binding,
            second.actor_id: config.binding,
        },
        order_managers={},
    )
    service._remember_physical_totals(
        "nobitex",
        PrivateStateSnapshot(
            exchange="nobitex",
            timestamp_ms=1,
            balances={
                "BTC": make_balance("nobitex", "BTC", 1.0, 0.0, 1, total=1.0),
                "IRT": make_balance("nobitex", "IRT", 1_000.0, 0.0, 1, total=1_000.0),
            },
        ),
    )
    service._apply_physical_coverage()
    wallet = service.wallet_for("btc-maker-actor")
    assert wallet is not None
    assert "wallet_exceeds_physical" in wallet.breach_reasons()


def test_shared_global_account_does_not_expose_the_full_physical_balance() -> None:
    from mm.core.models.instrument import InstrumentSpec, SymbolBinding
    from mm.core.models.ledger import JointLedgerSnapshot, VenueLedger
    from mm.core.wallet import make_balance

    local = InstrumentSpec(exchange="nobitex", symbol="BTC_IRT", base_asset="BTC", quote_asset="IRT")
    shared_global = InstrumentSpec(exchange="bitwana", symbol="USDT_IRT", base_asset="USDT", quote_asset="IRT")
    binding = SymbolBinding(local=local, global_=shared_global)
    first = AllocationRecord(
        allocation_id="a1",
        strategy_id="one",
        actor_id="actor-1",
        account_id="nobitex-main",
        allocated_quote=100.0,
        allocated_base=1.0,
    )
    second = AllocationRecord(
        allocation_id="a2",
        strategy_id="two",
        actor_id="actor-2",
        account_id="nobitex-main",
        allocated_quote=100.0,
        allocated_base=1.0,
    )
    service = VirtualLedgerService(
        bus=InMemoryBus(),
        allocations=(first, second),
        bindings={first.actor_id: binding, second.actor_id: binding},
        order_managers={},
    )
    wallet = service.wallet_for("actor-1")
    assert wallet is not None
    assert wallet.isolate_global_balances
    physical_global = VenueLedger(
        exchange="bitwana",
        symbol="USDT_IRT",
        base_asset="USDT",
        quote_asset="IRT",
        available_base=50.0,
        available_quote=9_000_000.0,
        balances={
            "USDT": make_balance("bitwana", "USDT", 50.0, 0.0, 1),
            "IRT": make_balance("bitwana", "IRT", 9_000_000.0, 0.0, 1),
        },
    )
    local_ledger = VenueLedger(
        exchange="nobitex",
        symbol="BTC_IRT",
        base_asset="BTC",
        quote_asset="IRT",
    )
    virtual = joint_ledger_with_virtual_local(
        JointLedgerSnapshot(local=local_ledger, global_=physical_global),
        wallet,
    )
    assert virtual.global_ is not None
    assert virtual.global_.available_quote == 0.0
    assert virtual.global_.available_base == 0.0
