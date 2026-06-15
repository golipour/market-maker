from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mm.api.runtime_state import (
    RuntimeControlSnapshot,
    build_runtime_overview,
    strategy_trading_status,
    worker_live_status,
)
from mm.registry.memory import MemoryRegistryStore
from mm.registry.models import WorkerAssignmentRecord


def _now() -> datetime:
    return datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_worker_live_status_alive_and_stale() -> None:
    now = _now()
    alive = WorkerAssignmentRecord(
        assignment_id="wa-1",
        worker_id="worker-1",
        strategy_ids=("btc-maker",),
        lease_expires_at=now + timedelta(seconds=120),
        heartbeat_at=now,
        status="active",
        created_at=now,
        updated_at=now,
    )
    live, remaining = worker_live_status(alive, now_ms=int(now.timestamp() * 1000))
    assert live == "alive"
    assert remaining == 120

    stale = WorkerAssignmentRecord(
        assignment_id="wa-2",
        worker_id="worker-2",
        strategy_ids=(),
        lease_expires_at=now - timedelta(seconds=10),
        heartbeat_at=now - timedelta(seconds=30),
        status="active",
        created_at=now,
        updated_at=now,
    )
    live, remaining = worker_live_status(stale, now_ms=int(now.timestamp() * 1000))
    assert live == "stale"
    assert remaining == 0


def test_strategy_trading_status_priority() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(
        exchange="nobitex",
        config={"market_data": {"mode": "rest"}, "private": {"mode": "rest"}},
    )
    store.upsert_account(
        account_id="acct",
        exchange="nobitex",
        credential_ref="TOKEN",
    )
    store.upsert_preset(
        preset_id="preset",
        config={"interval_ms": 5000, "liquidity_sourcing": {"mode": "off"}},
    )
    strategy = store.upsert_strategy(
        strategy_id="btc-maker",
        actor_id="btc-actor",
        account_id="acct",
        symbol="BTC_IRT",
        preset_id="preset",
        quoting_exchange="nobitex",
        liquidity_exchange="nobitex",
        status="active",
    )
    control = RuntimeControlSnapshot(
        paused_actors=("btc-actor",),
        allocation_breached_actors=("btc-actor",),
        kill_switch=True,
    )
    assert strategy_trading_status(strategy, control=control, worker_live="alive") == "kill_switch"

    control = RuntimeControlSnapshot(dry_run=True)
    assert strategy_trading_status(strategy, control=control, worker_live="alive") == "dry_run"

    control = RuntimeControlSnapshot()
    assert strategy_trading_status(strategy, control=control, worker_live=None) == "unassigned"
    assert strategy_trading_status(strategy, control=control, worker_live="offline") == "worker_offline"
    assert strategy_trading_status(strategy, control=control, worker_live="alive") == "running"

    paused = RuntimeControlSnapshot(paused_actors=("btc-actor",))
    assert strategy_trading_status(strategy, control=paused, worker_live="alive") == "paused"


def test_build_runtime_overview_maps_worker_to_strategy() -> None:
    store = MemoryRegistryStore()
    store.upsert_exchange(
        exchange="nobitex",
        config={"market_data": {"mode": "rest"}, "private": {"mode": "rest"}},
    )
    store.upsert_account(
        account_id="acct",
        exchange="nobitex",
        credential_ref="TOKEN",
    )
    store.upsert_preset(
        preset_id="preset",
        config={"interval_ms": 5000, "liquidity_sourcing": {"mode": "off"}},
    )
    store.upsert_strategy(
        strategy_id="btc-maker",
        actor_id="btc-actor",
        account_id="acct",
        symbol="BTC_IRT",
        preset_id="preset",
        quoting_exchange="nobitex",
        liquidity_exchange="nobitex",
        status="active",
    )
    store.upsert_worker_assignment(
        assignment_id="wa-1",
        worker_id="worker-1",
        strategy_ids=["btc-maker"],
        lease_ttl_seconds=300,
        status="active",
    )

    from mm.api.runtime_state import RuntimeControlCache
    from mm.bus.memory import InMemoryBus

    overview = build_runtime_overview(store, RuntimeControlCache(bus=InMemoryBus()))
    assert len(overview["workers"]) == 1
    assert overview["workers"][0]["live"] == "alive"
    assert overview["strategies"][0]["strategy_id"] == "btc-maker"
    assert overview["strategies"][0]["worker_id"] == "worker-1"
    assert overview["strategies"][0]["trading_status"] == "running"
