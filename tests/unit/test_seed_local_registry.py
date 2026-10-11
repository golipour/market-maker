from __future__ import annotations

import pytest

from mm.registry.memory import MemoryRegistryStore, RegistryNotFoundError
from mm.tools.seed_local_registry import (
    LOCAL_STRATEGY_ID,
    LOCAL_WORKER_ID,
    ensure_local_worker_registry,
    seed_local_worker_registry,
)


def test_seed_local_worker_registry_creates_active_assignment() -> None:
    store = MemoryRegistryStore()
    assert seed_local_worker_registry(store) is True

    assignment = store.get_active_worker_assignment(LOCAL_WORKER_ID)
    assert assignment.worker_id == LOCAL_WORKER_ID
    assert assignment.strategy_ids == (LOCAL_STRATEGY_ID,)

    bundle = store.load_runtime_bundle(LOCAL_STRATEGY_ID)
    assert bundle["strategy"].strategy_id == LOCAL_STRATEGY_ID
    assert bundle["exchanges"]["nobitex"].config["market_data"]["market_data_source"] == "bus"
    assert store.get_account("account-bitwana-main").exchange == "bitwana"
    assert store.get_market_source("bitwana-USDT_IRT").symbol == "USDT_IRT"
    assert store.get_platform_settings().bus["redis_url"] == "redis://localhost:6379/0"

    assert bundle["strategy"].strategy_id == LOCAL_STRATEGY_ID
    assert store.get_platform_settings().bus["redis_url"] == "redis://localhost:6379/0"

    preset = store.get_preset("thin_local_liquidity_weighted_mpc")
    optimizer = preset.config["controller"]["optimizer"]
    assert optimizer["kind"] == "monte_carlo_mpc_tensor_batched"
    assert "local" in optimizer
    assert "global" in optimizer
    assert preset.config["execution"]["quoting"]["planner"] == "replace"
    assert preset.config["controller"]["dry_run"] is True


def test_ensure_local_worker_registry_seeds_once(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryRegistryStore()
    monkeypatch.setattr(
        "mm.tools.seed_local_registry.build_registry_store",
        lambda **_: store,
    )
    monkeypatch.setattr(
        "mm.tools.seed_local_registry.registry_database_url",
        lambda: "postgresql://example/test",
    )

    assert ensure_local_worker_registry() is True
    assert ensure_local_worker_registry() is False
    store.get_active_worker_assignment(LOCAL_WORKER_ID)


def test_ensure_local_worker_registry_requires_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mm.tools.seed_local_registry.registry_database_url", lambda: "")
    with pytest.raises(RuntimeError, match="MM_DATABASE_URL"):
        ensure_local_worker_registry()


def test_seed_uses_bus_redis_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MM_BUS_REDIS_URL", "redis://redis:6379/0")
    store = MemoryRegistryStore()
    seed_local_worker_registry(store)
    assert store.get_platform_settings().bus["redis_url"] == "redis://redis:6379/0"


def test_seed_bitwana_venue_preserves_existing_account_credentials() -> None:
    from mm.tools.seed_local_registry import BITWANA_ACCOUNT_ID, BITWANA_EXCHANGE, seed_bitwana_venue

    store = MemoryRegistryStore()
    seed_bitwana_venue(store)

    custom_creds = '{"v":2,"auth_mode":"signed_key","api_key":"test_key","api_secret":"test_secret"}'
    store.upsert_account(
        account_id=BITWANA_ACCOUNT_ID,
        exchange=BITWANA_EXCHANGE,
        credential_ref=custom_creds,
    )

    # Calling seed_bitwana_venue again must NOT overwrite the account
    seed_bitwana_venue(store)

    account = store.get_account(BITWANA_ACCOUNT_ID)
    assert account.credential_ref == custom_creds


