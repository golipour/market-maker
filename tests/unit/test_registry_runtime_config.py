from __future__ import annotations

import pytest

from mm.registry.memory import MemoryRegistryStore
from mm.registry.runtime_config import runtime_config_from_registry, runtime_config_mapping_from_registry


def _seed_registry(store: MemoryRegistryStore) -> str:
    store.upsert_platform_settings(
        bus={"kind": "memory"},
        persistence={"enabled": False},
        observability={"metrics_enabled": False},
    )
    store.upsert_exchange(
        exchange="nobitex",
        config={
            "market_data": {"mode": "rest"},
            "private": {"mode": "rest", "poll_interval_ms": 5000},
            "fees": {"maker_fee": 0.0001},
        },
    )
    store.upsert_account(
        account_id="nobitex-main",
        exchange="nobitex",
        credential_ref="NOBITEX_API_TOKEN",
    )
    store.upsert_preset(
        preset_id="test_preset",
        config={
            "interval_ms": 5000,
            "liquidity_sourcing": {"mode": "off"},
            "controller": {"enabled": False, "actor_id": "placeholder"},
        },
    )
    store.upsert_strategy(
        strategy_id="btc-maker",
        actor_id="btc-maker-actor",
        account_id="nobitex-main",
        symbol="BTC_IRT",
        preset_id="test_preset",
        quoting_exchange="nobitex",
        liquidity_exchange="nobitex",
        status="active",
        config_overrides={"controller": {"dry_run": True}},
    )
    return "btc-maker"


def test_runtime_config_mapping_from_registry_uses_credential_ref() -> None:
    store = MemoryRegistryStore()
    strategy_id = _seed_registry(store)
    bundle = store.load_runtime_bundle(strategy_id)

    mapping = runtime_config_mapping_from_registry(**bundle)

    assert mapping["bots"]["btc-maker"]["symbol"] == "BTC_IRT"
    assert mapping["exchanges"]["nobitex"]["private"]["api_token_env"] == "NOBITEX_API_TOKEN"
    assert "api_token" not in mapping["exchanges"]["nobitex"]["private"]
    assert mapping["strategy_presets"]["test_preset"]["controller"]["actor_id"] == "btc-maker-actor"
    assert mapping["strategy_presets"]["test_preset"]["controller"]["dry_run"] is True
    assert mapping["bus"]["kind"] == "memory"


def test_runtime_config_from_registry_resolves_contract() -> None:
    store = MemoryRegistryStore()
    strategy_id = _seed_registry(store)
    bundle = store.load_runtime_bundle(strategy_id)

    config = runtime_config_from_registry(**bundle)

    assert config.bot is not None
    assert config.bot.bot_id == "btc-maker"
    assert config.local.exchange == "nobitex"
    assert config.local.symbol == "BTC_IRT"
    assert config.local_private is not None
    assert config.local_private.api_token_env == "NOBITEX_API_TOKEN"


def test_registry_store_rejects_actor_id_conflict() -> None:
    store = MemoryRegistryStore()
    _seed_registry(store)
    with pytest.raises(Exception, match="actor_id"):
        store.upsert_strategy(
            strategy_id="other",
            actor_id="btc-maker-actor",
            account_id="nobitex-main",
            symbol="ETH_IRT",
            preset_id="test_preset",
            quoting_exchange="nobitex",
            liquidity_exchange="nobitex",
        )
