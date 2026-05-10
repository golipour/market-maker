from __future__ import annotations

import os

import pytest

from mm.runtime.config import (
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    PersistenceRuntimeConfig,
    PrivateRuntimeConfig,
    runtime_config_from_mapping,
)
from mm.env_loader import load_mm_env
from mm.runtime.engine import build_runtime_components
from mm.runtime.print_events import build_print_queues


def test_runtime_config_uses_hedge_as_only_secondary_source() -> None:
    config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
        hedge=MarketDataSourceConfig(role="hedge", exchange="wallex", symbol="BTC_IRT"),
        interval_ms=1000,
    )

    contract = config.as_contract()

    assert contract["strategy_input_frame"]["hedge_is_anchor"] is True
    assert contract["hedge"]["role"] == "hedge"
    assert config.frame_topic == "strategy:frames:nobitex:BTC_IRT:1000"
    assert config.binding.comparable is True


def test_runtime_config_auto_enables_persistence_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MM_DATABASE_URL", "postgresql://example")
    config = runtime_config_from_mapping(
        {"primary": {"role": "primary", "exchange": "nobitex", "symbol": "BTC_IRT"}}
    )

    assert config.persistence.enabled is True


def test_load_mm_env_reads_repo_and_package_env(tmp_path, monkeypatch) -> None:
    package_root = tmp_path / "mm"
    package_root.mkdir()
    (tmp_path / ".env").write_text("MM_DATABASE_URL=postgresql://repo\n", encoding="utf-8")
    (package_root / ".env").write_text("NOBITEX_API_TOKEN=package-token\n", encoding="utf-8")
    monkeypatch.delenv("MM_DATABASE_URL", raising=False)
    monkeypatch.delenv("NOBITEX_API_TOKEN", raising=False)
    monkeypatch.setattr("mm.env_loader.mm_package_root", lambda: package_root)

    assert load_mm_env() is True

    assert os.environ["MM_DATABASE_URL"] == "postgresql://repo"
    assert os.environ["NOBITEX_API_TOKEN"] == "package-token"


@pytest.mark.asyncio
async def test_persistence_enabled_without_database_url_builds_no_writer(monkeypatch) -> None:
    monkeypatch.delenv("MM_DATABASE_URL", raising=False)
    monkeypatch.delenv("MM_PARQUET_ROOT", raising=False)
    monkeypatch.setattr("mm.runtime.engine.load_mm_env", lambda: False)
    config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
        persistence=PersistenceRuntimeConfig(enabled=True),
    )

    runtime = await build_runtime_components(config)

    assert runtime.persistence_daemons == ()


@pytest.mark.asyncio
async def test_persistence_enabled_with_database_url_builds_postgres_writer(monkeypatch) -> None:
    monkeypatch.setenv("MM_DATABASE_URL", "postgresql://example")
    monkeypatch.delenv("MM_PARQUET_ROOT", raising=False)
    monkeypatch.setattr("mm.runtime.engine.load_mm_env", lambda: False)
    config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
        persistence=PersistenceRuntimeConfig(enabled=True),
    )

    runtime = await build_runtime_components(config)

    assert [type(daemon).__name__ for daemon in runtime.persistence_daemons] == ["PostgresEventWriter"]


def test_controller_action_start_after_steps_from_mapping() -> None:
    config = runtime_config_from_mapping(
        {
            "primary": {"role": "primary", "exchange": "nobitex", "symbol": "BTC_IRT"},
            "interval_ms": 1000,
            "controller": {"enabled": True, "action_start_after_steps": 15},
        }
    )
    assert config.controller is not None
    assert config.controller.action_start_after_steps == 15


def test_controller_action_start_after_steps_clamps_negative() -> None:
    config = runtime_config_from_mapping(
        {
            "primary": {"role": "primary", "exchange": "nobitex", "symbol": "BTC_IRT"},
            "interval_ms": 1000,
            "controller": {"enabled": True, "action_start_after_steps": -3},
        }
    )
    assert config.controller is not None
    assert config.controller.action_start_after_steps == 0


def test_runtime_config_rejects_anchor_like_secondary_roles() -> None:
    with pytest.raises(ValueError, match="hedge source role"):
        MarketMakerRuntimeConfig(
            primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
            hedge=MarketDataSourceConfig(role="primary", exchange="wallex", symbol="BTC_IRT"),
        )


@pytest.mark.asyncio
async def test_runtime_builder_wires_primary_and_hedge_topics_without_starting_network() -> None:
    config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT", mode="rest"),
        hedge=MarketDataSourceConfig(role="hedge", exchange="wallex", symbol="BTC_IRT", mode="websocket"),
        interval_ms=1000,
        primary_private=PrivateRuntimeConfig(api_token=None),
    )

    runtime = await build_runtime_components(config)
    queues = build_print_queues(runtime, include_raw=True)
    labels = {label for label, _ in queues}

    assert "strategy:frames:nobitex:BTC_IRT:1000" in labels
    assert "market:events:nobitex:BTC_IRT:1000" in labels
    assert "market:events:wallex:BTC_IRT:1000" in labels
    assert "market:raw:nobitex:BTC_IRT:trades" in labels
    assert "market:raw:wallex:BTC_IRT:orderbook" in labels
    assert runtime.public_market_selections["primary"].transport == "rest"
    assert runtime.public_market_selections["hedge"].transport == "websocket"
