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
from mm.exchanges.nobitex.private_rest import NobitexApiKeyAuth
from mm.runtime.engine import build_runtime_components
from mm.tools.print_events import build_print_queues
from tests.unit.conftest import venues_payload


def test_runtime_config_global_is_optional_secondary_source() -> None:
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        global_=MarketDataSourceConfig(role="global", exchange="wallex", symbol="BTC_IRT"),
        interval_ms=1000,
    )

    contract = config.as_contract()

    assert contract["venues"]["global"]["role"] == "global"
    assert config.joint_frame_topic == "strategy:joint:frames:nobitex:BTC_IRT:1000"
    assert config.binding.comparable is True


def test_runtime_config_auto_enables_persistence_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MM_DATABASE_URL", "postgresql://example")
    config = runtime_config_from_mapping(venues_payload(local={"exchange": "nobitex", "symbol": "BTC_IRT"}))

    assert config.persistence.enabled is True


def test_load_mm_env_reads_repo_root_env(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "MM_DATABASE_URL=postgresql://repo\nNOBITEX_API_TOKEN=repo-token\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MM_DATABASE_URL", raising=False)
    monkeypatch.delenv("NOBITEX_API_TOKEN", raising=False)
    monkeypatch.setattr("mm.env_loader.repo_root", lambda: tmp_path)

    assert load_mm_env() is True

    assert os.environ["MM_DATABASE_URL"] == "postgresql://repo"
    assert os.environ["NOBITEX_API_TOKEN"] == "repo-token"


def test_private_runtime_config_resolves_nobitex_api_key_auth(monkeypatch) -> None:
    monkeypatch.setenv("NOBITEX_API_KEY", "public-key")
    monkeypatch.setenv("NOBITEX_API_PRIVATE_KEY", "private-key")
    config = PrivateRuntimeConfig()

    credentials = config.resolved_api_credentials("nobitex")

    assert isinstance(credentials, NobitexApiKeyAuth)
    assert credentials.public_key == "public-key"
    assert credentials.private_key == "private-key"


def test_private_runtime_config_prefers_nobitex_api_key_auth_over_token(monkeypatch) -> None:
    monkeypatch.setenv("NOBITEX_API_TOKEN", "legacy-token")
    monkeypatch.setenv("NOBITEX_API_KEY", "public-key")
    monkeypatch.setenv("NOBITEX_API_PRIVATE_KEY", "private-key")
    config = PrivateRuntimeConfig(api_token_env="NOBITEX_API_TOKEN")

    credentials = config.resolved_api_credentials("nobitex")

    assert isinstance(credentials, NobitexApiKeyAuth)
    assert credentials.public_key == "public-key"


def test_private_runtime_config_parses_fill_event_source() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(local={"role": "local", "exchange": "nobitex", "symbol": "BTC_IRT"}),
            "local_private": {
                "api_token_env": "NOBITEX_API_TOKEN",
                "fill_event_source": "open_order_delta",
            },
        }
    )

    assert config.local_private is not None
    assert config.local_private.fill_event_source == "open_order_delta"


def test_private_runtime_config_rejects_invalid_fill_event_source() -> None:
    with pytest.raises(ValueError, match="fill_event_source"):
        PrivateRuntimeConfig(fill_event_source="both")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_persistence_enabled_without_database_url_builds_no_writer(monkeypatch) -> None:
    monkeypatch.delenv("MM_DATABASE_URL", raising=False)
    monkeypatch.delenv("MM_PARQUET_ROOT", raising=False)
    monkeypatch.setattr("mm.runtime.engine.load_mm_env", lambda: False)
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
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
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        persistence=PersistenceRuntimeConfig(enabled=True),
    )

    runtime = await build_runtime_components(config)

    assert [type(daemon).__name__ for daemon in runtime.persistence_daemons] == ["PostgresEventWriter"]


def test_controller_placement_settings_from_mapping() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(local={"exchange": "nobitex", "symbol": "BTC_IRT"}),
            "controller": {
                "enabled": True,
                "placement_timeout_ms": 2500,
                "placement_max_concurrency": 4,
            },
        }
    )
    assert config.controller is not None
    assert config.controller.placement_timeout_ms == 2500
    assert config.controller.placement_max_concurrency == 4


def test_controller_action_start_after_steps_from_mapping() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(local={"role": "local", "exchange": "nobitex", "symbol": "BTC_IRT"}),
            "interval_ms": 1000,
            "controller": {"enabled": True, "action_start_after_steps": 15},
        }
    )
    assert config.controller is not None
    assert config.controller.action_start_after_steps == 15


def test_persistence_topic_templates_resolve_from_interval_and_venues() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(
                local={"exchange": "nobitex", "symbol": "USDT_IRT"},
                global_={"exchange": "wallex", "symbol": "USDT_IRT"},
            ),
            "interval_ms": 2500,
            "persistence": {
                "enabled": True,
                "topics": [
                    "{local_market_event_topic}",
                    "{global_market_event_topic}",
                    "{joint_frame_topic}",
                    "{local_execution_orders_topic}",
                    "{global_execution_state_topic}",
                    "{local_private_snapshot_topic}",
                    "{local_private_errors_topic}",
                    "{global_private_snapshot_topic}",
                    "{global_private_errors_topic}",
                    "{execution_errors_topic}",
                    "{ledger_topic}",
                ],
            },
        }
    )

    assert config.resolved_persistence_topics == (
        "market:events:nobitex:USDT_IRT:2500",
        "market:events:wallex:USDT_IRT:2500",
        "strategy:joint:frames:nobitex:USDT_IRT:2500",
        "execution:orders:nobitex",
        "execution:state:wallex",
        "private:state:nobitex:snapshot",
        "private:state:nobitex:errors",
        "private:state:wallex:snapshot",
        "private:state:wallex:errors",
        "execution:errors",
        "ledger:snapshot",
    )
    assert config.effective_persistence_topics == config.resolved_persistence_topics


def test_persistence_topics_default_when_unset() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(
                local={"exchange": "nobitex", "symbol": "USDT_IRT"},
                global_={"exchange": "wallex", "symbol": "USDT_IRT"},
            ),
            "interval_ms": 2500,
            "controller": {"enabled": True, "actor_id": "mpc-controller"},
            "persistence": {"enabled": True},
        }
    )

    assert config.persistence.topics == ()
    assert config.resolved_persistence_topics == ()
    assert "market:events:nobitex:USDT_IRT:2500" in config.effective_persistence_topics
    assert "market:events:wallex:USDT_IRT:2500" in config.effective_persistence_topics
    assert "strategy:joint:frames:nobitex:USDT_IRT:2500" in config.effective_persistence_topics
    assert "strategy:diagnostics:mpc-controller" in config.effective_persistence_topics
    assert "strategy:target_books:mpc-controller" in config.effective_persistence_topics
    assert "execution:errors" in config.effective_persistence_topics
    assert "private:state:nobitex:errors" in config.effective_persistence_topics
    assert "private:state:wallex:errors" in config.effective_persistence_topics


def test_persistence_topic_template_rejects_unknown_placeholder() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
            "persistence": {"enabled": True, "topics": ["market:events:{unknown}"]},
        }
    )

    with pytest.raises(ValueError, match="unknown persistence topic placeholder"):
        _ = config.resolved_persistence_topics


def test_controller_reconcile_from_mapping() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(local={"role": "local", "exchange": "nobitex", "symbol": "BTC_IRT"}),
            "interval_ms": 1000,
            "controller": {
                "enabled": True,
                "reconcile": {"price_tol_bps": 5.0, "qty_tol_rel": 0.01},
            },
        }
    )
    assert config.controller is not None
    assert config.controller.reconcile.price_tol_bps == 5.0
    assert config.controller.reconcile.qty_tol_rel == 0.01


def test_controller_reconcile_defaults_when_missing() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(local={"role": "local", "exchange": "nobitex", "symbol": "BTC_IRT"}),
            "interval_ms": 1000,
            "controller": {"enabled": True},
        }
    )
    assert config.controller is not None
    assert config.controller.reconcile.price_tol_bps == 0.0
    assert config.controller.reconcile.qty_tol_rel == 0.0


def test_controller_action_start_after_steps_clamps_negative() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(local={"role": "local", "exchange": "nobitex", "symbol": "BTC_IRT"}),
            "interval_ms": 1000,
            "controller": {"enabled": True, "action_start_after_steps": -3},
        }
    )
    assert config.controller is not None
    assert config.controller.action_start_after_steps == 0


def test_runtime_config_rejects_invalid_global_role() -> None:
    with pytest.raises(ValueError, match="global source role"):
        MarketMakerRuntimeConfig(
            local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
            global_=MarketDataSourceConfig(role="local", exchange="wallex", symbol="BTC_IRT"),
        )


@pytest.mark.asyncio
async def test_runtime_builder_wires_local_and_global_topics_without_starting_network() -> None:
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT", mode="rest"),
        global_=MarketDataSourceConfig(role="global", exchange="wallex", symbol="BTC_IRT", mode="websocket"),
        interval_ms=1000,
        local_private=PrivateRuntimeConfig(api_token=None),
    )

    runtime = await build_runtime_components(config)
    queues = build_print_queues(runtime, include_raw=True)
    labels = {label for label, _ in queues}

    assert runtime.frame_assembler.joint_output_topic == config.joint_frame_topic
    assert "market:events:nobitex:BTC_IRT:1000" in labels
    assert "market:events:wallex:BTC_IRT:1000" in labels
    assert "market:raw:nobitex:BTC_IRT:trades" in labels
    assert "market:raw:wallex:BTC_IRT:orderbook" in labels
    assert runtime.public_market_selections["local"].transport == "rest"
    assert runtime.public_market_selections["global"].transport == "websocket"
