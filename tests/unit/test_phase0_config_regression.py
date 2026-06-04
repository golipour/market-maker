"""Phase 0 regression tests: lock venues.local/global config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from mm.runtime.config import (
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    load_runtime_config,
    runtime_config_from_mapping,
)

EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "mm/configs/market_maker.runtime.example.json"


def _venues_config(*, global_: bool = True) -> dict:
    payload = {
        "venues": {
            "local": {"exchange": "nobitex", "symbol": "USDT_IRT", "mode": "rest"},
        },
        "interval_ms": 5000,
    }
    if global_:
        payload["venues"]["global"] = {
            "exchange": "wallex",
            "symbol": "USDT_IRT",
            "mode": "rest",
        }
    return payload


def test_single_venue_config_loads_without_global() -> None:
    config = runtime_config_from_mapping(_venues_config(global_=False))

    assert config.global_ is None
    assert config.local.normalized_exchange() == "nobitex"
    assert config.local.symbol == "USDT_IRT"
    assert config.joint_frame_topic == "strategy:joint:frames:nobitex:USDT_IRT:5000"
    assert config.local.market_event_topic(5000) == "market:events:nobitex:USDT_IRT:5000"

    contract = config.as_contract()
    assert contract["venues"]["global"] is None
    assert contract["joint_strategy_frame"]["topic"] == config.joint_frame_topic
    assert contract["controller"]["enabled"] is False


def test_local_plus_global_config_loads() -> None:
    config = runtime_config_from_mapping(
        {
            **_venues_config(),
            "interval_ms": 10000,
            "max_global_lag_ms": 2000,
            "global_grace_ms": 250,
            "local_private": {"api_token_env": "NOBITEX_API_TOKEN", "enable_execution": True},
            "global_private": {"api_token_env": "WALLEX_API_TOKEN", "enable_execution": True},
            "controller": {
                "enabled": True,
                "actor_id": "mpc-controller",
                "reconcile": {"price_tol_bps": 10.0, "qty_tol_rel": 0.01},
            },
        }
    )

    assert config.global_ is not None
    assert config.global_.role == "global"
    assert config.global_.normalized_exchange() == "wallex"
    assert config.global_.market_event_topic(10000) == "market:events:wallex:USDT_IRT:10000"
    assert config.max_global_lag_ms == 2000
    assert config.global_grace_ms == 250
    assert config.controller is not None
    assert config.controller.reconcile.price_tol_bps == 10.0

    contract = config.as_contract()
    assert contract["venues"]["global"]["exchange"] == "wallex"
    assert contract["max_global_lag_ms"] == 2000


def test_example_runtime_config_loads() -> None:
    config = load_runtime_config(EXAMPLE_CONFIG)

    assert config.local.normalized_exchange() == "wallex"
    assert config.local.symbol == "USDT_IRT"
    assert config.binding.local.exchange == "wallex"
    assert config.binding.local.symbol == "USDT_IRT"
    assert config.global_ is not None
    assert config.global_.normalized_exchange() == "nobitex"
    assert config.binding.global_ is not None
    assert config.binding.global_.exchange == "nobitex"
    assert config.binding.global_.symbol == "USDT_IRT"
    assert config.interval_ms == 10000
    assert config.controller is not None
    assert config.controller.enabled is True
    assert config.controller.actor_id == "mpc-controller"
    assert config.controller.action_start_after_steps == 10
    assert config.controller.execution_model == {
        "kind": "real",
        "local_maker_fee": 0.0015,
        "global_maker_fee": 0.0025,
    }
    assert config.liquidity_sourcing.enabled is False
    assert config.liquidity_sourcing.global_live_orders is False
    assert config.execution.local.planner == "replace"
    assert config.execution.global_.planner == "replace"
    assert config.bus.kind == "redis_streams"
    assert config.persistence.topics == ()
    assert config.resolved_persistence_topics == ()
    assert "market:events:wallex:USDT_IRT:10000" in config.effective_persistence_topics
    assert "market:events:nobitex:USDT_IRT:10000" in config.effective_persistence_topics
    assert "strategy:joint:frames:wallex:USDT_IRT:10000" in config.effective_persistence_topics
    assert "execution:errors" in config.effective_persistence_topics
    assert "private:state:wallex:errors" in config.effective_persistence_topics
    assert "private:state:nobitex:errors" in config.effective_persistence_topics


def test_single_venue_runtime_config_dataclass() -> None:
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"),
        interval_ms=1000,
    )

    assert config.global_ is None
    assert config.binding.comparable is True
    assert config.joint_frame_topic == "strategy:joint:frames:nobitex:BTC_IRT:1000"


@pytest.mark.parametrize(
    ("mapping", "expected_joint_topic"),
    [
        (
            {
                "venues": {"local": {"exchange": "nobitex", "symbol": "BTC_IRT"}},
                "interval_ms": 1000,
            },
            "strategy:joint:frames:nobitex:BTC_IRT:1000",
        ),
        (
            {
                "venues": {
                    "local": {"exchange": "nobitex", "symbol": "USDT_IRT"},
                    "global": {"exchange": "wallex", "symbol": "USDT_IRT"},
                },
                "interval_ms": 10000,
            },
            "strategy:joint:frames:nobitex:USDT_IRT:10000",
        ),
    ],
)
def test_config_joint_frame_topic_derived_from_local(mapping: dict, expected_joint_topic: str) -> None:
    config = runtime_config_from_mapping(mapping)
    assert config.joint_frame_topic == expected_joint_topic
