"""Phase 0 regression tests: lock current single-venue and primary+hedge config loading."""

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


def test_single_venue_config_loads_without_hedge() -> None:
    config = runtime_config_from_mapping(
        {
            "primary": {"exchange": "nobitex", "symbol": "USDT_IRT", "mode": "rest"},
            "interval_ms": 5000,
            "controller": {"enabled": True, "actor_id": "mpc-controller"},
        }
    )

    assert config.hedge is None
    assert config.primary.normalized_exchange() == "nobitex"
    assert config.primary.symbol == "USDT_IRT"
    assert config.frame_topic == "strategy:frames:nobitex:USDT_IRT:5000"
    assert config.primary.market_event_topic(5000) == "market:events:nobitex:USDT_IRT:5000"

    contract = config.as_contract()
    assert contract["hedge"] is None
    assert contract["strategy_input_frame"]["topic"] == config.frame_topic
    assert contract["controller"]["enabled"] is True


def test_primary_plus_hedge_config_loads() -> None:
    config = runtime_config_from_mapping(
        {
            "primary": {"exchange": "nobitex", "symbol": "USDT_IRT", "mode": "rest"},
            "hedge": {"exchange": "wallex", "symbol": "USDT_IRT", "mode": "rest"},
            "interval_ms": 10000,
            "max_hedge_lag_ms": 2000,
            "hedge_grace_ms": 250,
            "primary_private": {"api_token_env": "NOBITEX_API_TOKEN", "enable_execution": True},
            "hedge_private": {"api_token_env": "WALLEX_API_TOKEN", "enable_execution": True},
            "controller": {
                "enabled": True,
                "actor_id": "mpc-controller",
                "reconcile": {"price_tol_bps": 10.0, "qty_tol_rel": 0.01},
            },
            "hedging": {"enabled": False, "observe_only": True, "actor_id": "hedge-executor"},
        }
    )

    assert config.hedge is not None
    assert config.hedge.role == "hedge"
    assert config.hedge.normalized_exchange() == "wallex"
    assert config.hedge.market_event_topic(10000) == "market:events:wallex:USDT_IRT:10000"
    assert config.max_hedge_lag_ms == 2000
    assert config.hedge_grace_ms == 250
    assert config.controller is not None
    assert config.controller.reconcile.price_tol_bps == 10.0
    assert config.hedging is not None
    assert config.hedging.enabled is False

    contract = config.as_contract()
    assert contract["hedge"]["exchange"] == "wallex"
    assert contract["strategy_input_frame"]["hedge_is_anchor"] is True
    assert contract["strategy_input_frame"]["max_hedge_lag_ms"] == 2000


def test_example_runtime_config_loads() -> None:
    config = load_runtime_config(EXAMPLE_CONFIG)

    assert config.primary.normalized_exchange() == "nobitex"
    assert config.primary.symbol == "USDT_IRT"
    assert config.hedge is not None
    assert config.hedge.normalized_exchange() == "wallex"
    assert config.interval_ms == 10000
    assert config.controller is not None
    assert config.controller.enabled is True
    assert config.controller.actor_id == "mpc-controller"
    assert config.controller.action_start_after_steps == 10
    assert config.controller.optimizer is not None
    assert config.controller.optimizer.get("hedge_rollout_enabled") is False
    assert config.hedging is not None
    assert config.hedging.enabled is False
    assert config.bus.kind == "redis_streams"


def test_single_venue_runtime_config_dataclass() -> None:
    config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
        interval_ms=1000,
    )

    assert config.hedge is None
    assert config.binding.comparable is True
    assert config.frame_topic == "strategy:frames:nobitex:BTC_IRT:1000"


@pytest.mark.parametrize(
    ("mapping", "expected_frame_topic"),
    [
        (
            {"primary": {"exchange": "nobitex", "symbol": "BTC_IRT"}, "interval_ms": 1000},
            "strategy:frames:nobitex:BTC_IRT:1000",
        ),
        (
            {
                "primary": {"exchange": "nobitex", "symbol": "USDT_IRT"},
                "hedge": {"exchange": "wallex", "symbol": "USDT_IRT"},
                "interval_ms": 10000,
            },
            "strategy:frames:nobitex:USDT_IRT:10000",
        ),
    ],
)
def test_config_frame_topic_derived_from_primary(mapping: dict, expected_frame_topic: str) -> None:
    config = runtime_config_from_mapping(mapping)
    assert config.frame_topic == expected_frame_topic
