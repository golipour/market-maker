"""Phase 0 regression tests: lock bot/preset runtime config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from mm.runtime.config import (
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    load_runtime_config,
    runtime_config_from_mapping,
)
from tests.unit.conftest import venues_payload

EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "mm/configs/market_maker.runtime.example.json"


def _runtime_payload(*, global_: bool = True) -> dict:
    payload = venues_payload(
        local={"exchange": "nobitex", "symbol": "USDT_IRT", "mode": "rest"},
        global_={"exchange": "wallex", "symbol": "USDT_IRT", "mode": "rest"} if global_ else False,
    )
    payload["strategy_presets"]["test_preset"]["interval_ms"] = 5000
    return payload


def test_single_venue_config_loads_without_global() -> None:
    config = runtime_config_from_mapping(_runtime_payload(global_=False))

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
            **_runtime_payload(),
            "strategy_presets": {
                "test_preset": {
                    "interval_ms": 10000,
                    "max_global_lag_ms": 2000,
                    "global_grace_ms": 250,
                    "controller": {
                "enabled": True,
                "actor_id": "mpc-controller",
                "reconcile": {"price_tol_bps": 10.0, "qty_tol_rel": 0.01},
            },
                }
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

    assert config.local.normalized_exchange() == "bitpin"
    assert config.local.symbol == "PUMP_IRT"
    assert config.binding.local.exchange == "bitpin"
    assert config.binding.local.symbol == "PUMP_IRT"
    assert config.global_ is None
    assert config.interval_ms == 10000
    assert config.controller is not None
    assert config.controller.enabled is True
    assert config.controller.actor_id == "mpc-controller"
    assert config.controller.execution_after_steps == 10
    assert config.controller.negative_objective_action == "optimizer_action"
    assert config.controller.execution_model == {
        "kind": "real",
        "local_maker_fee": 3e-8,
    }
    assert config.liquidity_sourcing.enabled is False
    assert config.liquidity_sourcing.global_live_orders is False
    assert config.execution.local.enabled is True
    assert config.execution.global_.enabled is False
    assert config.execution.local.planner == "replace"
    assert config.execution.global_.planner == "replace"
    assert config.bus.kind == "redis_streams"
    assert config.persistence.topics == ()
    assert config.resolved_persistence_topics == ()
    assert "market:events:bitpin:PUMP_IRT:10000" in config.effective_persistence_topics
    assert "strategy:joint:frames:bitpin:PUMP_IRT:10000" in config.effective_persistence_topics
    assert "execution:errors" in config.effective_persistence_topics
    assert "private:state:bitpin:errors" in config.effective_persistence_topics
    local_optimizer = (config.controller.optimizer or {}).get("local", {})
    assert local_optimizer.get("w_book") == pytest.approx(0.002)
    assert local_optimizer.get("w_inv") == pytest.approx(0.01)
    assert "spread_structure" not in local_optimizer
    global_optimizer = (config.controller.optimizer or {}).get("global", {})
    assert global_optimizer.get("book_quality") == {
        "hard_structure": {
            "action_max_spread": 100.0,
            "action_min_depth": 0.3,
            "action_k_epsilon": 0.01,
        },
        "soft_scoring": {
            "tightness_spread_scale": 100.0,
            "presence_reference_qty": 1.0,
            "measure": "quoted",
        },
    }


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
                venues_payload(local={"exchange": "nobitex", "symbol": "BTC_IRT"}, global_=False)
                | {"strategy_presets": {"test_preset": {"interval_ms": 1000}}},
                "strategy:joint:frames:nobitex:BTC_IRT:1000",
            ),
            (
                venues_payload(
                    local={"exchange": "nobitex", "symbol": "USDT_IRT"},
                    global_={"exchange": "wallex", "symbol": "USDT_IRT"},
                )
                | {"strategy_presets": {"test_preset": {"interval_ms": 10000}}},
                "strategy:joint:frames:nobitex:USDT_IRT:10000",
            ),
    ],
)
def test_config_joint_frame_topic_derived_from_local(mapping: dict, expected_joint_topic: str) -> None:
    config = runtime_config_from_mapping(mapping)
    assert config.joint_frame_topic == expected_joint_topic


def test_flat_quote_spread_keys_are_rejected_in_optimizer_config() -> None:
    payload = _runtime_payload()
    payload["strategy_presets"]["test_preset"]["controller"] = {
        "enabled": True,
        "optimizer": {
            "local": {
                "cumulative_qty_floor_at_spread_max": 1.0,
                "quote_spread_max": 100.0,
            }
        },
    }
    with pytest.raises(ValueError, match="quote_spread_max"):
        runtime_config_from_mapping(payload)


def test_quote_imbalance_skew_strength_is_rejected_in_optimizer_config() -> None:
    payload = _runtime_payload()
    payload["strategy_presets"]["test_preset"]["controller"] = {
        "enabled": True,
        "optimizer": {
            "local": {
                "cumulative_qty_floor_at_spread_max": 1.0,
                "quote_imbalance_skew_strength": 0.5,
            }
        },
    }
    with pytest.raises(ValueError, match="quote_imbalance_skew_strength"):
        runtime_config_from_mapping(payload)


def test_quote_capacity_ratio_is_rejected_in_optimizer_config() -> None:
    payload = _runtime_payload()
    payload["strategy_presets"]["test_preset"]["controller"] = {
        "enabled": True,
        "optimizer": {
            "local": {
                "cumulative_qty_floor_at_spread_max": 1.0,
                "quote_capacity_ratio": 0.5,
            }
        },
    }
    with pytest.raises(ValueError, match="quote_capacity_ratio"):
        runtime_config_from_mapping(payload)
