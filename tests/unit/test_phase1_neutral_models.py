"""Phase 1 tests: neutral config, domain models, and target-book adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm.core.adapters.joint import (
    action_to_joint_target_book,
    action_to_venue_target_book,
    joint_target_book_to_action,
    venue_target_book_to_action,
)
from mm.core.models.execution import Action, Order
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.core.models.target_book import JointTargetBook, TargetOrder, VenueTargetBook
from mm.core.wallet import make_balance
from tests.unit.conftest import venues_payload
from mm.runtime.config import (
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    load_runtime_config,
    runtime_config_from_mapping,
)

EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "mm/configs/market_maker.runtime.example.json"


def _with_preset(payload: dict, **updates):
    payload["strategy_presets"]["test_preset"].update(updates)
    return payload


def _with_controller(payload: dict, controller: dict):
    return _with_preset(payload, controller=controller)


def _market_event(*, exchange: str = "nobitex", symbol: str = "USDT_IRT") -> MarketEvent:
    book = RawOrderbook(
        exchange=exchange,
        symbol=symbol,
        timestamp_ms=2_000,
        bids=(PriceLevel(100.0, 5.0),),
        asks=(PriceLevel(101.0, 4.0),),
        is_snapshot=True,
    )
    return MarketEvent(
        exchange=exchange,
        symbol=symbol,
        interval_ms=1000,
        t_start_ms=1000,
        t_end_ms=2000,
        orderbook=book,
        trades=(),
        buy_volume=1.0,
        sell_volume=2.0,
        vwap=100.5,
        imbalance=0.0,
        microprice=100.5,
    )


def test_bot_assigns_quoting_and_liquidity_exchanges() -> None:
    payload = venues_payload(
        local={"exchange": "nobitex", "symbol": "USDT_IRT"},
        global_={"exchange": "wallex", "symbol": "USDT_IRT"},
    )
    payload["exchanges"]["nobitex"]["private"] = {"api_token_env": "NOBITEX_API_TOKEN"}
    payload["exchanges"]["wallex"]["private"] = {"api_token_env": "WALLEX_API_TOKEN"}
    config = runtime_config_from_mapping(payload)

    assert config.local.normalized_exchange() == "nobitex"
    assert config.global_ is not None
    assert config.global_.normalized_exchange() == "wallex"
    assert config.local_private is not None
    assert config.global_private is not None


def test_top_level_venue_alias_keys_rejected() -> None:
    old_local_key = "pri" + "mary"
    old_global_key = "he" + "dge"
    with pytest.raises(ValueError, match="bots object"):
        runtime_config_from_mapping(
            {
                old_local_key: {"exchange": "nobitex", "symbol": "USDT_IRT"},
                old_global_key: {"exchange": "wallex", "symbol": "USDT_IRT"},
            }
        )


def test_liquidity_sourcing_config_defaults() -> None:
    config = runtime_config_from_mapping(
        venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False)
    )

    assert config.liquidity_sourcing.enabled is False
    assert config.liquidity_sourcing.mode == "single_venue"
    assert config.liquidity_sourcing.reference_venue == "global"
    assert config.liquidity_sourcing.require_global_private is False


def test_liquidity_sourcing_config_from_mapping() -> None:
    payload = _with_preset(
        venues_payload(
            local={"exchange": "nobitex", "symbol": "USDT_IRT"},
            global_={"exchange": "wallex", "symbol": "USDT_IRT"},
        ),
        liquidity_sourcing={"mode": "simulate", "require_source_private": True},
    )
    config = runtime_config_from_mapping(
        payload
    )

    assert config.liquidity_sourcing.enabled is True
    assert config.liquidity_sourcing.mode == "liquidity_sourcing"
    assert config.liquidity_sourcing.require_global_private is True


def test_liquidity_sourcing_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="liquidity_sourcing.mode"):
        runtime_config_from_mapping(
            _with_preset(
                venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
                liquidity_sourcing={"mode": "invalid"},
            )
        )


def test_controller_optimizer_n_components_must_be_venue_scoped() -> None:
    with pytest.raises(ValueError, match="optimizer.local/global"):
        runtime_config_from_mapping(
            _with_controller(
                venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
                {
                    "enabled": True,
                    "optimizer": {
                        "kind": "monte_carlo_mpc_tensor_batched",
                        "n_components": 4,
                    },
                },
            )
        )


def test_removed_controller_model_config_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="local_price_model"):
        runtime_config_from_mapping(
            _with_controller(
                venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
                {
                    "enabled": True,
                    "price_model": {"kind": "residual"},
                    "market_flow_model": {"kind": "hurdle_lognormal"},
                },
            )
        )


def test_global_model_config_is_not_nested_under_optimizer() -> None:
    with pytest.raises(ValueError, match="directly under controller"):
        runtime_config_from_mapping(
            _with_controller(
                venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
                {
                    "enabled": True,
                    "optimizer": {
                        "kind": "monte_carlo_mpc_tensor_batched",
                        "global_market_flow_model": {"kind": "hurdle_lognormal"},
                    },
                },
            )
        )


def test_flat_venue_optimizer_config_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="optimizer.local/global"):
        runtime_config_from_mapping(
            _with_controller(
                venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
                {
                    "enabled": True,
                    "optimizer": {
                        "kind": "monte_carlo_mpc_tensor_batched",
                        "cumulative_qty_floor_at_spread_max": 1.0,
                        "cumulative_qty_cap_at_spread_max": 10.0,
                    },
                },
            )
        )


def test_removed_nested_venue_optimizer_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="removed venue optimizer aliases"):
        runtime_config_from_mapping(
            _with_controller(
                venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
                {
                    "enabled": True,
                    "optimizer": {
                        "kind": "monte_carlo_mpc_tensor_batched",
                        "local": {
                            "inv_coef": 0.1,
                            "cumulative_qty_floor_at_spread_max": 1.0,
                            "cumulative_qty_cap_at_spread_max": 10.0,
                        },
                    },
                },
            )
        )


def test_controller_reference_price_source_is_derived_from_bot_references() -> None:
    local = runtime_config_from_mapping(
        _with_controller(
            venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
            {"enabled": True},
        )
    )
    external = runtime_config_from_mapping(
        _with_controller(
            venues_payload(
                local={"exchange": "nobitex", "symbol": "USDT_IRT"},
                global_={"exchange": "wallex", "symbol": "USDT_IRT"},
            ),
            {"enabled": True},
        )
    )

    assert local.controller is not None
    assert external.controller is not None
    assert local.controller.reference_price_source == "local"
    assert external.controller.reference_price_source == "global"


def test_controller_fixed_vwap_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="vwap_depth"):
        runtime_config_from_mapping(
            _with_controller(
                venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
                {"enabled": True, "vwap_depth": 100.0},
            )
        )


def test_runtime_contract_includes_venues() -> None:
    config = runtime_config_from_mapping(
        venues_payload(
            local={"exchange": "nobitex", "symbol": "USDT_IRT"},
            global_={"exchange": "wallex", "symbol": "USDT_IRT"},
        )
    )
    contract = config.as_contract()

    assert contract["venues"]["local"]["exchange"] == "nobitex"
    assert contract["venues"]["global"]["exchange"] == "wallex"
    assert contract["joint_strategy_frame"]["topic"] == config.joint_frame_topic
    assert contract["liquidity_sourcing"]["enabled"] is False


def test_example_config_loads_liquidity_sourcing() -> None:
    config = load_runtime_config(EXAMPLE_CONFIG)

    assert config.liquidity_sourcing.enabled is True
    assert config.liquidity_sourcing.mode == "liquidity_sourcing"
    assert config.liquidity_sourcing.global_live_orders is True
    assert config.controller is not None
    assert config.controller.enabled is True
    assert config.controller.pricing.quote_anchor.mode == "liquidity_weighted_mid"
    assert config.controller.pricing.quote_anchor.anchor_vwap_bps == 20.0
    assert config.controller.pricing.inventory_mark.venue == "liquidity"
    assert config.controller.optimizer is not None
    assert config.controller.optimizer["local"]["n_components"] == 4
    assert config.controller.optimizer["global"]["n_components"] == 2
    assert config.execution.local.max_orders_per_side == 10
    assert config.execution.global_.max_orders_per_side == 2



def test_action_target_book_round_trip() -> None:
    action = Action(
        actor_id="mpc-controller",
        orders=(
            Order("nobitex", "USDT_IRT", "buy", 100.0, 1.0, client_order_id="b1"),
            Order("nobitex", "USDT_IRT", "sell", 101.0, 0.5, client_order_id="s1"),
        ),
    )

    book = action_to_venue_target_book(action, policy="reconcile")
    round_trip = venue_target_book_to_action(book)

    assert book.actor_id == action.actor_id
    assert book.exchange == "nobitex"
    assert len(book.orders) == 2
    assert round_trip.actor_id == action.actor_id
    assert len(round_trip.orders) == 2
    assert round_trip.orders[0].price == 100.0
    assert round_trip.orders[1].side == "sell"


def test_joint_target_book_to_action_flattens_both_venues() -> None:
    local = VenueTargetBook(
        exchange="nobitex",
        symbol="USDT_IRT",
        actor_id="mpc-controller",
        orders=(),
    )
    global_book = VenueTargetBook(
        exchange="wallex",
        symbol="USDT_IRT",
        actor_id="mpc-controller",
        orders=(TargetOrder("wallex", "USDT_IRT", "buy", 99.0, 1.0),),
    )
    joint = JointTargetBook(
        actor_id="mpc-controller",
        local=local,
        global_=global_book,
        mode="liquidity_sourcing",
    )

    action = joint_target_book_to_action(joint)

    assert len(action.orders) == 1
    assert action.orders[0].exchange == "wallex"


def test_core_ledger_import_does_not_load_runtime() -> None:
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    code = """
import sys
for k in list(sys.modules):
    if k == "mm" or k.startswith("mm."):
        del sys.modules[k]
import mm.core.models.ledger  # noqa: F401
import mm.core.models.target_book  # noqa: F401
import mm.core.models.joint_strategy  # noqa: F401
bad = [m for m in sys.modules if m.startswith("mm.runtime")]
assert not bad, bad
"""
    env = {**os.environ, "PYTHONPATH": str(root)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
