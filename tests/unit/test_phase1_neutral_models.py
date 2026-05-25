"""Phase 1 tests: neutral config aliases, domain models, and compatibility adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm.core.adapters.joint import (
    action_to_joint_target_book,
    action_to_venue_target_book,
    joint_target_book_to_action,
    strategy_input_frame_to_joint_strategy_frame,
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


def test_venues_local_global_config_aliases() -> None:
    config = runtime_config_from_mapping(
        {
            "venues": {
                "local": {"exchange": "nobitex", "symbol": "USDT_IRT"},
                "global": {"exchange": "wallex", "symbol": "USDT_IRT"},
            },
            "local_private": {"api_token_env": "NOBITEX_API_TOKEN"},
            "global_private": {"api_token_env": "WALLEX_API_TOKEN"},
        }
    )

    assert config.local.normalized_exchange() == "nobitex"
    assert config.global_ is not None
    assert config.global_.normalized_exchange() == "wallex"
    assert config.local_private is config.local_private
    assert config.global_private is config.global_private
    assert config.config_warnings == ()


def test_legacy_primary_hedge_keys_warn_and_map_to_local_global() -> None:
    config = runtime_config_from_mapping(
        {
            "primary": {"exchange": "nobitex", "symbol": "USDT_IRT"},
            "hedge": {"exchange": "wallex", "symbol": "USDT_IRT"},
        }
    )

    assert config.local.normalized_exchange() == "nobitex"
    assert config.global_ is not None
    assert config.global_.normalized_exchange() == "wallex"
    assert any("primary" in warning for warning in config.config_warnings)
    assert any("hedge" in warning for warning in config.config_warnings)


def test_liquidity_sourcing_config_defaults() -> None:
    config = runtime_config_from_mapping(
        venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False)
    )

    assert config.liquidity_sourcing.enabled is False
    assert config.liquidity_sourcing.mode == "single_venue"
    assert config.liquidity_sourcing.reference_venue == "global"
    assert config.liquidity_sourcing.require_global_private is False


def test_liquidity_sourcing_config_from_mapping() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
            "liquidity_sourcing": {
                "enabled": True,
                "mode": "liquidity_sourcing",
                "reference_venue": "global",
                "require_global_private": True,
            },
        }
    )

    assert config.liquidity_sourcing.enabled is True
    assert config.liquidity_sourcing.mode == "liquidity_sourcing"
    assert config.liquidity_sourcing.require_global_private is True


def test_liquidity_sourcing_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="liquidity_sourcing.mode"):
        runtime_config_from_mapping(
            {
                **venues_payload(local={"exchange": "nobitex", "symbol": "USDT_IRT"}, global_=False),
                "liquidity_sourcing": {"mode": "invalid"},
            }
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
    assert config.controller is not None
    assert config.controller.enabled is True



def test_joint_models_serialize_through_json() -> None:
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="USDT_IRT"),
        global_=MarketDataSourceConfig(role="global", exchange="wallex", symbol="USDT_IRT"),
    )
    local_private = PrivateStateSnapshot(
        "nobitex",
        2000,
        balances={
            "USDT": make_balance("nobitex", "USDT", 10.0, 0.0, 2000),
            "IRT": make_balance("nobitex", "IRT", 1_000_000.0, 0.0, 2000),
        },
    )
    frame = StrategyInputFrame(local=_market_event(exchange="nobitex"),
        global_=_market_event(exchange="wallex"),
        local_private=local_private,
        global_status="ready",
        global_lag_ms=50,
        created_at_ms=2100,
    )

    joint_frame = strategy_input_frame_to_joint_strategy_frame(frame, runtime.binding)
    payload = joint_frame.as_payload()
    json.dumps(payload)

    assert payload["mode"] == "single_venue"
    assert payload["sync_status"]["local_market_ready"] is True
    assert payload["sync_status"]["global_market_ready"] is True
    assert payload["ledger"]["local"]["exchange"] == "nobitex"


def test_strategy_input_frame_adapter_respects_liquidity_sourcing_mode() -> None:
    runtime = MarketMakerRuntimeConfig(local=MarketDataSourceConfig(role="local", exchange="nobitex", symbol="USDT_IRT"),
    )
    frame = StrategyInputFrame(local=_market_event())

    joint_frame = strategy_input_frame_to_joint_strategy_frame(
        frame,
        runtime.binding,
        liquidity_sourcing_enabled=True,
        liquidity_sourcing_mode="liquidity_sourcing",
    )

    assert joint_frame.mode == "liquidity_sourcing"


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
