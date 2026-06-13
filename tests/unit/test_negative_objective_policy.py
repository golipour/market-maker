"""Tests for post-optimizer negative-objective action policy."""

from __future__ import annotations

import pytest

from mm.core.models.instrument import InstrumentSpec, SymbolBinding
from mm.mpc.mpc_state import State
from mm.mpc.optimizer_diagnostics import EpochObjectiveMetrics, OptimizerRunDiagnostics
from mm.runtime.config import ControllerRuntimeConfig, OrderbookGridConfig
from mm.strategies.negative_objective_policy import (
    apply_negative_objective_policy,
    best_risk_adjusted_return,
    normalize_negative_objective_action_mode,
    zero_sigmoid_action,
)
def _binding() -> SymbolBinding:
    return SymbolBinding(
        local=InstrumentSpec(
            exchange="nobitex",
            symbol="BTC_IRT",
            base_asset="BTC",
            quote_asset="IRT",
            price_step=1.0,
            qty_step=1e-6,
            min_qty=1e-6,
        )
    )


def _state() -> State:
    from mm.core.orderflow_state import OrderbookRelative

    return State(
        mid=100.0,
        ob_rel=OrderbookRelative(bids=(), asks=()),
        buy_vol=1.0,
        sell_vol=1.0,
        inventory=1.0,
        cash=10_000.0,
        buy_vwap=99.0,
        sell_vwap=101.0,
    )


def _config() -> ControllerRuntimeConfig:
    return ControllerRuntimeConfig(
        enabled=True,
        orderbook_grid=OrderbookGridConfig(spread_max=200.0, spread_step=20.0),
        optimizer={
            "n_components": 1,
            "local": {
                "quote_size_min_per_side": 0.3,
                "quote_size_max_per_side": 2.0,
            },
        },
    )


def _negative_optimizer_run(*, risk_adjusted_return: float) -> OptimizerRunDiagnostics:
    metrics = EpochObjectiveMetrics.from_tensors(
        objective_value=risk_adjusted_return,
        components={"risk_adjusted_return": risk_adjusted_return},
        expected_total_reward=risk_adjusted_return,
        expected_total_return=risk_adjusted_return,
        expected_next_reward=risk_adjusted_return,
        main_reward_mean=risk_adjusted_return,
        global_reward_mean=0.0,
    )
    return OptimizerRunDiagnostics(best=metrics, final=metrics)


def test_normalize_negative_objective_action_mode_accepts_legacy_aliases() -> None:
    assert normalize_negative_objective_action_mode("zero") == "zero_action"
    assert normalize_negative_objective_action_mode("min") == "floor_action"
    assert normalize_negative_objective_action_mode("off") == "optimizer_action"
    assert normalize_negative_objective_action_mode("max") == "zero_action"


def test_runtime_config_parses_negative_objective_action() -> None:
    config = ControllerRuntimeConfig(
        enabled=True,
        negative_objective_action=normalize_negative_objective_action_mode("zero_action"),
    )
    assert config.negative_objective_action == "zero_action"


def test_best_risk_adjusted_return_prefers_component() -> None:
    run = _negative_optimizer_run(risk_adjusted_return=-0.5)
    assert best_risk_adjusted_return(run) == pytest.approx(-0.5)


def test_apply_max_zeroes_action_when_objective_negative() -> None:
    local = zero_sigmoid_action(1)
    local = type(local)(
        bid_L=(1.0,),
        bid_k=local.bid_k,
        bid_x0=local.bid_x0,
        ask_L=(1.0,),
        ask_k=local.ask_k,
        ask_x0=local.ask_x0,
    )
    out, global_out, meta = apply_negative_objective_policy(
        mode="zero_action",
        local_action=local,
        global_action=None,
        optimizer_run=_negative_optimizer_run(risk_adjusted_return=-0.1),
        config=_config(),
        binding=_binding(),
        state=_state(),
        global_active=False,
    )
    assert meta["applied"] is True
    assert meta["resolved_action"] == "zero_action"
    assert out.effective_depth_bid(20.0) == 0.0
    assert out.effective_depth_ask(20.0) == 0.0
    assert global_out is None


def test_zero_action_respects_threshold() -> None:
    local = zero_sigmoid_action(1)
    out, _, meta = apply_negative_objective_policy(
        mode="zero_action",
        zero_threshold=0.2,
        local_action=local,
        global_action=None,
        optimizer_run=_negative_optimizer_run(risk_adjusted_return=-0.1),
        config=_config(),
        binding=_binding(),
        state=_state(),
        global_active=False,
    )
    assert meta["applied"] is False
    assert out is local


def test_apply_off_keeps_optimizer_action_when_objective_negative() -> None:
    local = zero_sigmoid_action(1)
    out, _, meta = apply_negative_objective_policy(
        mode="optimizer_action",
        local_action=local,
        global_action=None,
        optimizer_run=_negative_optimizer_run(risk_adjusted_return=-0.1),
        config=_config(),
        binding=_binding(),
        state=_state(),
        global_active=False,
    )
    assert meta["applied"] is False
    assert out is local


def test_apply_min_clamps_to_floor_when_objective_negative() -> None:
    local = zero_sigmoid_action(1)
    local = type(local)(
        bid_L=(2.0,),
        bid_k=local.bid_k,
        bid_x0=local.bid_x0,
        ask_L=(2.0,),
        ask_k=local.ask_k,
        ask_x0=local.ask_x0,
    )
    out, _, meta = apply_negative_objective_policy(
        mode="floor_action",
        local_action=local,
        global_action=None,
        optimizer_run=_negative_optimizer_run(risk_adjusted_return=-0.1),
        config=_config(),
        binding=_binding(),
        state=_state(),
        global_active=False,
    )
    assert meta["applied"] is True
    assert meta["resolved_action"] == "floor_action"
    assert out.effective_depth_bid(20.0) < local.effective_depth_bid(20.0)


def test_positive_objective_never_overrides() -> None:
    local = zero_sigmoid_action(1)
    out, _, meta = apply_negative_objective_policy(
        mode="zero_action",
        local_action=local,
        global_action=None,
        optimizer_run=_negative_optimizer_run(risk_adjusted_return=0.25),
        config=_config(),
        binding=_binding(),
        state=_state(),
        global_active=False,
    )
    assert meta["applied"] is False
    assert out is local
