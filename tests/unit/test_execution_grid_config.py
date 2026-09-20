from __future__ import annotations

import math

import pytest

from mm.runtime.config import (
    ExecutionGridConfig,
    OrderbookGridConfig,
    runtime_config_from_mapping,
)
from tests.unit.conftest import venues_payload


def _with_controller(payload: dict, controller: dict) -> dict:
    payload["strategy_presets"]["test_preset"]["controller"] = controller
    return payload


def test_default_execution_grid_mirrors_orderbook_grid() -> None:
    canonical = OrderbookGridConfig(spread_max=100.0, spread_step=30.0)
    execution = ExecutionGridConfig()
    assert execution.spread_ladder(canonical=canonical) == canonical.spread_ladder()
    assert execution.spread_ladder(canonical=canonical) == (30.0, 60.0, 90.0, 100.0)


def test_uniform_override_stays_inside_canonical_max() -> None:
    canonical = OrderbookGridConfig(spread_max=200.0, spread_step=20.0)
    execution = ExecutionGridConfig(mode="uniform", spread_max=100.0, spread_step=25.0)
    ladder = execution.spread_ladder(canonical=canonical)
    assert ladder == (25.0, 50.0, 75.0, 100.0)
    assert ladder[-1] == 100.0


def test_execution_spread_max_cannot_exceed_canonical() -> None:
    canonical = OrderbookGridConfig(spread_max=100.0, spread_step=10.0)
    execution = ExecutionGridConfig(mode="uniform", spread_max=150.0)
    with pytest.raises(ValueError, match="must be <= orderbook_grid.spread_max"):
        execution.spread_ladder(canonical=canonical)


def test_log_ladder_is_geometric_and_includes_outer_edge() -> None:
    canonical = OrderbookGridConfig(spread_max=1000.0, spread_step=50.0)
    execution = ExecutionGridConfig(mode="log", spread_min=50.0, levels=5, spread_max=1000.0)
    ladder = execution.spread_ladder(canonical=canonical)
    assert len(ladder) == 5
    assert ladder[0] == pytest.approx(50.0)
    assert ladder[-1] == pytest.approx(1000.0)
    # Constant log spacing: ratios between consecutive points are equal.
    ratios = [ladder[i + 1] / ladder[i] for i in range(len(ladder) - 1)]
    assert all(math.isclose(r, ratios[0], rel_tol=1e-9) for r in ratios)


def test_hybrid_ladder_has_linear_inner_then_geometric_outer() -> None:
    canonical = OrderbookGridConfig(spread_max=1000.0, spread_step=10.0)
    execution = ExecutionGridConfig(
        mode="hybrid",
        inner_linear_max=50.0,
        inner_step=10.0,
        levels=4,
        spread_max=1000.0,
    )
    ladder = execution.spread_ladder(canonical=canonical)
    assert ladder[0] == pytest.approx(10.0)
    assert 50.0 in {round(s, 9) for s in ladder} or any(abs(s - 50.0) < 1e-6 for s in ladder)
    assert ladder[-1] == pytest.approx(1000.0)
    assert all(ladder[i] < ladder[i + 1] for i in range(len(ladder) - 1))


def test_explicit_ladder_appends_spread_max() -> None:
    canonical = OrderbookGridConfig(spread_max=200.0, spread_step=20.0)
    execution = ExecutionGridConfig(mode="explicit", spreads=(20.0, 80.0, 150.0))
    assert execution.spread_ladder(canonical=canonical) == (20.0, 80.0, 150.0, 200.0)


def test_runtime_parses_execution_grid() -> None:
    config = runtime_config_from_mapping(
        _with_controller(
            venues_payload(local={"exchange": "nobitex", "symbol": "BTC_IRT"}),
            {
                "enabled": True,
                "orderbook_grid": {"spread_max": 200.0, "spread_step": 20.0},
                "execution_grid": {
                    "mode": "log",
                    "spread_min": 20.0,
                    "levels": 6,
                },
            },
        )
    )
    assert config.controller is not None
    ladder = config.controller.execution_spread_ladder()
    assert ladder[0] == pytest.approx(20.0)
    assert ladder[-1] == pytest.approx(200.0)
    assert len(ladder) == 6
    contract = config.controller.execution_grid.as_contract(
        canonical=config.controller.orderbook_grid
    )
    assert contract["mode"] == "log"
    assert contract["role"] == "execution_only"
    assert contract["spread_ladder"] == list(ladder)


def test_runtime_rejects_invalid_execution_grid() -> None:
    with pytest.raises(ValueError, match="spread_min"):
        runtime_config_from_mapping(
            _with_controller(
                venues_payload(local={"exchange": "nobitex", "symbol": "BTC_IRT"}, global_=False),
                {
                    "enabled": True,
                    "orderbook_grid": {"spread_max": 200.0, "spread_step": 20.0},
                    "execution_grid": {"mode": "log", "levels": 4},
                },
            )
        )
