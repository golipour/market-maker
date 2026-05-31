from __future__ import annotations

import pytest

from mm.runtime.config import OrderbookGridConfig, runtime_config_from_mapping
from tests.unit.conftest import venues_payload


def test_orderbook_grid_levels_and_ladder() -> None:
    grid = OrderbookGridConfig(spread_max=2000.0, spread_step=100.0)
    assert grid.levels == 20
    assert grid.spread_ladder() == tuple(100.0 * (i + 1) for i in range(20))


def test_orderbook_grid_ladder_clamps_final_level_to_spread_max() -> None:
    grid = OrderbookGridConfig(spread_max=100.0, spread_step=30.0)
    assert grid.levels == 4
    assert grid.spread_ladder() == (30.0, 60.0, 90.0, 100.0)


def test_orderbook_grid_contract() -> None:
    grid = OrderbookGridConfig(spread_max=100.0, spread_step=25.0)
    contract = grid.as_contract()
    assert contract["spread_max"] == 100.0
    assert contract["spread_step"] == 25.0
    assert contract["levels"] == 4
    assert contract["spread_ladder"] == [25.0, 50.0, 75.0, 100.0]


def test_runtime_parses_orderbook_grid_spread_fields() -> None:
    config = runtime_config_from_mapping(
        {
            **venues_payload(local={"exchange": "nobitex", "symbol": "BTC_IRT"}),
            "controller": {
                "enabled": True,
                "orderbook_grid": {"spread_max": 500.0, "spread_step": 50.0},
            },
        }
    )
    assert config.controller is not None
    assert config.controller.orderbook_grid.spread_max == 500.0
    assert config.controller.orderbook_grid.spread_step == 50.0
    assert config.controller.orderbook_grid.levels == 10


def test_orderbook_grid_requires_spread_fields() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        runtime_config_from_mapping(
            {
                **venues_payload(local={"exchange": "nobitex", "symbol": "BTC_IRT"}, global_=False),
                "controller": {"enabled": True, "orderbook_grid": {"spread_max": 100.0}},
            }
        )


def test_orderbook_grid_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError, match="spread_step"):
        OrderbookGridConfig(spread_max=100.0, spread_step=0.0)
