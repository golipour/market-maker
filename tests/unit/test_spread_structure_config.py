from __future__ import annotations

import pytest

from mm.mpc.l_param import SpreadStructureParams
from mm.mpc.mpc_tensor_batched import VenueOptimizerParams


def test_spread_structure_parses_nested_config() -> None:
    params = VenueOptimizerParams.from_mapping(
        {
            "spread_structure": {
                "center_sum_max": 100.0,
                "center_min_depth": 0.3,
                "center_k_epsilon": 0.02,
            }
        }
    )
    assert params.spread_structure is not None
    assert params.spread_structure.center_sum_max == pytest.approx(100.0)
    assert params.spread_structure.center_min_depth == pytest.approx(0.3)
    assert params.spread_structure.center_k_epsilon == pytest.approx(0.02)


def test_missing_spread_structure_is_unstructured() -> None:
    params = VenueOptimizerParams.from_mapping({"quote_size_min_per_side": 1.0})
    l_param = params.l_param_config(spread_max=200.0, n_components=4)
    assert params.spread_structure is None
    assert params.reservation_offset is None
    assert l_param.spread_structured is False


def test_spread_structure_sum_cap_only_enables_structured_mode() -> None:
    spread = SpreadStructureParams(center_sum_max=80.0, center_min_depth=0.0)
    l_param = VenueOptimizerParams(spread_structure=spread).l_param_config(
        spread_max=200.0,
        n_components=2,
    )
    assert l_param.spread_structured is True
    assert l_param.quote_spread_max == pytest.approx(80.0)
    assert l_param.quote_spread_min_depth is None
