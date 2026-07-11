from __future__ import annotations

import pytest

from mm.mpc.l_param import SpreadStructureParams
from mm.mpc.mpc_tensor_batched import VenueOptimizerParams
from mm.mpc.objective_terms import BookQualityHardStructure, BookQualityParams


def test_book_quality_hard_structure_parses_nested_config() -> None:
    params = VenueOptimizerParams.from_mapping(
        {
            "book_quality": {
                "hard_structure": {
                    "action_max_spread": 100.0,
                    "action_min_depth": 0.3,
                    "action_k_epsilon": 0.02,
                }
            }
        }
    )
    assert params.spread_structure is not None
    assert params.spread_structure.center_sum_max == pytest.approx(100.0)
    assert params.spread_structure.center_min_depth == pytest.approx(0.3)
    assert params.spread_structure.center_k_epsilon == pytest.approx(0.02)


def test_missing_hard_structure_is_unstructured() -> None:
    params = VenueOptimizerParams.from_mapping({"cumulative_qty_floor_at_spread_max": 1.0})
    l_param = params.l_param_config(spread_max=200.0, n_components=4)
    assert params.spread_structure is None
    assert params.reservation_skew is None
    assert l_param.spread_structured is False


def test_venue_level_spread_structure_is_rejected() -> None:
    with pytest.raises(ValueError, match="book_quality.hard_structure"):
        VenueOptimizerParams.from_mapping(
            {
                "spread_structure": {
                    "center_sum_max": 100.0,
                }
            }
        )


def test_hard_structure_sum_cap_only_enables_structured_mode() -> None:
    book = BookQualityParams(
        hard_structure=BookQualityHardStructure(action_max_spread=80.0, action_min_depth=0.0)
    )
    l_param = VenueOptimizerParams(book_quality=book).l_param_config(
        spread_max=200.0,
        n_components=2,
    )
    assert l_param.spread_structured is True
    assert l_param.quote_spread_max == pytest.approx(80.0)
    assert l_param.quote_spread_min_depth is None


def test_hard_structure_legacy_keys_rejected() -> None:
    with pytest.raises(ValueError, match="action_max_spread"):
        BookQualityHardStructure.from_mapping(
            {
                "center_sum_max": 100.0,
                "center_min_depth": 0.3,
            }
        )
