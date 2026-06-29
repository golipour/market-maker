from __future__ import annotations

import pytest

from mm.mpc.mpc_tensor_batched import VenueOptimizerParams
from mm.mpc.objective_terms import BookQualityParams, BookQualitySoftScoring
from mm.mpc.l_param import ReservationSkewParams
from mm.mpc.optimizer_config_keys import (
    CUMULATIVE_QTY_CAP_AT_SPREAD_MAX,
    CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX,
    MAX_ABS_SKEW,
    PRESENCE_REFERENCE_QTY,
    RESERVATION_SKEW,
    SOFT_SCORING,
    TIGHTNESS_SPREAD_SCALE,
)


def test_venue_optimizer_params_rejects_legacy_keys() -> None:
    with pytest.raises(ValueError, match="removed keys"):
        VenueOptimizerParams.from_mapping({"quote_size_min_per_side": 1.0})


def test_venue_optimizer_params_canonical_round_trip() -> None:
    params = VenueOptimizerParams.from_mapping(
        {
            CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX: 1.0,
            RESERVATION_SKEW: {MAX_ABS_SKEW: 0.002},
            "book_quality": {
                SOFT_SCORING: {
                    TIGHTNESS_SPREAD_SCALE: 200.0,
                    PRESENCE_REFERENCE_QTY: 1.0,
                }
            },
        }
    )
    assert params.cumulative_qty_floor_at_spread_max == 1.0
    assert params.reservation_skew is not None
    assert params.reservation_skew.max_abs_skew == 0.002
    assert params.book_quality is not None
    assert params.book_quality.soft_scoring.tightness_spread_scale == 200.0

    out = params.as_config(spread_max=200.0, n_components=2)
    assert out[CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX] == 1.0
    assert out[RESERVATION_SKEW] == {MAX_ABS_SKEW: 0.002}
    assert out["book_quality"][SOFT_SCORING][TIGHTNESS_SPREAD_SCALE] == 200.0


def test_book_quality_soft_scoring_canonical_round_trip() -> None:
    soft = BookQualitySoftScoring.from_mapping(
        {
            PRESENCE_REFERENCE_QTY: 2.0,
            TIGHTNESS_SPREAD_SCALE: 80.0,
            "measure": "combined",
        }
    )
    assert soft.presence_reference_qty == 2.0
    assert soft.tightness_spread_scale == 80.0
    mapping = soft.as_mapping()
    assert mapping[PRESENCE_REFERENCE_QTY] == 2.0
    assert mapping[TIGHTNESS_SPREAD_SCALE] == 80.0


def test_book_quality_soft_scoring_rejects_legacy_keys() -> None:
    with pytest.raises(ValueError, match="removed keys"):
        BookQualitySoftScoring.from_mapping({"spread_target": 100.0})


def test_reservation_skew_rejects_legacy_keys() -> None:
    with pytest.raises(ValueError, match="removed keys"):
        ReservationSkewParams.from_mapping({"r_max": 0.001})


def test_reservation_skew_canonical_round_trip() -> None:
    skew = ReservationSkewParams.from_mapping({MAX_ABS_SKEW: 0.001})
    assert skew is not None
    assert skew.max_abs_skew == 0.001
    assert skew.as_mapping() == {MAX_ABS_SKEW: 0.001}


def test_book_quality_params_rejects_legacy_keys() -> None:
    with pytest.raises(ValueError, match="removed keys"):
        BookQualityParams.from_mapping({"depth_target": 1.0})
