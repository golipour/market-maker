"""mm RealExecutionModel is standalone (no market_making_controller import)."""

from __future__ import annotations

from mm.core.models.real_execution import RealExecutionModel


def test_maker_fee():
    m = RealExecutionModel(maker_fee=0.0001)
    assert m.maker_fee == 0.0001
    assert m.local_maker_fee == 0.0001
    assert m.global_maker_fee == 0.0001


def test_venue_scoped_maker_fees_override_default_maker_fee():
    m = RealExecutionModel(maker_fee=0.0001, local_maker_fee=0.0002, global_maker_fee=0.0003)
    assert m.maker_fee == 0.0001
    assert m.local_maker_fee == 0.0002
    assert m.global_maker_fee == 0.0003


def test_unknown_config_keys_rejected():
    try:
        RealExecutionModel(maker_fee=0.0, extra_from_json=123)  # type: ignore[call-arg]
    except TypeError:
        return
    raise AssertionError("unknown RealExecutionModel keys should be rejected")


def test_lives_under_mm_package():
    assert RealExecutionModel.__module__ == "mm.core.models.real_execution"
