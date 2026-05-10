"""mm RealExecutionModel is standalone (no market_making_controller import)."""

from __future__ import annotations

from mm.core.models.real_execution import RealExecutionModel


def test_maker_fee_only():
    m = RealExecutionModel(maker_fee=0.0001)
    assert m.maker_fee == 0.0001
    assert not hasattr(m, "hedge_fee")


def test_maker_and_hedge_fee():
    m = RealExecutionModel(maker_fee=0.00015, hedge_fee=0.0000025)
    assert m.maker_fee == 0.00015
    assert m.hedge_fee == 0.0000025


def test_unknown_config_keys_ignored():
    m = RealExecutionModel(maker_fee=0.0, extra_from_json=123)
    assert m.maker_fee == 0.0


def test_lives_under_mm_package():
    assert RealExecutionModel.__module__ == "mm.core.models.real_execution"
