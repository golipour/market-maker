"""ObservedStepExecutionModel fee attributes for MPC rollouts."""

from __future__ import annotations

from mm.core.orderflow_execution import ObservedStepExecutionModel


def test_maker_fee_always_set():
    m = ObservedStepExecutionModel(maker_fee=0.0001)
    assert m.maker_fee == 0.0001
    assert not hasattr(m, "hedge_fee")


def test_hedge_fee_set_when_provided():
    m = ObservedStepExecutionModel(maker_fee=0.0, hedge_fee=0.0000025)
    assert m.maker_fee == 0.0
    assert m.hedge_fee == 0.0000025


def test_hedge_fee_omitted_when_none():
    m = ObservedStepExecutionModel(maker_fee=0.0, hedge_fee=None)
    assert m.maker_fee == 0.0
    assert not hasattr(m, "hedge_fee")
