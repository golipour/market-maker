"""ObservedStepExecutionModel fee attributes for MPC rollouts."""

from __future__ import annotations

from mm.core.orderflow_execution import ObservedStepExecutionModel


def test_maker_fee_always_set():
    m = ObservedStepExecutionModel(maker_fee=0.0001)
    assert m.maker_fee == 0.0001


def test_unknown_config_keys_rejected():
    unexpected_key = "global" + "_fee"
    try:
        ObservedStepExecutionModel(**{"maker_fee": 0.0, unexpected_key: 0.0})
    except TypeError:
        return
    raise AssertionError("unknown ObservedStepExecutionModel keys should be rejected")
