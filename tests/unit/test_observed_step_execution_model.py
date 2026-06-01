"""ObservedStepExecutionModel fee attributes for MPC rollouts."""

from __future__ import annotations

from mm.core.orderflow_execution import ObservedStepExecutionModel


def test_maker_fee_always_set():
    m = ObservedStepExecutionModel(maker_fee=0.0001)
    assert m.maker_fee == 0.0001
    assert m.local_maker_fee == 0.0001
    assert m.global_maker_fee == 0.0001


def test_venue_scoped_maker_fees_override_default():
    m = ObservedStepExecutionModel(
        maker_fee=0.0001,
        local_maker_fee=0.0002,
        global_maker_fee=0.0003,
    )
    assert m.maker_fee == 0.0001
    assert m.local_maker_fee == 0.0002
    assert m.global_maker_fee == 0.0003


def test_unknown_config_keys_rejected():
    unexpected_key = "global" + "_fee"
    try:
        ObservedStepExecutionModel(**{"maker_fee": 0.0, unexpected_key: 0.0})
    except TypeError:
        return
    raise AssertionError("unknown ObservedStepExecutionModel keys should be rejected")
