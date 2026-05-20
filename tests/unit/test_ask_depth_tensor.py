from __future__ import annotations

import math

import torch

from mm.mpc.state_tensor import effective_depth_ask_tensor, effective_depth_bid_tensor


def test_ask_and_bid_effective_depth_both_positive() -> None:
    """Regression: ask side must read params[3n:], not an empty slice."""
    n = 2
    flat = [
        1.0,
        0.5,
        20.0,
        1.2,
        0.6,
        40.0,
        1.0,
        0.5,
        25.0,
        1.2,
        0.6,
        45.0,
    ]
    params = torch.tensor(flat, dtype=torch.float64)
    s = torch.tensor([0.0, 50.0, 100.0], dtype=torch.float64)
    bid = effective_depth_bid_tensor(params, s, n)
    ask = effective_depth_ask_tensor(params, s, n)
    assert float(bid[-1]) > 0.0
    assert float(ask[-1]) > 0.0
    assert math.isclose(float(bid[-1]), float(ask[-1]), rel_tol=0.5)
