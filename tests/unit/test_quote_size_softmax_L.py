from __future__ import annotations

import math

import numpy as np
import torch

from mm.mpc.l_param import (
    LParamConfig,
    effective_qrange_from_flat_np,
    positive_L_numpy,
    sigmoid_action_from_flat_raw,
)
from mm.mpc.mpc_tensor_batched import _initial_action_uniform_marginal_qrange
from mm.mpc.state_tensor import effective_depth_bid_tensor


def test_softmax_L_sums_to_target_qrange() -> None:
    spread_max = 100.0
    q_target = 3.0
    n = 4
    flat = _initial_action_uniform_marginal_qrange(
        n_components=n,
        spread_max=spread_max,
        quote_size_target_per_side=q_target,
        use_softmax_L=True,
    )
    assert flat is not None
    l_param = LParamConfig(
        use_softmax=True,
        quote_size_target=q_target,
        spread_max=spread_max,
    )
    q_bid = effective_qrange_from_flat_np(
        flat, n, side_off=0, spread_max=spread_max, l_param=l_param
    )
    q_ask = effective_qrange_from_flat_np(
        flat, n, side_off=3 * n, spread_max=spread_max, l_param=l_param
    )
    assert math.isclose(q_bid, q_target, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(q_ask, q_target, rel_tol=1e-5, abs_tol=1e-5)


def test_softmax_L_torch_matches_numpy_at_spread_max() -> None:
    spread_max = 80.0
    q_target = 2.5
    n = 3
    flat = _initial_action_uniform_marginal_qrange(
        n, spread_max, q_target, use_softmax_L=True
    )
    assert flat is not None
    l_param = LParamConfig(
        use_softmax=True, quote_size_target=q_target, spread_max=spread_max
    )
    params = torch.tensor(flat[: 3 * n], dtype=torch.float64)
    s = torch.tensor([0.0, spread_max], dtype=torch.float64)
    q_torch = float(
        (
            effective_depth_bid_tensor(params, s, n, l_param=l_param)[1]
            - effective_depth_bid_tensor(params, s, n, l_param=l_param)[0]
        ).item()
    )
    q_np = effective_qrange_from_flat_np(
        flat, n, side_off=0, spread_max=spread_max, l_param=l_param
    )
    assert math.isclose(q_torch, q_np, rel_tol=1e-6)
    assert math.isclose(q_torch, q_target, rel_tol=1e-5)


def test_sigmoid_action_from_flat_raw_uses_effective_L() -> None:
    spread_max = 50.0
    q_target = 4.0
    flat = _initial_action_uniform_marginal_qrange(2, spread_max, q_target, use_softmax_L=True)
    assert flat is not None
    l_param = LParamConfig(
        use_softmax=True, quote_size_target=q_target, spread_max=spread_max
    )
    action = sigmoid_action_from_flat_raw(flat, 2, l_param=l_param)
    assert action.depth_bid(spread_max) == action.effective_depth_bid(spread_max)
    assert math.isclose(action.effective_depth_bid(spread_max), q_target, rel_tol=1e-5)
