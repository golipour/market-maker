from __future__ import annotations

import math

import numpy as np
import torch

from mm.mpc.l_param import (
    LParamConfig,
    effective_depth_at_center_from_flat_np,
    effective_qrange_from_flat_np,
    materialize_side_np,
    sigmoid_action_from_flat_raw,
    validate_l_param_config,
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
        n_components=n,
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
    params = torch.tensor(flat[: 6 * n], dtype=torch.float64)
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


def test_spread_structured_centers_bounded_by_max() -> None:
    spread_max = 100.0
    s_max = 20.0
    q_total = 3.0
    q_spread = 0.5
    n = 4
    l_param = LParamConfig(
        use_softmax=True,
        quote_size_target=q_total,
        spread_max=spread_max,
        n_components=n,
        quote_spread_target=s_max,
        quote_spread_ref_depth=q_spread,
    )
    validate_l_param_config(l_param)
    flat = _initial_action_uniform_marginal_qrange(
        n, spread_max, q_total, use_softmax_L=True, l_param=l_param
    )
    assert flat is not None
    _, _, bid_x0 = materialize_side_np(flat, side_off=0, n_components=n, l_param=l_param)
    _, _, ask_x0 = materialize_side_np(flat, side_off=3 * n, n_components=n, l_param=l_param)
    assert bid_x0[0] + ask_x0[0] <= s_max + 1e-5
    assert bid_x0[0] + ask_x0[0] > 0.0
    assert 0.0 <= bid_x0[0] <= spread_max
    assert 0.0 <= ask_x0[0] <= spread_max

    flat_tight = flat.copy()
    flat_tight[0] = -6.0
    _, _, bid_t = materialize_side_np(flat_tight, side_off=0, n_components=n, l_param=l_param)
    _, _, ask_t = materialize_side_np(flat_tight, side_off=3 * n, n_components=n, l_param=l_param)
    assert bid_t[0] + ask_t[0] < bid_x0[0] + ask_x0[0]


def test_spread_structured_depth_at_center_and_spread_max() -> None:
    spread_max = 100.0
    s_target = 24.0
    q_total = 3.0
    q_spread = 0.5
    n = 4
    l_param = LParamConfig(
        use_softmax=True,
        quote_size_target=q_total,
        spread_max=spread_max,
        n_components=n,
        quote_spread_target=s_target,
        quote_spread_ref_depth=q_spread,
    )
    flat = _initial_action_uniform_marginal_qrange(
        n, spread_max, q_total, use_softmax_L=True, l_param=l_param
    )
    assert flat is not None
    for side_off in (0, 3 * n):
        q_center = effective_depth_at_center_from_flat_np(
            flat, n, side_off=side_off, l_param=l_param
        )
        q_max = effective_qrange_from_flat_np(
            flat, n, side_off=side_off, spread_max=spread_max, l_param=l_param
        )
        assert q_center >= q_spread - 1e-5
        assert math.isclose(q_max, q_total, rel_tol=1e-4, abs_tol=1e-4)


def test_spread_structured_rejects_n_components_one() -> None:
    l_param = LParamConfig(
        use_softmax=True,
        quote_size_target=3.0,
        spread_max=100.0,
        n_components=1,
        quote_spread_target=10.0,
        quote_spread_ref_depth=0.5,
    )
    try:
        validate_l_param_config(l_param)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "n_components" in str(exc)
