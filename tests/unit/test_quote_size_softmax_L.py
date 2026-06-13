from __future__ import annotations

import math

import numpy as np
import torch

from mm.mpc.l_param import (
    LParamConfig,
    action_param_len,
    depth_logit_index,
    effective_depth_at_center_from_flat_np,
    effective_qrange_from_flat_np,
    materialize_side_np,
    sigmoid_action_from_flat_raw,
    validate_l_param_config,
)
from mm.mpc.mpc_tensor_batched import _initial_action_uniform_marginal_qrange
from mm.mpc.state_tensor import effective_depth_bid_tensor


def _l_param(
    *,
    q_min: float = 1.0,
    q_max: float = 10.0,
    spread_max: float = 100.0,
    n_components: int = 4,
    quote_spread_max: float | None = None,
    quote_spread_min_depth: float | None = None,
) -> LParamConfig:
    return LParamConfig(
        quote_size_min=q_min,
        quote_size_max=q_max,
        spread_max=spread_max,
        n_components=n_components,
        quote_spread_max=quote_spread_max,
        quote_spread_min_depth=quote_spread_min_depth,
    )


def test_depth_logit_zero_targets_midpoint_of_range() -> None:
    spread_max = 100.0
    q_min, q_max = 1.0, 10.0
    q_mid = 0.5 * (q_min + q_max)
    n = 4
    l_param = _l_param(q_min=q_min, q_max=q_max, spread_max=spread_max, n_components=n)
    flat = _initial_action_uniform_marginal_qrange(
        n_components=n,
        spread_max=spread_max,
        l_param=l_param,
    )
    q_bid = effective_qrange_from_flat_np(
        flat, n, side_off=0, spread_max=spread_max, l_param=l_param
    )
    q_ask = effective_qrange_from_flat_np(
        flat, n, side_off=3 * n, spread_max=spread_max, l_param=l_param
    )
    assert q_min <= q_bid <= q_max
    assert q_min <= q_ask <= q_max
    assert math.isclose(q_bid, q_mid, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(q_ask, q_mid, rel_tol=1e-5, abs_tol=1e-5)


def test_depth_logit_torch_matches_numpy_at_spread_max() -> None:
    spread_max = 80.0
    q_min, q_max = 2.0, 8.0
    n = 3
    l_param = _l_param(q_min=q_min, q_max=q_max, spread_max=spread_max, n_components=n)
    flat = _initial_action_uniform_marginal_qrange(
        n, spread_max, l_param=l_param
    )
    params = torch.tensor(flat[: action_param_len(n)], dtype=torch.float64)
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
    assert q_min <= q_torch <= q_max


def test_sigmoid_action_from_flat_raw_uses_effective_L() -> None:
    spread_max = 50.0
    q_min, q_max = 2.0, 6.0
    l_param = _l_param(q_min=q_min, q_max=q_max, spread_max=spread_max, n_components=2)
    flat = _initial_action_uniform_marginal_qrange(2, spread_max, l_param=l_param)
    action = sigmoid_action_from_flat_raw(flat, 2, l_param=l_param)
    assert action.depth_bid(spread_max) == action.effective_depth_bid(spread_max)
    depth = action.effective_depth_bid(spread_max)
    assert q_min <= depth <= q_max


def test_spread_structured_centers_bounded_by_max() -> None:
    spread_max = 100.0
    s_max = 20.0
    q_min, q_max = 1.0, 10.0
    q_spread = 0.5
    n = 4
    l_param = _l_param(
        q_min=q_min,
        q_max=q_max,
        spread_max=spread_max,
        n_components=n,
        quote_spread_max=s_max,
        quote_spread_min_depth=q_spread,
    )
    validate_l_param_config(l_param)
    flat = _initial_action_uniform_marginal_qrange(
        n, spread_max, l_param=l_param
    )
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
    s_max = 24.0
    q_min, q_max = 1.0, 10.0
    q_spread = 0.5
    n = 4
    l_param = _l_param(
        q_min=q_min,
        q_max=q_max,
        spread_max=spread_max,
        n_components=n,
        quote_spread_max=s_max,
        quote_spread_min_depth=q_spread,
    )
    flat = _initial_action_uniform_marginal_qrange(
        n, spread_max, l_param=l_param
    )
    for side_off in (0, 3 * n):
        q_center = effective_depth_at_center_from_flat_np(
            flat, n, side_off=side_off, l_param=l_param
        )
        q_at_max = effective_qrange_from_flat_np(
            flat, n, side_off=side_off, spread_max=spread_max, l_param=l_param
        )
        assert q_center >= q_spread - 1e-5
        assert q_min <= q_at_max <= q_max


def test_spread_structured_depth_max_is_hard_cap_for_near_singular_solve() -> None:
    spread_max = 100.0
    q_min, q_max = 1.0, 10.0
    n = 4
    l_param = _l_param(
        q_min=q_min,
        q_max=q_max,
        spread_max=spread_max,
        n_components=n,
        quote_spread_max=100.0,
        quote_spread_min_depth=0.3,
    )
    flat = np.array(
        [
            13.21739818,
            -27.30868284,
            -19.61156255,
            -10.8076019,
            -2.00779819,
            4.16676122,
            3.72563968,
            2.57493972,
            3.96698046,
            -4.95146261,
            -13.27122907,
            1.08777245,
            -22.6883692,
            14.95628219,
            27.17614343,
            -26.75421075,
            16.93962847,
            -16.57002653,
            -9.8153187,
            -27.98770947,
            28.14514782,
            3.72573632,
            -25.10511407,
            -10.70666193,
            28.63686397,
            -26.39752455,
        ],
        dtype=np.float64,
    )

    action = sigmoid_action_from_flat_raw(
        flat,
        n,
        l_param=l_param,
        bid_quote_size_max=q_max,
        ask_quote_size_max=q_max,
    )

    assert action.effective_depth_bid(spread_max) <= q_max + 1e-9
    assert action.effective_depth_ask(spread_max) <= q_max + 1e-9


def test_spread_structured_rejects_n_components_one() -> None:
    l_param = _l_param(
        q_min=1.0,
        q_max=10.0,
        spread_max=100.0,
        n_components=1,
        quote_spread_max=10.0,
        quote_spread_min_depth=0.5,
    )
    try:
        validate_l_param_config(l_param)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "n_components" in str(exc)


def test_depth_logit_selects_bounds() -> None:
    spread_max = 50.0
    q_min, q_max = 2.0, 8.0
    n = 2
    l_param = _l_param(q_min=q_min, q_max=q_max, spread_max=spread_max, n_components=n)
    flat = _initial_action_uniform_marginal_qrange(n, spread_max, l_param=l_param)
    bid_depth_idx = depth_logit_index(side_off=0, n_components=n)
    flat_min = flat.copy()
    flat_min[bid_depth_idx] = -30.0
    flat_max = flat.copy()
    flat_max[bid_depth_idx] = 30.0
    q_lo = effective_qrange_from_flat_np(
        flat_min, n, side_off=0, spread_max=spread_max, l_param=l_param
    )
    q_hi = effective_qrange_from_flat_np(
        flat_max, n, side_off=0, spread_max=spread_max, l_param=l_param
    )
    assert math.isclose(q_lo, q_min, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(q_hi, q_max, rel_tol=1e-4, abs_tol=1e-4)
