from __future__ import annotations

import numpy as np
import pytest
import torch

from mm.mpc.l_param import (
    LParamConfig,
    action_param_len,
    reservation_logit_index,
    reservation_r_offset_from_flat_np,
    sigmoid_action_from_flat_raw,
)
from mm.mpc.mpc_tensor_batched import VenueOptimizerParams
from mm.mpc.state_tensor import effective_depth_bid_tensor, effective_depth_ask_tensor


def test_action_param_len_includes_reservation_logit() -> None:
    assert action_param_len(4) == 6 * 4 + 3
    assert reservation_logit_index(4) == action_param_len(4) - 1


def test_reservation_offset_parses_from_config() -> None:
    params = VenueOptimizerParams.from_mapping({"reservation_offset": {"r_max": 0.0015}})
    assert params.reservation_offset is not None
    assert params.reservation_offset.r_max == pytest.approx(0.0015)
    l_param = params.l_param_config(spread_max=200.0, n_components=2)
    assert l_param.reservation_r_max == pytest.approx(0.0015)


def test_reservation_shift_moves_effective_depth_on_market_grid() -> None:
    n = 2
    spread_max = 100.0
    mid = 1000.0
    r_max = 0.002
    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=10.0,
        spread_max=spread_max,
        n_components=n,
        quote_spread_max=50.0,
        quote_spread_min_depth=0.5,
        reservation_r_max=r_max,
    )
    flat = np.zeros(action_param_len(n), dtype=np.float64)
    flat[reservation_logit_index(n)] = 10.0
    r_offset = reservation_r_offset_from_flat_np(flat, n_components=n, r_max=r_max)
    assert r_offset == pytest.approx(r_max * np.tanh(10.0))

    spread = torch.tensor([5.0, 20.0, 40.0], dtype=torch.float64)
    mid_t = torch.tensor(mid, dtype=torch.float64)
    r_t = torch.tensor(r_offset, dtype=torch.float64)
    depth_shifted = effective_depth_bid_tensor(
        torch.tensor(flat, dtype=torch.float64),
        spread,
        n,
        l_param=l_param,
        mid_price=mid_t,
        reservation_r_offset=r_t,
    )
    flat_zero_r = flat.copy()
    flat_zero_r[reservation_logit_index(n)] = 0.0
    depth_neutral = effective_depth_bid_tensor(
        torch.tensor(flat_zero_r, dtype=torch.float64),
        spread,
        n,
        l_param=l_param,
        mid_price=mid_t,
        reservation_r_offset=torch.tensor(0.0, dtype=torch.float64),
    )
    assert torch.all(depth_shifted >= depth_neutral - 1e-9)
    assert float(depth_shifted[0]) > float(depth_neutral[0]) + 1e-6


def test_effective_action_applies_reservation_shift_at_execution() -> None:
    n = 1
    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=5.0,
        spread_max=50.0,
        n_components=n,
        reservation_r_max=0.001,
    )
    flat = np.zeros(action_param_len(n), dtype=np.float64)
    flat[reservation_logit_index(n)] = 5.0
    mid = 500.0
    action = sigmoid_action_from_flat_raw(flat, n, l_param=l_param, mid_price=mid)
    assert action.reservation_mid == pytest.approx(mid)
    assert action.reservation_r_offset > 0.0
    at_touch = action.effective_depth_bid(1.0)
    neutral = sigmoid_action_from_flat_raw(
        np.zeros(action_param_len(n)),
        n,
        l_param=l_param,
        mid_price=mid,
    ).effective_depth_bid(1.0)
    assert at_touch > neutral


def test_shifted_effective_depth_stays_monotone_on_grid() -> None:
    n = 3
    spread_max = 80.0
    mid = 250.0
    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=20.0,
        spread_max=spread_max,
        n_components=n,
        quote_spread_max=40.0,
        quote_spread_min_depth=0.2,
        reservation_r_max=0.001,
    )
    flat = np.zeros(action_param_len(n), dtype=np.float64)
    flat[reservation_logit_index(n)] = 2.5
    spread = torch.linspace(1.0, spread_max, steps=40, dtype=torch.float64)
    mid_t = torch.tensor(mid, dtype=torch.float64)
    r_t = torch.tensor(
        reservation_r_offset_from_flat_np(flat, n_components=n, r_max=l_param.reservation_r_max),
        dtype=torch.float64,
    )
    params = torch.tensor(flat, dtype=torch.float64)
    bid = effective_depth_bid_tensor(
        params,
        spread,
        n,
        l_param=l_param,
        mid_price=mid_t,
        reservation_r_offset=r_t,
    )
    ask = effective_depth_ask_tensor(
        params,
        spread,
        n,
        l_param=l_param,
        mid_price=mid_t,
        reservation_r_offset=r_t,
    )
    assert torch.all(bid[1:] + 1e-9 >= bid[:-1])
    assert torch.all(ask[1:] + 1e-9 >= ask[:-1])
