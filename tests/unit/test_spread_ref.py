from __future__ import annotations

import math

import numpy as np
import torch

from mm.mpc.l_param import sigmoid_action_from_flat_raw
from mm.mpc.mpc_tensor_batched import _initial_action_uniform_marginal_qrange
from mm.mpc.spread_ref import (
    SpreadRefConfig,
    decode_first_centers_np,
    effective_depth_at_first_center_np,
)
from mm.mpc.state_tensor import effective_depth_bid_tensor


def test_spread_ref_centers_sum_at_most_S() -> None:
    cfg = SpreadRefConfig(spread_ref_quantity=1.0, spread_max_at_ref=40.0)
    for u_skew, u_tight in [(0.0, 0.0), (2.0, -1.0), (-3.0, 4.0)]:
        x0_bid, x0_ask = decode_first_centers_np(u_skew, u_tight, cfg)
        assert x0_bid >= 0.0
        assert x0_ask >= 0.0
        assert x0_bid + x0_ask <= cfg.spread_max_at_ref + 1e-9


def test_spread_ref_effective_depth_at_center_at_least_Q() -> None:
    cfg = SpreadRefConfig(spread_ref_quantity=2.0, spread_max_at_ref=50.0)
    flat = _initial_action_uniform_marginal_qrange(
        2,
        50.0,
        10.0,
        use_softmax_L=True,
        spread_ref=cfg,
    )
    assert flat is not None
    action = sigmoid_action_from_flat_raw(flat, 2, l_param=None, spread_ref=cfg)
    d_bid = action.effective_depth_bid(action.bid_x0[0])
    d_ask = action.effective_depth_ask(action.ask_x0[0])
    assert d_bid >= cfg.spread_ref_quantity - 1e-5
    assert d_ask >= cfg.spread_ref_quantity - 1e-5
    assert action.bid_x0[0] + action.ask_x0[0] <= cfg.spread_max_at_ref + 1e-5


def test_warm_start_rejects_legacy_x0_flat_when_spread_ref_enabled() -> None:
    """spread_ref comp-0 slots must be logits; legacy warm start stored decoded x0 (~500)."""
    from mm.mpc.mpc_state import State, OrderbookRelative
    from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched

    class P:
        def sample_residual(self, shape, device=None, dtype=None):
            import torch

            return torch.zeros(shape, dtype=dtype or torch.float64, device=device), torch.zeros(
                shape, dtype=dtype or torch.float64, device=device
            )

        _target_depth_lots = 100.0

    class F:
        def sample(self, shape, device=None, dtype=None):
            import torch

            z = torch.ones(shape, dtype=dtype or torch.float64, device=device) * 0.2
            return z, z

    class E:
        maker_fee = 0.0

    opt = MonteCarloMPCTensorBatched(
        P(),
        F(),
        E(),
        n_components=2,
        n_samples=4,
        horizon=1,
        adam_epochs=15,
        spread_max=1000.0,
        spread_step=10.0,
        device="cpu",
        warm_start=True,
        quote_size_target_per_side=3.0,
        quote_size_softmax_L=True,
        spread_ref_constrain=False,
    )
    st = State(
        mid=100000.0,
        ob_rel=OrderbookRelative(bids=((10.0, 1.0),), asks=((10.0, 1.0),)),
        buy_vol=1.0,
        sell_vol=1.0,
        inventory=0.0,
        cash=1e9,
        buy_vwap=99900.0,
        sell_vwap=100100.0,
    )
    opt.optimize(st)
    assert opt._last_raw_flat is not None
    legacy = np.array(opt._last_raw_flat, dtype=np.float64)
    legacy[2] = 490.0
    legacy[8] = 510.0

    opt.spread_ref = SpreadRefConfig(spread_ref_quantity=1.0, spread_max_at_ref=50.0)
    opt._last_raw_flat = legacy
    action = opt.optimize(st)
    assert action.effective_depth_ask(1000.0) > 0.5
    assert opt._raw_flat_valid_for_warm_start(opt._last_raw_flat, 2)


def test_spread_ref_torch_matches_numpy_centers() -> None:
    cfg = SpreadRefConfig(spread_ref_quantity=1.5, spread_max_at_ref=30.0)
    flat = _initial_action_uniform_marginal_qrange(1, 30.0, 5.0, spread_ref=cfg)
    assert flat is not None
    params = torch.tensor(flat, dtype=torch.float64)
    x0_bid_np, x0_ask_np = decode_first_centers_np(0.0, 0.0, cfg)
    action = sigmoid_action_from_flat_raw(flat, 1, l_param=None, spread_ref=cfg)
    assert math.isclose(action.bid_x0[0], x0_bid_np, rel_tol=1e-9)
    assert math.isclose(action.ask_x0[0], x0_ask_np, rel_tol=1e-9)
    s = torch.tensor([action.bid_x0[0]], dtype=torch.float64)
    d_torch = float(
        effective_depth_bid_tensor(
            params[:3],
            s,
            1,
            spread_ref=cfg,
            full_params=params,
        )[0]
    )
    d_np = effective_depth_at_first_center_np(
        np.array([action.bid_L[0]]),
        np.array([action.bid_k[0]]),
        np.array([action.bid_x0[0]]),
    )
    assert math.isclose(d_torch, d_np, rel_tol=1e-6)
