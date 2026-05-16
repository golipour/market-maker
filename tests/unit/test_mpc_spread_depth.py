from __future__ import annotations

from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched


def test_spread_depth_stored() -> None:
    opt = object()
    flow = object()
    exe = object()
    mpc = MonteCarloMPCTensorBatched(opt, flow, exe, spread_depth=12.5)
    assert mpc.spread_depth == 12.5


def test_hedge_enabled_explicit_overrides_hedge_coef() -> None:
    opt = object()
    flow = object()
    exe = object()
    on = MonteCarloMPCTensorBatched(opt, flow, exe, hedge_enabled=True, hedge_coef=0.0)
    assert on.hedge_enabled is True
    off = MonteCarloMPCTensorBatched(opt, flow, exe, hedge_enabled=False, hedge_coef=1.0)
    assert off.hedge_enabled is False


def test_hedge_enabled_defaults_from_hedge_coef() -> None:
    opt = object()
    flow = object()
    exe = object()
    assert MonteCarloMPCTensorBatched(opt, flow, exe, hedge_coef=0.0).hedge_enabled is False
    assert MonteCarloMPCTensorBatched(opt, flow, exe, hedge_coef=0.01).hedge_enabled is True
