"""Tests for effective_depth_stacked_tensor and soft-fill compilation."""

from __future__ import annotations

import os
import pytest

torch = pytest.importorskip("torch")

from mm.mpc.l_param import LParamConfig
from mm.mpc.state_tensor import (
    effective_depth_ask_tensor,
    effective_depth_bid_tensor,
    effective_depth_stacked_tensor,
)
from mm.mpc.mpc_tensor_batched import (
    MonteCarloMPCTensorBatched,
    soft_fill_batched,
    _get_compiled_soft_fill,
)


@pytest.mark.parametrize("n_components", [1, 2, 4])
@pytest.mark.parametrize("has_reservation", [False, True])
def test_effective_depth_stacked_matches_separate(n_components: int, has_reservation: bool) -> None:
    gen = torch.Generator().manual_seed(42)
    spread_max = 200.0
    grid_levels = 10
    n_samples = 8

    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=1000.0,
        spread_max=spread_max,
        n_components=n_components,
        reservation_r_max=0.05 if has_reservation else 0.0,
    )
    param_len = 6 * n_components + 3
    params = torch.randn(param_len, dtype=torch.float64, generator=gen, requires_grad=True)
    grid = torch.arange(1, grid_levels + 1, dtype=torch.float64) * 20.0
    mid = torch.full((n_samples, 1), 100.0, dtype=torch.float64)
    r_offset = torch.tensor(0.02, dtype=torch.float64) if has_reservation else None

    q_min_b = torch.full((n_samples, 1), 1.0, dtype=torch.float64)
    q_max_b = torch.full((n_samples, 1), 50.0, dtype=torch.float64)
    q_min_a = torch.full((n_samples, 1), 1.0, dtype=torch.float64)
    q_max_a = torch.full((n_samples, 1), 50.0, dtype=torch.float64)

    # Separate evaluations
    db = effective_depth_bid_tensor(
        params,
        grid,
        n_components,
        l_param=l_param,
        quote_size_min=q_min_b,
        quote_size_max=q_max_b,
        mid_price=mid,
        reservation_r_offset=r_offset,
    )
    da = effective_depth_ask_tensor(
        params,
        grid,
        n_components,
        l_param=l_param,
        quote_size_min=q_min_a,
        quote_size_max=q_max_a,
        mid_price=mid,
        reservation_r_offset=r_offset,
    )
    expected = torch.stack([db, da], dim=0)

    # Stacked evaluation
    stacked = effective_depth_stacked_tensor(
        params,
        grid,
        n_components,
        l_param=l_param,
        bid_quote_size_min=q_min_b,
        bid_quote_size_max=q_max_b,
        ask_quote_size_min=q_min_a,
        ask_quote_size_max=q_max_a,
        mid_price=mid,
        reservation_r_offset=r_offset,
    )

    assert torch.equal(expected, stacked)

    # Test backward pass equivalence
    expected.sum().backward()
    grad_expected = params.grad.clone()
    params.grad.zero_()

    stacked.sum().backward()
    grad_stacked = params.grad.clone()

    assert torch.allclose(grad_expected, grad_stacked, atol=1e-12, rtol=1e-12)


def test_effective_depth_stacked_without_sample_dim() -> None:
    spread_max = 100.0
    n = 2
    l_param = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=100.0,
        spread_max=spread_max,
        n_components=n,
    )
    params = torch.randn(6 * n + 3, dtype=torch.float64)
    grid = torch.arange(1, 6, dtype=torch.float64) * 20.0

    db = effective_depth_bid_tensor(params, grid, n, l_param=l_param)
    da = effective_depth_ask_tensor(params, grid, n, l_param=l_param)
    expected = torch.stack([db, da], dim=0)

    stacked = effective_depth_stacked_tensor(params, grid, n, l_param=l_param)
    assert torch.equal(expected, stacked)


def test_soft_fill_compile_cache_and_fallback() -> None:
    # Verify _get_compiled_soft_fill returns callable
    compiled = _get_compiled_soft_fill(torch)
    assert callable(compiled)

    # Verify fallback handling when compile fails or raises
    class BrokenTorch:
        @staticmethod
        def compile(*args, **kwargs):
            raise RuntimeError("forced compile error")

    import mm.mpc.mpc_tensor_batched as M
    orig_kernel = M._COMPILED_SOFT_FILL_KERNEL
    orig_failed = M._COMPILE_SOFT_FILL_FAILED
    try:
        M._COMPILED_SOFT_FILL_KERNEL = None
        M._COMPILE_SOFT_FILL_FAILED = False
        res = M._get_compiled_soft_fill(BrokenTorch)
        assert res is soft_fill_batched
        assert M._COMPILE_SOFT_FILL_FAILED is True
    finally:
        M._COMPILED_SOFT_FILL_KERNEL = orig_kernel
        M._COMPILE_SOFT_FILL_FAILED = orig_failed
