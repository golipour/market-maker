"""Regression tests for the memoized scalar bound tensors in ``l_param``.

The depth decoder runs ~80x per optimizer decide, so the quote-size bound
tensors are cached by ``(value, dtype, device, clamped)``. These tests pin
down that the cache is transparent: identical results to an uncached build,
correct invalidation when a bound/dtype/device changes, and no interference
with autograd when a bound is itself a differentiable tensor.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mm.mpc.l_param import LParamConfig, _scalar_tensor


def _uncached(value, default, like, clamp_zero):
    """Reference implementation matching the pre-cache inline logic."""
    if torch.is_tensor(value):
        out = value.to(dtype=like.dtype, device=like.device)
        return torch.clamp(out, min=0.0) if clamp_zero else out
    if value is None:
        value = default
    out = torch.tensor(float(value), dtype=like.dtype, device=like.device)
    return torch.clamp(out, min=0.0) if clamp_zero else out


@pytest.mark.parametrize("clamp_zero", [False, True])
def test_scalar_tensor_matches_uncached_for_scalars(clamp_zero):
    like = torch.zeros(4, dtype=torch.float64)
    for value in (0.0, 1.0, 2.5, 1e6, -7.0, float("inf")):
        got = _scalar_tensor(value, None, like, clamp_zero=clamp_zero)
        want = _uncached(value, None, like, clamp_zero=clamp_zero)
        assert torch.equal(got, want)


def test_scalar_tensor_falls_back_to_default():
    like = torch.zeros(4, dtype=torch.float64)
    got = _scalar_tensor(None, 3.5, like, clamp_zero=True)
    want = _uncached(None, 3.5, like, clamp_zero=True)
    assert torch.equal(got, want)


def test_scalar_tensor_honours_dtype():
    for dtype in (torch.float32, torch.float64):
        like = torch.zeros(4, dtype=dtype)
        got = _scalar_tensor(7.25, None, like, clamp_zero=False)
        assert got.dtype == dtype
        assert float(got) == pytest.approx(7.25)


def test_scalar_tensor_caches_and_invalidates_on_value():
    like = torch.zeros(4, dtype=torch.float64)
    first = _scalar_tensor(2.0, None, like, clamp_zero=False)
    second = _scalar_tensor(2.0, None, like, clamp_zero=False)
    assert first is second, "identical scalar should hit the cache"
    third = _scalar_tensor(9.0, None, like, clamp_zero=False)
    assert third is not first
    assert float(third) == pytest.approx(9.0)


def test_scalar_tensor_invalidates_on_clamp_flag():
    like = torch.zeros(4, dtype=torch.float64)
    raw = _scalar_tensor(-4.0, None, like, clamp_zero=False)
    clamped = _scalar_tensor(-4.0, None, like, clamp_zero=True)
    assert raw is not clamped
    assert float(raw) == pytest.approx(-4.0)
    assert float(clamped) == pytest.approx(0.0)


def test_scalar_tensor_tensor_input_is_not_cached_and_keeps_grad():
    like = torch.zeros(4, dtype=torch.float64)
    src = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    got = _scalar_tensor(src, None, like, clamp_zero=True)
    assert got.requires_grad
    got.backward()
    assert float(src.grad) == pytest.approx(1.0)
    other = _scalar_tensor(src.detach() + 1.0, None, like, clamp_zero=True)
    assert float(other) == pytest.approx(6.0)


def _reference_decoder(cfg, u_depth, mn, mx):
    lo = _uncached(mn, cfg.quote_size_min, u_depth, True)
    configured_hi_val = (
        float(cfg.quote_size_max) if cfg.quote_size_max is not None else float("inf")
    )
    configured_hi = _uncached(configured_hi_val, None, u_depth, False)
    if mx is None and cfg.quote_size_max is None:
        hi = lo
    elif mx is None:
        hi = configured_hi
    elif torch.is_tensor(mx):
        hi = torch.clamp(mx.to(dtype=u_depth.dtype, device=u_depth.device), min=0.0)
    else:
        hi = _uncached(float(mx), None, u_depth, True)
    hi = torch.maximum(lo, torch.minimum(configured_hi, hi))
    return lo + torch.sigmoid(u_depth) * (hi - lo)


@pytest.mark.parametrize(
    "qmin,qmax",
    [(1.0, 100000.0), (0.0, None), (5.0, 50.0), (-3.0, 7.0)],
)
def test_quote_size_from_depth_logit_torch_is_cache_transparent(qmin, qmax):
    """The decoder must return exactly what an uncached build returns."""
    cfg = LParamConfig(quote_size_min=qmin, quote_size_max=qmax, spread_max=200.0)
    gen = torch.Generator().manual_seed(7)
    for mn in (None, 0.0, 2.5, 1e6, -7.0):
        for mx in (None, 0.0, 3.5, 1e9, -2.0):
            u = torch.randn(64, generator=gen, dtype=torch.float64)
            got = cfg.quote_size_from_depth_logit_torch(
                u, quote_size_min=mn, quote_size_max=mx
            )
            want = _reference_decoder(cfg, u, mn, mx)
            assert torch.equal(got, want), (mn, mx)


def test_quote_size_from_depth_logit_torch_grad_through_tensor_bound():
    """A tensor bound must stay differentiable despite the scalar cache."""
    cfg = LParamConfig(quote_size_min=1.0, quote_size_max=1000.0, spread_max=200.0)
    u = torch.randn(8, dtype=torch.float64, requires_grad=True)
    lo = torch.full((8, 1), 4.0, dtype=torch.float64, requires_grad=True)
    out = cfg.quote_size_from_depth_logit_torch(u, quote_size_min=lo)
    out.sum().backward()
    assert u.grad is not None
    assert lo.grad is not None
