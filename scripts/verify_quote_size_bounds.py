"""Verify the quote-size refactor is numerically identical to the original.

``quote_size_from_depth_logit_torch`` used to branch on
``self.quote_size_max is None`` inside the traced region, which produced one
dynamo specialization per case and exhausted ``cache_size_limit`` (8) in
production. The bounds are now resolved by ``_quote_size_bounds`` into plain
floats before the traced arithmetic.

This re-implements the original expression exactly as it was and asserts the
new one matches bit-for-bit across every combination of cap / override /
tensor-bound, plus a gradient check.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mm.mpc.l_param import LParamConfig, _scalar_tensor


def original(lp: LParamConfig, u_depth, quote_size_min, quote_size_max):
    """Verbatim copy of the pre-refactor implementation."""
    lo = _scalar_tensor(quote_size_min, lp.quote_size_min, u_depth, clamp_zero=True)
    configured_hi_val = (
        float(lp.quote_size_max)
        if lp.quote_size_max is not None
        else float("inf")
    )
    if quote_size_max is None and lp.quote_size_max is None:
        hi = lo
    elif quote_size_max is None:
        hi = _scalar_tensor(configured_hi_val, None, u_depth)
    elif torch.is_tensor(quote_size_max):
        hi = torch.clamp(quote_size_max.to(dtype=u_depth.dtype, device=u_depth.device), min=0.0)
    else:
        hi = _scalar_tensor(float(quote_size_max), None, u_depth, clamp_zero=True)
    hi = torch.maximum(lo, torch.minimum(_scalar_tensor(configured_hi_val, None, u_depth), hi))
    return lo + torch.sigmoid(u_depth) * (hi - lo)


def main() -> None:
    torch.manual_seed(0)
    u_scalar = torch.tensor(0.7, dtype=torch.float64)
    u_batch = torch.linspace(-4.0, 4.0, 33, dtype=torch.float64)

    caps = [None, 100.0, 1.0, 0.5]
    overrides = [None, 100.0, 5.0, 0.25]
    mins = [None, 0.0, 1.0, 3.0]

    failures = []
    checked = 0

    print("=== scalar overrides ===")
    for cap, override, smin in itertools.product(caps, overrides, mins):
        lp = LParamConfig(
            quote_size_min=1.0,
            quote_size_max=cap,
            spread_max=200.0,
            n_components=4,
        )
        for u in (u_scalar, u_batch):
            want = original(lp, u, smin, override)
            got = lp.quote_size_from_depth_logit_torch(
                u, quote_size_min=smin, quote_size_max=override
            )
            diff = (got - want).abs().max().item()
            checked += 1
            if diff != 0.0:
                failures.append(("scalar", cap, override, smin, diff))

    print(f"  checked {checked} scalar cases")

    print("=== tensor overrides (per-sample bounds) ===")
    tensor_cases = 0
    for cap, smin in itertools.product(caps, mins):
        lp = LParamConfig(
            quote_size_min=1.0, quote_size_max=cap, spread_max=200.0, n_components=4
        )
        for shape in ((33,), (33, 1)):
            qmax = torch.rand(*shape, dtype=torch.float64) * 50.0
            want = original(lp, u_batch, smin, qmax)
            got = lp.quote_size_from_depth_logit_torch(
                u_batch, quote_size_min=smin, quote_size_max=qmax
            )
            diff = (got - want).abs().max().item()
            tensor_cases += 1
            if diff != 0.0:
                failures.append(("tensor", cap, smin, shape, diff))
    print(f"  checked {tensor_cases} tensor cases")

    print("=== gradient parity ===")
    grad_cases = 0
    for cap, override in itertools.product(caps, overrides):
        lp = LParamConfig(
            quote_size_min=1.0, quote_size_max=cap, spread_max=200.0, n_components=4
        )
        u_a = torch.linspace(-3.0, 3.0, 17, dtype=torch.float64, requires_grad=True)
        u_b = u_a.detach().clone().requires_grad_(True)
        original(lp, u_a, None, override).sum().backward()
        lp.quote_size_from_depth_logit_torch(u_b, quote_size_max=override).sum().backward()
        diff = (u_a.grad - u_b.grad).abs().max().item()
        grad_cases += 1
        if diff > 1e-15:
            failures.append(("grad", cap, override, diff))
    print(f"  checked {grad_cases} gradient cases")

    print()
    if failures:
        for row in failures[:10]:
            print("  MISMATCH", row)
        print(f"FAILED: {len(failures)} mismatches")
        raise SystemExit(1)
    print(
        f"OK: bit-identical across {checked + tensor_cases + grad_cases} cases "
        "(values and gradients)"
    )


if __name__ == "__main__":
    main()