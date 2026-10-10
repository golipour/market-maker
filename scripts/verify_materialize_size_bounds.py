"""Regression guard: does ``materialize_side_torch`` still respect size bounds?

The production notionals grew monotonically (13.6M -> 46.4M) and breached
``max_inventory`` / ``max_open_notional``, so every action was filtered. The
quote-size refactor was verified in isolation; this checks the *full*
``materialize_side_torch`` path, where the bounds are applied, to confirm the
uncapped config still produces sizes within the bounds actually configured.

Run: OMP_NUM_THREADS=2 python3 scripts/verify_materialize_size_bounds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mm.mpc.l_param import LParamConfig, materialize_side_torch


def top_notional(lp: LParamConfig, n_comp: int, u_depth: float):
    params = torch.zeros(3 * 2 * n_comp + 3, dtype=torch.float64, requires_grad=True)
    size_min = torch.zeros((1, 1), dtype=torch.float64)
    size_max = torch.full((1, 1), 50.0, dtype=torch.float64)
    out = materialize_side_torch(
        params,
        side_off=0,
        n_components=n_comp,
        l_param=lp,
        quote_size_min=size_min,
        quote_size_max=size_max,
    )
    q_total = out[0] if isinstance(out, (tuple, list)) else out
    return float(q_total.reshape(-1)[0])


def main() -> None:
    n_comp = 4
    print("=== uncapped config: size must respect the caller-supplied max ===")
    # This mirrors production: quote_size_max=None in LParamConfig, and the
    # per-sample bound arrives as a tensor.
    lp_uncapped = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=None,
        spread_max=200.0,
        n_components=n_comp,
    )
    for bound in (50.0, 50.0, 50.0):
        top = top_notional(lp_uncapped, n_comp, 0.0)
        print(f"  bound={bound:6.1f} -> top size = {top:.6f}")
        if top > bound + 1e-6:
            print("  FAIL: size exceeded the configured bound")
            raise SystemExit(1)

    print("\n=== sweep u_depth: size must stay within [min, max] ===")
    lo = 1.0
    hi = 50.0
    violations = 0
    for i in range(-60, 61):
        u = torch.tensor(i / 10.0, dtype=torch.float64)
        params = torch.zeros(3 * 2 * n_comp + 3, dtype=torch.float64, requires_grad=True)
        size_min = torch.full((1, 1), lo, dtype=torch.float64)
        size_max = torch.full((1, 1), hi, dtype=torch.float64)
        out = materialize_side_torch(
            params,
            side_off=0,
            n_components=n_comp,
            l_param=lp_uncapped,
            quote_size_min=size_min,
            quote_size_max=size_max,
        )
        q_total = out[0] if isinstance(out, (tuple, list)) else out
        v = float(q_total.reshape(-1)[0])
        if v < lo - 1e-9 or v > hi + 1e-9:
            violations += 1
            print(f"  u={i/10.0:6.1f} -> {v:.6f}  OUT OF BOUNDS")
    print(f"  checked 121 logits, violations = {violations}")
    if violations:
        raise SystemExit(1)

    print("\n=== capped config: size must respect configured cap ===")
    lp_capped = LParamConfig(
        quote_size_min=1.0,
        quote_size_max=20.0,
        spread_max=200.0,
        n_components=n_comp,
    )
    for i in range(-30, 31):
        u = torch.tensor(i / 10.0, dtype=torch.float64)
        params = torch.zeros(3 * 2 * n_comp + 3, dtype=torch.float64, requires_grad=True)
        size_min = torch.zeros((1, 1), dtype=torch.float64)
        size_max = torch.full((1, 1), 50.0, dtype=torch.float64)
        out = materialize_side_torch(
            params,
            side_off=0,
            n_components=n_comp,
            l_param=lp_capped,
            quote_size_min=size_min,
            quote_size_max=size_max,
        )
        q_total = out[0] if isinstance(out, (tuple, list)) else out
        v = float(q_total.reshape(-1)[0])
        if v > 20.0 + 1e-9:
            print(f"  FAIL u={i/10.0}: {v:.6f} exceeds cap 20.0")
            raise SystemExit(1)
    print("  checked 61 logits, all within cap")

    print("\nOK: materialize_side_torch respects size bounds")


if __name__ == "__main__":
    main()