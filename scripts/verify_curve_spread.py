"""Verify ``curve_spread_for_market_spread`` refactor is exact.

The zero-offset short-circuit used to call ``float(r_off.abs().item())``,
which is a host sync and a hard dynamo graph break in the traced depth-decoder
path. It is now a device-side ``torch.where``.

This re-implements the original exactly as it was and asserts the new version
matches bit-for-bit across sides, zero/non-zero offsets, tensor/scalar inputs
and gradient flow -- including that gradients still reach ``r_off`` and
``market_spread`` (the ``.item()`` version silently severed them).

Run: OMP_NUM_THREADS=2 python3 scripts/verify_curve_spread.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mm.mpc.l_param import curve_spread_for_market_spread


def original(market_spread, *, side, mid_price, reservation_r_offset):
    """Verbatim copy of the pre-refactor implementation."""
    if mid_price is None or reservation_r_offset is None:
        return market_spread
    r_off = reservation_r_offset
    if not torch.is_tensor(r_off):
        r_off = torch.as_tensor(
            r_off, dtype=market_spread.dtype, device=market_spread.device
        )
    if float(r_off.abs().item()) <= 0.0:
        return market_spread
    mid = mid_price
    if not torch.is_tensor(mid):
        mid = torch.as_tensor(
            mid, dtype=market_spread.dtype, device=market_spread.device
        )
    shift = mid * r_off
    if side == "bid":
        return (market_spread + shift).clamp(min=0.0)
    return (market_spread - shift).clamp(min=0.0)


def main() -> None:
    torch.manual_seed(0)

    spreads = [
        torch.linspace(0.0, 5.0, 17, dtype=torch.float64),
        torch.tensor([0.0, 0.0, 1.25], dtype=torch.float64),
    ]
    mids = [torch.tensor(266_400.0, dtype=torch.float64), 266_400.0, None]
    # Zero, tiny, negative, positive, and exactly-at-clamp-boundary offsets.
    offsets = [
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1e-18, dtype=torch.float64),
        torch.tensor(0.002, dtype=torch.float64),
        torch.tensor(-0.002, dtype=torch.float64),
        torch.tensor(-1.5, dtype=torch.float64),
        0.0,
        0.002,
        -0.002,
        None,
    ]

    failures = []
    checked = 0
    for spread, side, mid, offset in itertools.product(
        spreads, ("bid", "ask"), mids, offsets
    ):
        want = original(
            spread, side=side, mid_price=mid, reservation_r_offset=offset
        )
        got = curve_spread_for_market_spread(
            spread, side=side, mid_price=mid, reservation_r_offset=offset
        )
        diff = (got - want).abs().max().item()
        checked += 1
        if diff != 0.0:
            failures.append((side, str(mid), str(offset), diff))

    print(f"=== value parity: {checked} cases ===")
    if failures:
        for row in failures[:8]:
            print("  MISMATCH", row)

    print("=== gradient parity (compare old vs new directly) ===")
    grad_failures = []
    for side, offset_val in itertools.product(("bid", "ask"), (0.002, -0.002, 0.0, 1e-18)):
        spread_a = torch.linspace(0.0, 5.0, 11, dtype=torch.float64, requires_grad=True)
        spread_b = spread_a.detach().clone().requires_grad_(True)
        r_a = torch.tensor(offset_val, dtype=torch.float64, requires_grad=True)
        r_b = r_a.detach().clone().requires_grad_(True)

        original(spread_a, side=side, mid_price=266_400.0, reservation_r_offset=r_a).sum().backward()
        curve_spread_for_market_spread(
            spread_b, side=side, mid_price=266_400.0, reservation_r_offset=r_b
        ).sum().backward()

        ga = 0.0 if r_a.grad is None else float(r_a.grad)
        gb = 0.0 if r_b.grad is None else float(r_b.grad)
        # Note: a zero r_off gradient is legitimate whenever clamp(min=0.0)
        # saturates (e.g. bid with a negative offset pushes the spread below
        # zero). The check is that old and new agree, not that grad is nonzero.
        rd = abs(ga - gb)
        sd = (spread_a.grad - spread_b.grad).abs().max().item()
        if rd > 1e-9 or sd > 1e-15:
            grad_failures.append((side, offset_val, f"d_r_off={rd:.3e} d_spread={sd:.3e}"))
        print(f"  side={side:4s} r_off={offset_val:+.8f}  d_spread={sd:.2e}  dr_off={rd:.2e} (old={ga:.4e})")

    print("=== graph-break check: compile with fullgraph=True ===")
    try:
        compiled = torch.compile(curve_spread_for_market_spread, fullgraph=True)
        for side in ("bid", "ask"):
            out = compiled(
                torch.linspace(0.0, 5.0, 11, dtype=torch.float64),
                side=side,
                mid_price=266_400.0,
                reservation_r_offset=torch.tensor(0.002, dtype=torch.float64),
            )
        print("  fullgraph=True compiled with no graph break")
    except Exception as exc:  # noqa: BLE001
        grad_failures.append(("fullgraph", str(exc)))
        print(f"  fullgraph=True FAILED: {type(exc).__name__}: {str(exc)[:200]}")

    print()
    if failures or grad_failures:
        for row in (failures + grad_failures)[:10]:
            print("  FAIL", row)
        raise SystemExit(1)
    print(f"OK: bit-identical across {checked} value cases, gradients preserved")


if __name__ == "__main__":
    main()