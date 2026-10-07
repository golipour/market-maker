"""Verify the reservation-branch refactor preserves stacked/separated parity.

Before this change ``effective_depth_stacked_tensor`` selected its path with
``float(reservation_r_offset.abs().item()) > 0``. That ``.item()`` is a graph
break, so a venue with ``reservation_skew`` enabled always landed on the
unbatched two-decoder path. The branch is now derived from
``l_param.reservation_r_max`` (configuration) instead, which dynamo can
specialize on.

This asserts the refactor selects the same path as before and that whichever
path it picks still agrees with the separate decoders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mm.mpc.l_param import LParamConfig
from mm.mpc.state_tensor import (
    effective_depth_ask_tensor,
    effective_depth_bid_tensor,
    effective_depth_stacked_tensor,
)

N = 4
SPREAD_MAX = 200.0
R_MAX = 0.002


def _l_param(n_comp: int, r_max: float) -> LParamConfig:
    return LParamConfig(
        quote_size_min=1.0,
        quote_size_max=100.0,
        spread_max=SPREAD_MAX,
        n_components=n_comp,
        reservation_r_max=r_max,
    )


def _case(n_comp: int, r_max: float, spread: torch.Tensor, seed: int = 0):
    l_param = _l_param(n_comp, r_max)
    torch.manual_seed(seed)
    params = torch.randn(
        3 * 2 * n_comp + 3, dtype=torch.float64, requires_grad=True
    )
    size_min = torch.zeros((1, 1), dtype=torch.float64)
    size_max = torch.ones((1, 1), dtype=torch.float64)
    kwargs = {
        "use_softplus": True,
        "l_param": l_param,
        "bid_quote_size_min": size_min,
        "bid_quote_size_max": size_max,
        "ask_quote_size_min": size_min,
        "ask_quote_size_max": size_max,
    }
    if r_max > 0.0:
        kwargs["mid_price"] = torch.ones((1, 1), dtype=torch.float64) * 100.0
        kwargs["reservation_r_offset"] = torch.tensor(
            0.0015, dtype=torch.float64, requires_grad=True
        )
    stacked = effective_depth_stacked_tensor(
        params, spread, n_comp, **kwargs
    )
    # The separate decoders take unprefixed quote-size bounds; the stacked
    # entrypoint takes bid_/ask_-prefixed ones for the same values.
    bid_kwargs = {
        **kwargs,
        "quote_size_min": size_min,
        "quote_size_max": size_max,
    }
    ask_kwargs = dict(bid_kwargs)
    for key in (
        "bid_quote_size_min",
        "bid_quote_size_max",
        "ask_quote_size_min",
        "ask_quote_size_max",
    ):
        bid_kwargs.pop(key, None)
        ask_kwargs.pop(key, None)
    separated = torch.stack(
        [
            effective_depth_bid_tensor(params, spread, n_comp, **bid_kwargs),
            effective_depth_ask_tensor(params, spread, n_comp, **ask_kwargs),
        ],
        dim=0,
    )
    return stacked, separated


def main() -> None:
    # Production passes a 1-D spread grid (mm/mpc/_spread_grid_tensor: upper
    # bucket edges), not a per-sample matrix.
    spread = (
        torch.arange(1.0, 11.0, dtype=torch.float64).clamp(max=SPREAD_MAX)
    )
    failures = []

    print("=== parity: stacked vs separated decoders ===")
    for n_comp in (4, 2):
        for r_max in (0.0, R_MAX):
            stacked, separated = _case(n_comp, r_max, spread)
            diff = (stacked - separated).abs().max().item()
            finite = bool(torch.isfinite(stacked).all())
            print(
                f"n_components={n_comp} reservation_r_max={r_max}: "
                f"shape={tuple(stacked.shape)} "
                f"max|diff|={diff:.3e} finite={finite}"
            )
            if diff > 1e-12 or not finite:
                failures.append((n_comp, r_max, diff, finite))

    print("\n=== path selection is configuration-derived (no .item()) ===")
    from mm.mpc.state_tensor import _reservation_active

    for r_max, expected in ((R_MAX, True), (0.0, False)):
        actual = _reservation_active(_l_param(4, r_max))
        print(f"reservation_r_max={r_max} -> active={actual} (expected {expected})")
        if actual != expected:
            failures.append(("branch", r_max, actual, expected))

    print("\n=== dispatch has no .item() call (AST, ignores comments) ===")
    import ast
    import inspect

    from mm.mpc import state_tensor

    tree = ast.parse(inspect.getsource(state_tensor.effective_depth_stacked_tensor))
    item_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    ]
    print(f".item() calls in effective_depth_stacked_tensor: {len(item_calls)}")
    if item_calls:
        failures.append(("graph_break", len(item_calls), True, False))

    print()
    if failures:
        print(f"FAILED: {failures}")
        raise SystemExit(1)
    print("OK: parity preserved and dispatch is graph-break free")


if __name__ == "__main__":
    main()
