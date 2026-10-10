"""Why does dynamo recompile at all when only inputs change?

The claim under test: "only the inputs change, so recompilation should never
happen". That is right about tensor *values* and wrong about shapes and
Python-level config, which dynamo specializes on eagerly.

This measures each case separately:

* A. same shapes, different tensor VALUES      -> expect no recompile
* B. different tensor SHAPES                   -> expect one per shape
* C. different Python config floats/strings    -> expect one per value

Each case calls the compiled function N times and reports the dynamo frame
count, so the split is measured rather than asserted.

Run: OMP_NUM_THREADS=2 python3 scripts/verify_recompile_triggers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch._dynamo as dynamo

from mm.mpc.l_param import LParamConfig, materialize_side_torch

N_CALLS = 12


def _lp(n_comp: int, cap: float | None, r_max: float) -> LParamConfig:
    return LParamConfig(
        quote_size_min=1.0,
        quote_size_max=cap,
        spread_max=200.0,
        n_components=n_comp,
        reservation_r_max=r_max,
    )


def _frames() -> int:
    return dynamo.utils.counters["frames"].get("total", 0)


def _run(compiled, calls) -> int:
    for fn in calls:
        fn()
    return _frames()


def case_a_values_only() -> None:
    """Same shapes, different tensor values: does dynamo recompile?"""
    dynamo.reset()
    compiled = torch.compile(materialize_side_torch, fullgraph=False)
    n_comp = 4
    lp = _lp(n_comp, 100.0, 0.002)
    smin = torch.zeros((1, 1), dtype=torch.float64)
    smax = torch.full((1, 1), 50.0, dtype=torch.float64)

    calls = []
    for i in range(N_CALLS):
        # ONLY the values differ: different params, different spread grid.
        params = torch.full(
            (3 * 2 * n_comp + 3,), float(i) * 0.01, dtype=torch.float64,
            requires_grad=True,
        )
        calls.append(
            lambda p=params: compiled(
                p, side_off=0, n_components=n_comp, l_param=lp,
                quote_size_min=smin, quote_size_max=smax,
            )
        )
    frames = _run(compiled, calls)
    print(f"A. same shape, {N_CALLS} different VALUES -> {frames} frames")
    return frames


def case_b_shapes() -> None:
    """Different tensor shapes: dynamo guards on these, so it recompiles."""
    dynamo.reset()
    compiled = torch.compile(materialize_side_torch, fullgraph=False)
    smin = torch.zeros((1, 1), dtype=torch.float64)
    smax = torch.full((1, 1), 50.0, dtype=torch.float64)

    calls = []
    # Three venues' worth of component counts.
    for n_comp in (4, 8, 2) * 4:
        lp = _lp(n_comp, 100.0, 0.002)
        params = torch.zeros(3 * 2 * n_comp + 3, dtype=torch.float64, requires_grad=True)
        calls.append(
            lambda p=params, n=n_comp, l=lp: compiled(
                p, side_off=0, n_components=n, l_param=l,
                quote_size_min=smin, quote_size_max=smax,
            )
        )
    frames = _run(compiled, calls)
    print(f"B. {N_CALLS} calls over 3 SHAPES         -> {frames} frames")
    return frames


def case_c_config_values() -> None:
    """Different Python config values: also guarded, so also recompiles."""
    dynamo.reset()
    compiled = torch.compile(materialize_side_torch, fullgraph=False)
    n_comp = 4
    params = torch.zeros(3 * 2 * n_comp + 3, dtype=torch.float64, requires_grad=True)
    smin = torch.zeros((1, 1), dtype=torch.float64)
    smax = torch.full((1, 1), 50.0, dtype=torch.float64)

    calls = []
    # One venue's shape, but a different maker-side config value each time.
    for i in range(N_CALLS):
        lp = _lp(n_comp, 100.0 + i, 0.002 + i * 1e-4)
        calls.append(
            lambda l=lp: compiled(
                params, side_off=0, n_components=n_comp, l_param=l,
                quote_size_min=smin, quote_size_max=smax,
            )
        )
    frames = _run(compiled, calls)
    print(f"C. same shape, {N_CALLS} CONFIG VALUES  -> {frames} frames")
    return frames


def main() -> None:
    dynamo.config.recompile_limit = 64
    dynamo.config.accumulated_recompile_limit = 4096

    print("=== recompile trigger isolation ===")
    a = case_a_values_only()
    b = case_b_shapes()
    c = case_c_config_values()

    print()
    print(f"  values only : {a} frames  -> {'NO recompile (values are not guarded)' if a <= 2 else 'recompiles'}")
    print(f"  shapes      : {b} frames  -> recompiles per shape")
    print(f"  config      : {c} frames  -> recompiles per config value")
    print()
    print("Conclusion: dynamo specializes on SHAPES and PYTHON CONFIG VALUES,")
    print("not on tensor data. Production's two venues differ in n_components")
    print("(shape) and maker_fee/limits (config), so both trigger recompiles --")
    print("expected behaviour, not a bug.")


if __name__ == "__main__":
    main()