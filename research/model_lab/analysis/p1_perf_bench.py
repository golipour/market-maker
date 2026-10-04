"""Benchmark the optimizer at the *production* preset, not the unit-test fixture.

Earlier numbers in this investigation (30.8ms decide, 1.5x for the stacked
decoder, 1.5x for torch.compile) were measured with unit-test fixture
parameters: ``adam_lr=0.05``, ``horizon=4``, ``spread_max=20``. The shipped
preset is different:

    n_samples 64, horizon 2, adam_epochs 20, adam_lr 1.0,
    n_components 4, spread_max 200, spread_step 20

``adam_lr`` differs by 20x, which changes the whole convergence picture, and
``spread_max`` differs by 10x, which changes the quote grid from 10 levels to
10 levels of a very different scale. Any performance or convergence claim
has to be stated at the production configuration.

Variants measured:
  eager_stacked   current HEAD, stacked decoder, no compile
  eager_unstacked stacked decoder disabled (separate bid/ask decoders)
  compiled        stacked decoder + torch.compile on soft_fill
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_states import build_states  # noqa: E402

from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched  # noqa: E402


class _FixedFlow:
    def sample(self, shape, device=None, dtype=None):
        dt = dtype or torch.float64
        return (
            torch.ones(shape, dtype=dt, device=device),
            torch.ones(shape, dtype=dt, device=device),
        )


class _FixedPrice:
    def sample_residual(self, shape, device=None, dtype=None):
        dt = dtype or torch.float64
        z = torch.zeros(shape, dtype=dt, device=device)
        return z, z


class _Exec:
    maker_fee = 0.0


PROD = dict(
    n_components=4,
    n_samples=64,
    horizon=2,
    spread_max=200.0,
    spread_step=20.0,
    adam_lr=1.0,
    adam_epochs=20,
)


def _opt(compile_soft_fill: bool = False, **over):
    kw = {**PROD, **over}
    return MonteCarloMPCTensorBatched(
        local_price_model=_FixedPrice(),
        local_market_flow_model=_FixedFlow(),
        execution_model=_Exec(),
        reference_price_source="local",
        fixed_mc_samples=True,
        torch_threads=1,
        compile_soft_fill=compile_soft_fill,
        **kw,
    )


def timeit(opt, states, warm=3, reps=7):
    for _ in range(warm):
        for st, _ in states:
            opt.optimize_with_diagnostics(st)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for st, _ in states:
            opt.optimize_with_diagnostics(st)
        samples.append((time.perf_counter() - t0) * 1000.0 / len(states))
    return samples


def main() -> None:
    n_states = int(os.environ.get("P1_STATES", "12"))
    states = build_states(n_states=n_states)
    print(f"production preset: {PROD}")
    print(f"corpus: {len(states)} real Nobitex states\n")

    results = {}

    opt_eager = _opt(compile_soft_fill=False)
    s = timeit(opt_eager, states)
    results["eager_stacked"] = s
    print(f"eager_stacked    median {statistics.median(s):7.2f} ms/decide   "
          f"min {min(s):7.2f}  max {max(s):7.2f}")

    # Unstacked: force the two-separate-decoder path by disabling the
    # structured/vectorized branch via a non-structured LParamConfig.
    import mm.mpc.state_tensor as st_mod

    real_stacked = st_mod.effective_depth_stacked_tensor

    def _sep_only(params, spread, n_components, **kw):
        """Emulate the pre-stacking decoder: two independent evaluations."""
        torch = st_mod._get_torch()
        db = st_mod.effective_depth_bid_tensor(
            params, spread, n_components,
            use_softplus=kw.get("use_softplus", True),
            l_param=kw.get("l_param"),
            quote_size_min=kw.get("bid_quote_size_min"),
            quote_size_max=kw.get("bid_quote_size_max"),
            mid_price=kw.get("mid_price"),
            reservation_r_offset=kw.get("reservation_r_offset"),
        )
        da = st_mod.effective_depth_ask_tensor(
            params, spread, n_components,
            use_softplus=kw.get("use_softplus", True),
            l_param=kw.get("l_param"),
            quote_size_min=kw.get("ask_quote_size_min"),
            quote_size_max=kw.get("ask_quote_size_max"),
            mid_price=kw.get("mid_price"),
            reservation_r_offset=kw.get("reservation_r_offset"),
        )
        return torch.stack([db, da], dim=0)

    import mm.mpc.mpc_tensor_batched as mpc_mod

    mpc_mod.effective_depth_stacked_tensor = _sep_only
    try:
        opt_un = _opt(compile_soft_fill=False)
        su = timeit(opt_un, states)
        results["eager_unstacked"] = su
        print(f"eager_unstacked  median {statistics.median(su):7.2f} ms/decide   "
              f"min {min(su):7.2f}  max {max(su):7.2f}")
    finally:
        mpc_mod.effective_depth_stacked_tensor = real_stacked

    opt_c = _opt(compile_soft_fill=True)
    sc = timeit(opt_c, states, warm=1, reps=5)
    results["compiled"] = sc
    print(f"compiled         median {statistics.median(sc):7.2f} ms/decide   "
          f"min {min(sc):7.2f}  max {max(sc):7.2f}")

    me = statistics.median(results["eager_stacked"])
    mu = statistics.median(results["eager_unstacked"])
    mc = statistics.median(results["compiled"])
    print()
    print(f"stacking gain (eager): {mu / me:.3f}x   ({mu:.2f} -> {me:.2f} ms)")
    print(f"compile gain (stacked): {me / mc:.3f}x   ({me:.2f} -> {mc:.2f} ms)")
    print(f"compile gain vs unstacked: {mu / mc:.3f}x")

    # Parity of decisions across the three variants.
    vecs = {}
    objs = {}
    for name, mk in [
        ("eager_stacked", lambda: _opt(False)),
        ("compiled", lambda: _opt(True)),
    ]:
        o = mk()
        vv, jj = [], []
        for st, _ in states:
            for _ in range(3):
                a, d = o.optimize_with_diagnostics(st)
            vv.append(np.concatenate([np.atleast_1d(np.asarray(getattr(a, f"{s}_{p}"), dtype=float))
                                      for s in ("bid", "ask") for p in ("L", "k", "x0")]))
            jj.append(float(d.best.objective_value))
        vecs[name] = np.array(vv)
        objs[name] = np.array(jj)

    dv = np.max(np.abs(vecs["eager_stacked"] - vecs["compiled"]))
    do = np.max(np.abs(objs["eager_stacked"] - objs["compiled"]))
    print(f"\nparity eager vs compiled: max|dAct|={dv:.3e}  max|dJ|={do:.3e}")

    out = {
        "production": PROD,
        "n_states": len(states),
        "medians_ms": {
            "eager_stacked": me,
            "eager_unstacked": mu,
            "compiled": mc,
        },
        "samples": results,
        "parity": {"max_action_diff": float(dv), "max_objective_diff": float(do)},
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "p1_perf.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
