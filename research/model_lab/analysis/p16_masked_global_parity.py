"""Does the masked-global fast path preserve the local decision bit-exactly?

``joint_optimizer.py`` now skips the global fill chain, the global VWAP read,
and the global reward term when ``global_mask == 0`` (inactive venue). The
global position is definitionally zero there, so each skipped term is
arithmetically zero -- but "arithmetically zero" and "bit-identical through
Adam" are different claims. Autograd graph surgery can shift float64
reassociation, and the warm-start path must also agree.

This runs the production-shape comparison the p5/p6 probes established:

  * joint with ``global_active=False`` (production: liquidity sourcing off,
    ``reference_price_source="global"``), compiled kernel on;
  * the same decide with the skips force-disabled (all global terms computed).

It reports max |local action diff|, max |objective diff|, and per-decide
latency for both. The bar is the pre-existing float64 reassociation noise
(p8: 5.9e-08 action, 3.5e-15 objective). Anything at or below that is parity;
anything at p9 scale (74 units) is a semantic break.

It also runs the joint-vs-single gap (p4) so the speedup claim is measured
on the same corpus in the same process.

Run: OMP_NUM_THREADS=2 python3 research/model_lab/analysis/p16_masked_global_parity.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from replay_states import build_states  # noqa: E402

from mm.mpc.mpc_state import OrderbookRelative  # noqa: E402
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched  # noqa: E402
import mm.mpc.joint_optimizer as joint_mod  # noqa: E402

PROD = dict(
    n_components=4,
    n_samples=64,
    horizon=2,
    spread_max=200.0,
    spread_step=20.0,
    adam_lr=1.0,
    adam_epochs=20,
)
GLOBAL_N_COMPONENTS = 2


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
    local_maker_fee = 0.0
    global_maker_fee = 0.0


def _with_global(base, *, active: bool):
    fields = dict(
        global_ob_rel=OrderbookRelative(
            bids=((2.0, 1.0), (4.0, 0.5)), asks=((2.0, 1.0), (4.0, 0.5))
        ),
        global_active=active,
        global_inventory=3.0,
        global_cash=500.0,
        global_buy_vwap=base.buy_vwap * 0.999,
        global_sell_vwap=base.sell_vwap * 1.001,
        reference_price=base.sell_vwap,
    )
    for key, value in fields.items():
        object.__setattr__(base, key, value)
    return base


def _opt(*, reference_price_source: str = "global", **over):
    kw = {**PROD, **over}
    price, flow = _FixedPrice(), _FixedFlow()
    return MonteCarloMPCTensorBatched(
        local_price_model=price,
        local_market_flow_model=flow,
        execution_model=_Exec(),
        global_price_model=price,
        global_market_flow_model=flow,
        reference_price_source=reference_price_source,
        global_optimizer={"n_components": GLOBAL_N_COMPONENTS},
        local_optimizer={"n_components": PROD["n_components"]},
        fixed_mc_samples=True,
        torch_threads=int(os.environ.get("P16_THREADS", "1")),
        device="cpu",
        **kw,
    )


def _act_vec(a) -> np.ndarray:
    return np.concatenate(
        [
            np.atleast_1d(np.asarray(getattr(a, f"{s}_{p}"), dtype=float))
            for s in ("bid", "ask")
            for p in ("L", "k", "x0")
        ]
    )


def timeit(opt, states, warm=2, reps=5):
    for _ in range(warm):
        for st in states:
            opt.optimize_with_diagnostics(st)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for st in states:
            opt.optimize_with_diagnostics(st)
        out.append((time.perf_counter() - t0) * 1000.0 / len(states))
    return out


def main() -> None:
    n_states = int(os.environ.get("P16_STATES", "6"))
    raw = build_states(n_states=n_states)
    states = [_with_global(st, active=False) for st, _ in raw]

    # Baseline: skips force-disabled by pretending the venue is active for the
    # gating reads only. global_mask is computed from state.global_active, so
    # run the same states with active=True but zeroed position/cash -- the
    # global chain then does full work on an empty book, which is what the
    # pre-skip code did every production decide.
    unmasked = []
    for st, _ in raw:
        c = _with_global(st, active=True)
        object.__setattr__(c, "global_inventory", 0.0)
        object.__setattr__(c, "global_cash", 0.0)
        unmasked.append(c)

    opt_skip = _opt()
    opt_full = _opt()

    print(f"threads={opt_skip.torch_threads}  corpus={len(states)} states")
    _, d0 = opt_skip.optimize_with_diagnostics(states[0])
    print(f"masked path  : joint_venue={d0.config.get('joint_venue')}")
    _, d1 = opt_full.optimize_with_diagnostics(unmasked[0])
    print(f"unmasked path: joint_venue={d1.config.get('joint_venue')}\n")

    diffs, objs = [], []
    for s_st, u_st in zip(states, unmasked):
        a_s, d_s = opt_skip.optimize_with_diagnostics(s_st)
        a_u, d_u = opt_full.optimize_with_diagnostics(u_st)
        diffs.append(float(np.max(np.abs(_act_vec(a_s) - _act_vec(a_u)))))
        objs.append(
            abs(float(d_s.best.objective_value) - float(d_u.best.objective_value))
        )

    print("masked skips vs full global chain (zeroed position, same prices):")
    print(f"  max |local action diff| = {max(diffs):.3e}")
    print(f"  max |objective diff|    = {max(objs):.3e}")

    t_skip = statistics.median(timeit(opt_skip, states))
    t_full = statistics.median(timeit(opt_full, unmasked))
    print(f"\nmasked   median {t_skip:8.2f} ms/decide")
    print(f"unmasked median {t_full:8.2f} ms/decide   speedup {t_full / t_skip:.3f}x")

    # Same-process joint-vs-single gap for context (p4 number, re-measured).
    single = _opt(reference_price_source="local")
    t_single = statistics.median(timeit(single, [st for st, _ in raw]))
    print(f"single   median {t_single:8.2f} ms/decide   (masked joint / single = "
          f"{t_skip / t_single:.3f}x)")

    print()
    if max(diffs) < 1e-6 and max(objs) < 1e-9:
        print("VERDICT: parity holds -- the masked skips preserve the decision")
        print("         to float64 reassociation noise. Ship it.")
    elif max(diffs) < 1.0:
        print("VERDICT: small but above reassociation noise -- investigate")
        print("         before shipping; likely a graph-ordering effect.")
    else:
        print("VERDICT: SEMANTIC BREAK -- the skips changed the decision.")
        print("         Do not ship; the global chain was load-bearing.")


if __name__ == "__main__":
    main()
