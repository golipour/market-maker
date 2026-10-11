"""Is the single-venue rollout a special case of the joint rollout?

If it is, then a joint rollout with a degenerate global venue -- same book, same
prices, zero inventory, zero cash, masked fills -- must reduce *exactly* to the
single-venue rollout. That would mean:

  1. the two code paths are one algorithm with a specialization, not two
     algorithms, and
  2. the single-venue path exists only as a fast path, so its correctness is a
     performance question while the joint path defines the semantics.

This measures that reduction directly. It runs both paths on the same states
and compares the local action and the objective. It also reports the wall time
of each, because if the reduction holds then the whole per-decide budget
argument rests on the joint path doing redundant work.

Run: OMP_NUM_THREADS=2 python3 research/model_lab/analysis/p9_special_case.py
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

from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched  # noqa: E402

PROD = dict(
    n_components=4,
    n_samples=64,
    horizon=2,
    spread_max=200.0,
    spread_step=20.0,
    adam_lr=1.0,
    adam_epochs=20,
)


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


def _opt(*, reference_price_source: str, n_components: int = 4):
    kw = dict(PROD)
    price, flow = _FixedPrice(), _FixedFlow()
    return MonteCarloMPCTensorBatched(
        local_price_model=price,
        local_market_flow_model=flow,
        execution_model=_Exec(),
        global_price_model=price,
        global_market_flow_model=flow,
        reference_price_source=reference_price_source,
        local_optimizer={"n_components": n_components},
        global_optimizer={"n_components": n_components},
        fixed_mc_samples=True,
        torch_threads=int(os.environ.get("P9_THREADS", "1")),
        device="cpu",
        compile_soft_fill=False,
        **kw,
    )


def _act_vec(a) -> np.ndarray:
    return np.concatenate([
        np.atleast_1d(np.asarray(getattr(a, f"{s}_{p}"), dtype=float))
        for s in ("bid", "ask") for p in ("L", "k", "x0")
    ])


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
    n_states = int(os.environ.get("P9_STATES", "6"))
    raw = build_states(n_states=n_states)
    states = [st for st, _ in raw]

    # The degenerate global venue: literally the same book, same prices, and a
    # flat position, so a correctly-specialized joint rollout has nothing extra
    # to simulate and must reproduce the single-venue answer.
    def _mirror_local(st):
        object.__setattr__(st, "global_ob_rel", st.ob_rel)
        object.__setattr__(st, "global_active", True)
        object.__setattr__(st, "global_inventory", 0.0)
        object.__setattr__(st, "global_cash", 0.0)
        object.__setattr__(st, "global_buy_vol", 0.0)
        object.__setattr__(st, "global_sell_vol", 0.0)
        object.__setattr__(st, "global_buy_vwap", float(st.buy_vwap))
        object.__setattr__(st, "global_sell_vwap", float(st.sell_vwap))
        return st

    mirrored = [_mirror_local(st) for st in states]

    single = _opt(reference_price_source="local")
    joint = _opt(reference_price_source="global")

    print(f"threads={single.torch_threads}  corpus={len(states)} states")
    _, d_s = single.optimize_with_diagnostics(states[0])
    _, d_j = joint.optimize_with_diagnostics(mirrored[0])
    print(f"single-venue path: joint_venue={d_s.config.get('joint_venue')}")
    print(f"joint path       : joint_venue={d_j.config.get('joint_venue')}\n")

    diffs, obj_diffs = [], []
    for s_st, m_st in zip(states, mirrored):
        a_s, dd_s = single.optimize_with_diagnostics(s_st)
        a_j, dd_j = joint.optimize_with_diagnostics(m_st)
        diffs.append(float(np.max(np.abs(_act_vec(a_s) - _act_vec(a_j)))))
        obj_diffs.append(abs(float(dd_s.best.objective_value) - float(dd_j.best.objective_value)))

    print("single vs joint-with-degenerate-global:")
    print(f"  max |action diff|    = {max(diffs):.3e}")
    print(f"  max |objective diff| = {max(obj_diffs):.3e}")

    t_s = statistics.median(timeit(single, states))
    t_j = statistics.median(timeit(joint, mirrored))
    print(f"\nsingle-venue median {t_s:8.2f} ms/decide")
    print(f"joint (degenerate)  {t_j:8.2f} ms/decide   ratio {t_j / t_s:.2f}x")

    print()
    if max(diffs) == 0.0 and max(obj_diffs) == 0.0:
        print("VERDICT: exact reduction. The single-venue path IS the joint path")
        print("         with a degenerate second venue, and the 2.3x gap is")
        print("         redundant work, not different semantics.")
    elif max(diffs) < 1e-9 and max(obj_diffs) < 1e-12:
        print("VERDICT: equal to float64 reassociation. Same algorithm; the gap")
        print("         is redundant work, not different semantics.")
    else:
        print("VERDICT: NOT a reduction. The two paths encode different models,")
        print("         so unifying them is a semantics change, not a refactor.")


if __name__ == "__main__":
    main()