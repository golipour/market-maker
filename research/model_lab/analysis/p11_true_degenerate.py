"""Is the single-venue rollout the joint rollout with an absent global venue?

p9/p10 showed a 74-unit action gap, but that test made the global venue *live*
(``global_active=True``), which activates a second quoting chain. That is not
the production case. Production runs ``global_active=False`` with
``global_status: "disabled"`` and 100% ``mode: single_venue``.

So the honest reduction test is against the production shape, and it has to
distinguish three cases:

  A. No global venue at all        -> the single-venue rollout is selected
  B. Global venue present but idle  -> what production actually runs
  C. Global venue live              -> genuine two-venue operation

If B equals A on the same prices, then the joint rollout carries a redundant
chain and the single-venue path is a legitimate fast path for it. If B differs
from A even with a mirrored book, the difference comes from the *marking*
book, not from extra quoting, and that is a modelling decision rather than
wasted work.

Run: OMP_NUM_THREADS=2 python3 research/model_lab/analysis/p11_true_degenerate.py
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


def _opt(*, reference_price_source: str):
    kw = dict(PROD)
    price, flow = _FixedPrice(), _FixedFlow()
    return MonteCarloMPCTensorBatched(
        local_price_model=price,
        local_market_flow_model=flow,
        execution_model=_Exec(),
        global_price_model=price,
        global_market_flow_model=flow,
        reference_price_source=reference_price_source,
        local_optimizer={"n_components": 4},
        global_optimizer={"n_components": 4},
        fixed_mc_samples=True,
        torch_threads=1,
        device="cpu",
        compile_soft_fill=False,
        **kw,
    )


def vec(a) -> np.ndarray:
    return np.concatenate([
        np.atleast_1d(np.asarray(getattr(a, f"{s}_{p}"), dtype=float))
        for s in ("bid", "ask") for p in ("L", "k", "x0")
    ])


def main() -> None:
    n = int(os.environ.get("P11_STATES", "6"))
    raw = build_states(n_states=n)
    states = [st for st, _ in raw]

    # B: global venue present but idle, mirroring the local book exactly.
    # depth_scale becomes global_mask = 0, so the global chain quotes nothing,
    # and its inventory/cash stay at zero. Its book is the market's only.
    idle = []
    for st in states:
        object.__setattr__(st, "global_ob_rel", st.ob_rel)
        object.__setattr__(st, "global_active", False)
        object.__setattr__(st, "global_inventory", 0.0)
        object.__setattr__(st, "global_cash", 0.0)
        object.__setattr__(st, "global_buy_vol", 0.0)
        object.__setattr__(st, "global_sell_vol", 0.0)
        object.__setattr__(st, "global_buy_vwap", float(st.buy_vwap))
        object.__setattr__(st, "global_sell_vwap", float(st.sell_vwap))
        idle.append(st)

    single = _opt(reference_price_source="local")
    joint = _opt(reference_price_source="global")

    _, d_s = single.optimize_with_diagnostics(states[0])
    _, d_j = joint.optimize_with_diagnostics(idle[0])
    print(f"single selected joint_venue = {d_s.config.get('joint_venue')}")
    print(f"joint  selected joint_venue = {d_j.config.get('joint_venue')}")
    print(f"joint rollout flags: consolidated_wealth={d_j.rollout.get('consolidated_wealth')} "
          f"global_reference={d_j.rollout.get('global_reference')}\n")

    diffs, objs = [], []
    for a, b in zip(states, idle):
        act_s, dd_s = single.optimize_with_diagnostics(a)
        act_j, dd_j = joint.optimize_with_diagnostics(b)
        diffs.append(float(np.max(np.abs(vec(act_s) - vec(act_j)))))
        objs.append(abs(float(dd_s.best.objective_value) - float(dd_j.best.objective_value)))

    print("A (no global venue) vs B (global venue present but idle, mirrored book):")
    print(f"  max |action diff|    = {max(diffs):.3e}")
    print(f"  max |objective diff| = {max(objs):.3e}")

    def t(opt, sts, warm=2, reps=5):
        for _ in range(warm):
            for s in sts:
                opt.optimize_with_diagnostics(s)
        out = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for s in sts:
                opt.optimize_with_diagnostics(s)
            out.append((time.perf_counter() - t0) * 1000.0 / len(sts))
        return statistics.median(out)

    t_a = t(single, states)
    t_b = t(joint, idle)
    print(f"\n  A single-venue  {t_a:7.2f} ms/decide")
    print(f"  B idle global   {t_b:7.2f} ms/decide   ratio {t_b / t_a:.2f}x")

    print()
    if max(diffs) < 1e-9 and max(objs) < 1e-12:
        print("VERDICT: B reduces to A. The idle global chain is redundant work;")
        print("         the joint rollout is the general algorithm and the")
        print("         single-venue path is a valid fast path for it.")
    else:
        print("VERDICT: B does NOT reduce to A. The difference is the *marking*")
        print("         book (the joint path prices inventory off the global")
        print("         VWAPs even when it quotes nothing there), so the gap is")
        print("         a modelling choice, not wasted work.")


if __name__ == "__main__":
    main()