"""Why do the single-venue and degenerate-global-joint rollouts disagree?

p9 showed they are NOT a special case: with a global venue that mirrors the
local book, flat inventory and zero cash, the two paths still differ by ~74
action units and 1.8e-4 in objective. Since the degenerate global venue should
contribute nothing, that gap is either

  (a) a modelling term that is supposed to vanish but does not, or
  (b) a genuine difference in how the two paths price inventory.

This isolates *which term* differs by diffing the two paths term by term on the
same state, rather than only comparing the final action:

  - initial wealth W0            (joint adds global_cash/global_inventory)
  - the per-step reward          (joint sums local + global step rewards)
  - the terminal objective J     (entropic risk-adjusted return)
  - validity / viability terms   (joint applies global ones too)

Whichever term moves first identifies the cause.

Run: OMP_NUM_THREADS=2 python3 research/model_lab/analysis/p10_reduction_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from replay_states import build_states  # noqa: E402

from mm.mpc.mpc_state import OrderbookRelative  # noqa: E402
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched  # noqa: E402
from mm.mpc.joint_mpc import wealth_0_from_state, reference_price_from_state  # noqa: E402

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


def main() -> None:
    raw = build_states(n_states=int(os.environ.get("P10_STATES", "3")))
    st = raw[0][0]

    # Snapshot the pristine state before the degenerate-global mutation.
    pristine = dict(
        inventory=float(st.inventory),
        cash=float(st.cash),
        buy_vwap=float(st.buy_vwap),
        sell_vwap=float(st.sell_vwap),
        reference_price=float(st.reference_price or st.mid),
        global_inventory=float(getattr(st, "global_inventory", 0.0) or 0.0),
        global_cash=float(getattr(st, "global_cash", 0.0) or 0.0),
        global_sell_vwap=float(getattr(st, "global_sell_vwap", 0.0) or 0.0),
        global_buy_vwap=float(getattr(st, "global_buy_vwap", 0.0) or 0.0),
    )
    print("pristine state:")
    for k, v in pristine.items():
        print(f"  {k:18s} = {v}")

    print("\n-- wealth_0 / reference price under each path --")
    for src in ("local", "global"):
        ref = reference_price_from_state(st, src)
        w0 = wealth_0_from_state(st, src)
        print(f"  source={src:6s} ref={ref:14.6f}  wealth_0={w0:18.6f}")

    # Now make the global venue degenerate: mirror the local book, flat
    # position, zero cash.
    object.__setattr__(st, "global_ob_rel", st.ob_rel)
    object.__setattr__(st, "global_active", True)
    object.__setattr__(st, "global_inventory", 0.0)
    object.__setattr__(st, "global_cash", 0.0)
    object.__setattr__(st, "global_buy_vol", 0.0)
    object.__setattr__(st, "global_sell_vol", 0.0)
    object.__setattr__(st, "global_buy_vwap", pristine["buy_vwap"])
    object.__setattr__(st, "global_sell_vwap", pristine["sell_vwap"])

    print("\n-- after making the global venue degenerate --")
    for src in ("local", "global"):
        ref = reference_price_from_state(st, src)
        w0 = wealth_0_from_state(st, src)
        print(f"  source={src:6s} ref={ref:14.6f}  wealth_0={w0:18.6f}")

    # With global_inventory = 0 the consolidated wealth should be exactly the
    # local wealth, because consolidated_wealth = local + global + (li+gi)*ref
    # and gi == 0.
    print("\n-- consolidated vs local wealth decomposition --")
    ref_local = reference_price_from_state(st, "local")
    li, gi = pristine["inventory"], 0.0
    lc, gc = pristine["cash"], 0.0
    local_only = lc + li * ref_local
    consolidated = lc + gc + (li + gi) * ref_local
    print(f"  local_only       = {local_only:.6f}")
    print(f"  consolidated     = {consolidated:.6f}")
    print(f"  difference       = {consolidated - local_only:.3e}  <- 0 means W0 agrees")

    # Now the term that is NOT masked in the joint rollout: the global step
    # reward is summed into step_reward regardless of global_mask. With gi == 0
    # the *level* contribution vanishes, but the global chain still evolves
    # global_cash/global_inv through the rollout and can leave a residual.
    single = _opt(reference_price_source="local")
    joint = _opt(reference_price_source="global")
    a_s, d_s = single.optimize_with_diagnostics(st)
    a_j, d_j = joint.optimize_with_diagnostics(st)

    def vec(a):
        return np.concatenate([
            np.atleast_1d(np.asarray(getattr(a, f"{s}_{p}"), dtype=float))
            for s in ("bid", "ask") for p in ("L", "k", "x0")
        ])

    print("\n-- terminal objectives on the identical state --")
    print(f"  single J = {float(d_s.best.objective_value):.12e}")
    print(f"  joint  J = {float(d_j.best.objective_value):.12e}")
    print(f"  diff      = {abs(float(d_s.best.objective_value) - float(d_j.best.objective_value)):.3e}")
    print(f"  max |dAct| = {float(np.max(np.abs(vec(a_s) - vec(a_j)))):.3e}")

    for label, dd in (("single", d_s), ("joint", d_j)):
        roll = dd.rollout
        keys = ("consolidated_wealth", "global_rollout", "global_reference",
                "global_fill_bid_mean", "global_fill_ask_mean",
                "local_fill_bid_mean", "local_fill_ask_mean")
        print(f"\n  {label} rollout flags:")
        for k in keys:
            if k in roll:
                print(f"    {k:24s} = {roll[k]}")


if __name__ == "__main__":
    main()