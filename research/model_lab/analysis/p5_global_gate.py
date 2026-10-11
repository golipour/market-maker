"""Is the global rollout chain doing full work when the global venue is idle?

``optimize_joint_with_diagnostics`` always evaluates both venue chains per
horizon step. The global chain's contribution is scaled by ``global_mask``
(``depth_scale=global_mask``), so when ``state.global_active`` is False the
gradient contribution is zero -- but the arithmetic is still performed, inside
the autograd graph, on every epoch.

That matters because production frames are frequently ``single_venue``: the
``JointTargetBook`` payloads carry ``"mode": "single_venue"``. If the global
chain is inactive on a meaningful share of decisions, skipping it is a pure
structural win with no effect on the returned action.

This measures, at the shipped preset:
  1. cost with ``global_active=True``  (global chain fully live)
  2. cost with ``global_active=False`` (global chain masked to zero)
  3. whether the two return bit-identical local actions

If (2) is not materially cheaper than (1), the global chain is not being
skipped and there is no free win; if (1) ~= (2), skipping it when inactive is
worth roughly the difference.

Run: OMP_NUM_THREADS=2 python3 research/model_lab/analysis/p5_global_gate.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from replay_states import build_states  # noqa: E402

from mm.mpc.mpc_state import OrderbookRelative, State  # noqa: E402
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


def _with_global(base: State, *, active: bool) -> State:
    fields = dict(
        global_ob_rel=OrderbookRelative(bids=((2.0, 1.0), (4.0, 0.5)), asks=((2.0, 1.0), (4.0, 0.5))),
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


def _opt(**over):
    kw = {**PROD, **over}
    price, flow = _FixedPrice(), _FixedFlow()
    return MonteCarloMPCTensorBatched(
        local_price_model=price,
        local_market_flow_model=flow,
        execution_model=_Exec(),
        global_price_model=price,
        global_market_flow_model=flow,
        reference_price_source="global",
        global_optimizer={"n_components": GLOBAL_N_COMPONENTS},
        local_optimizer={"n_components": PROD["n_components"]},
        fixed_mc_samples=True,
        torch_threads=int(os.environ.get("P5_THREADS", "1")),
        device="cpu",
        **kw,
    )


def timeit(opt, states, warm=3, reps=7):
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


def _action_vec(a) -> np.ndarray:
    return np.concatenate([
        np.atleast_1d(np.asarray(getattr(a, f"{s}_{p}"), dtype=float))
        for s in ("bid", "ask") for p in ("L", "k", "x0")
    ])


def main() -> None:
    n_states = int(os.environ.get("P5_STATES", "6"))
    raw = build_states(n_states=n_states)
    active_states = [_with_global(st, active=True) for st, _ in raw]
    inactive_states = [_with_global(st, active=False) for st, _ in raw]

    opt = _opt()

    t_active = timeit(opt, active_states)
    t_inactive = timeit(opt, inactive_states)
    ms_active = statistics.median(t_active)
    ms_inactive = statistics.median(t_inactive)

    print(f"threads={opt.torch_threads}  corpus={len(active_states)} states\n")
    print(f"global_active=True   median {ms_active:8.2f} ms/decide")
    print(f"global_active=False  median {ms_inactive:8.2f} ms/decide")
    print(f"inactive / active    = {ms_inactive / ms_active:.3f}x")

    # Does masking the global chain change the local action? If it does, the
    # chain is not actually inert when inactive and skipping it is not safe.
    max_diff = 0.0
    for a_st, i_st in zip(active_states, inactive_states):
        a_act, _ = opt.optimize_with_diagnostics(a_st)
        a_inact, _ = opt.optimize_with_diagnostics(i_st)
        max_diff = max(max_diff, float(np.max(np.abs(_action_vec(a_act) - _action_vec(a_inact)))))

    print(f"\nmax |local action(active) - local action(inactive)| = {max_diff:.3e}")
    print("  -> identical means the global chain is inert when masked; a skip is bit-exact"
          if max_diff == 0.0 else
          "  -> NON-zero means the global chain still influences the local action; skipping is NOT exact")

    out = {
        "ms": {"global_active_true": ms_active, "global_active_false": ms_inactive},
        "ratio": ms_inactive / ms_active,
        "max_local_action_diff": max_diff,
        "samples": {"active": t_active, "inactive": t_inactive},
    }
    path = Path(__file__).resolve().parent / "out" / "p5_global_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()