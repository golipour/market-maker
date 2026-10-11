"""Benchmark the *joint* optimizer path, which is what production runs.

The earlier perf study (``p1_perf_bench.py``) measured
``optimize_with_diagnostics`` with ``reference_price_source="local"`` and no
global venue, so it exercised the single-venue rollout in
``mm/mpc/mpc_tensor_batched.py``. Production dispatches to
``optimize_joint_with_diagnostics`` (``mm/mpc/joint_optimizer.py``) whenever
``reference_price_source == "global"`` -- the shipped default -- or liquidity
sourcing is on. That path:

  * evaluates **two** venue chains per horizon step (``soft_fill_batched`` is
    called for local and for global), and
  * never consults ``_get_compiled_soft_fill``, so ``compile_soft_fill`` and
    every optimization built on top of it is unreachable there.

So the published decide-time numbers describe a code path production does not
execute. This benchmark measures the path production actually runs, at the
shipped preset, so the remaining optimization targets are grounded.

Run: OMP_NUM_THREADS=2 python3 research/model_lab/analysis/p4_joint_bench.py
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


# Shipped preset (mm/configs/market_maker.runtime.example.json).
PROD = dict(
    n_components=4,
    n_samples=64,
    horizon=2,
    spread_max=200.0,
    spread_step=20.0,
    adam_lr=1.0,
    adam_epochs=20,
)
# Global venue uses its own component count (preset: optimizer.global).
GLOBAL_N_COMPONENTS = 2


class _FixedFlow:
    """Zero-variance flow: isolates cost from MC noise, matches p1 fixtures."""

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


def _joint_state(base: State) -> State:
    """Add an active global venue so the joint path does real work.

    Production frames carry a global book (``global_ob_rel``) and
    ``global_active``; without them ``global_mask == 0`` and the global chain
    degenerates. ``object.__setattr__`` mirrors the frozen dataclass construction
    used by the existing joint-MPC tests.
    """
    fields = dict(
        global_ob_rel=OrderbookRelative(bids=((2.0, 1.0), (4.0, 0.5)), asks=((2.0, 1.0), (4.0, 0.5))),
        global_active=True,
        global_inventory=3.0,
        global_cash=500.0,
        global_buy_vwap=base.buy_vwap * 0.999,
        global_sell_vwap=base.sell_vwap * 1.001,
        reference_price=base.sell_vwap,
    )
    for key, value in fields.items():
        object.__setattr__(base, key, value)
    return base


def _opt(*, joint: bool, **over):
    kw = {**PROD, **over}
    price = _FixedPrice()
    flow = _FixedFlow()
    return MonteCarloMPCTensorBatched(
        local_price_model=price,
        local_market_flow_model=flow,
        execution_model=_Exec(),
        global_price_model=price if joint else None,
        global_market_flow_model=flow if joint else None,
        # ``global`` is the shipped default and is what routes to the joint path.
        reference_price_source="global" if joint else "local",
        global_optimizer={"n_components": GLOBAL_N_COMPONENTS} if joint else None,
        local_optimizer={"n_components": PROD["n_components"]},
        fixed_mc_samples=True,
        torch_threads=int(os.environ.get("P4_THREADS", "1")),
        device="cpu",
        **kw,
    )


def timeit(opt, states, warm=3, reps=7):
    for _ in range(warm):
        for st in states:
            opt.optimize_with_diagnostics(st)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for st in states:
            opt.optimize_with_diagnostics(st)
        samples.append((time.perf_counter() - t0) * 1000.0 / len(states))
    return samples


def main() -> None:
    n_states = int(os.environ.get("P4_STATES", "12"))
    raw = build_states(n_states=n_states)
    states = [_joint_state(st) for st, _ in raw]

    print(f"joint preset: {PROD} global_n_components={GLOBAL_N_COMPONENTS}")
    print(f"threads={_opt(joint=False).torch_threads}  corpus={len(states)} states\n")

    out = {"production": PROD, "threads": os.environ.get("P4_THREADS", "1"), "n_states": len(states)}

    # Confirm the routing actually differs, so we are not timing the same path twice.
    single = _opt(joint=False)
    joint = _opt(joint=True)
    _, d_joint = joint.optimize_with_diagnostics(states[0])
    print(f"joint path taken: config.joint_venue={d_joint.config.get('joint_venue')} "
          f"global_active={d_joint.config.get('global_active')}\n")

    s_single = timeit(single, states)
    ms_single = statistics.median(s_single)
    print(f"single-venue (p1 baseline path)  median {ms_single:8.2f} ms/decide")

    s_joint = timeit(joint, states)
    ms_joint = statistics.median(s_joint)
    print(f"joint (production path)          median {ms_joint:8.2f} ms/decide")
    print(f"\njoint / single = {ms_joint / ms_single:.2f}x")

    # Where the joint time goes, by epoch count: linear in epochs means the
    # per-epoch rollout is the cost, superlinear would implicate setup.
    print("\nepoch scaling on the joint path (cost per epoch):")
    scaling = {}
    for epochs in (4, 8, 12, 20):
        e = timeit(_opt(joint=True, adam_epochs=epochs), states, warm=1, reps=3)
        med = statistics.median(e)
        scaling[epochs] = med
        print(f"  epochs={epochs:3d}  median {med:8.2f} ms  ({med / epochs:6.2f} ms/epoch)")

    # Decision parity: the joint path must not silently differ from the
    # single-venue one, but its objective should be comparable in magnitude.
    vecs = {}
    for name, o in (("single", single), ("joint", joint)):
        vv = []
        for st in states:
            a, d = o.optimize_with_diagnostics(st)
            vv.append(np.concatenate([
                np.atleast_1d(np.asarray(getattr(a, f"{s}_{p}"), dtype=float))
                for s in ("bid", "ask") for p in ("L", "k", "x0")
            ]))
        vecs[name] = np.array(vv)

    out.update(
        {
            "medians_ms": {"single_venue": ms_single, "joint": ms_joint},
            "joint_over_single": ms_joint / ms_single,
            "epoch_scaling_ms": scaling,
            "samples": {"single_venue": s_single, "joint": s_joint},
        }
    )
    path = Path(__file__).resolve().parent / "out" / "p4_joint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()