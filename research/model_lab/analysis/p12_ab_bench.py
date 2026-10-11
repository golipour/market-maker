"""A/B the specialization-axis fixes: measure, do not assume.

Three changes were made to reduce dynamo specializations on the joint decide
path. Only one of them measurably removed specializations; the question this
answers is whether any of them COST runtime, because ``mark_dynamic`` blocks
constant folding and a signature keyed on more scalars means a coarser cache
key. So measure decide latency under each configuration.

Variants are produced by toggling module-level flags, not by editing files, so
every arm runs against identical Python bytecode and the same process.

Run: OMP_NUM_THREADS=2 python3 research/model_lab/analysis/p12_ab_bench.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "model_lab" / "analysis"))

import torch

from replay_states import build_states

import mm.mpc.mpc_tensor_batched as mpc_mod
import mm.mpc.joint_optimizer as joint_mod
from mm.mpc.mpc_state import OrderbookRelative
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched

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
        ones = torch.ones(shape, dtype=dt, device=device)
        return ones, ones.clone()


class _FixedPrice:
    def sample_residual(self, shape, device=None, dtype=None):
        dt = dtype or torch.float64
        z = torch.zeros(shape, dtype=dt, device=device)
        return z, z.clone()


class _Exec:
    maker_fee = 0.0
    local_maker_fee = 0.0
    global_maker_fee = 0.0


def _with_global(base):
    fields = dict(
        global_ob_rel=OrderbookRelative(
            bids=((2.0, 1.0), (4.0, 0.5)), asks=((2.0, 1.0), (4.0, 0.5))
        ),
        global_active=False,
        global_inventory=3.0,
        global_cash=500.0,
        global_buy_vwap=base.buy_vwap * 0.999,
        global_sell_vwap=base.sell_vwap * 1.001,
        reference_price=base.sell_vwap,
    )
    for key, value in fields.items():
        object.__setattr__(base, key, value)
    return base


def _build(compile_soft_fill):
    price, flow = _FixedPrice(), _FixedFlow()
    return MonteCarloMPCTensorBatched(
        local_price_model=price,
        local_market_flow_model=flow,
        execution_model=_Exec(),
        global_price_model=price,
        global_market_flow_model=flow,
        reference_price_source="global",
        global_optimizer={"n_components": 2},
        local_optimizer={"n_components": PROD["n_components"]},
        fixed_mc_samples=True,
        torch_threads=1,
        device="cpu",
        compile_soft_fill=compile_soft_fill,
        **PROD,
    )


def timeit(opt, states, warm=3, reps=9):
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


def _specs():
    try:
        from torch._C._dynamo.eval_frame import _debug_get_cache_entry_list

        return len(list(_debug_get_cache_entry_list(mpc_mod.soft_fill_batched.__code__)))
    except Exception:
        return -1


def main():
    n_states = int(os.environ.get("P12_STATES", "6"))
    raw = build_states(n_states=n_states)
    states = [_with_global(st) for st, _ in raw]

    if mpc_mod._COMPILE_SOFT_FILL_FAILED:
        print("inductor unavailable; cannot A/B the compiled path")
        return

    eager = _build(False)
    t_eager = timeit(eager, states)

    results = {"eager": t_eager}

    # Arm 1: all fixes (mark_dynamic active).
    compiled = _build(True)
    t_all = timeit(compiled, states)
    results["all fixes"] = t_all
    specs_all = _specs()

    # Arm 2: mark_dynamic disabled, everything else in place. Patch the symbol
    # the call sites resolve at call time. Absent on the baseline commit, in
    # which case there is nothing to disable and the arms coincide.
    real_mark = getattr(mpc_mod, "_mark_batch_dims_dynamic", None)
    if real_mark is not None:
        joint_mod.mark_batch_dims_dynamic = lambda *a, **k: None
        t_nomark = timeit(_build(True), states)
        joint_mod.mark_batch_dims_dynamic = real_mark
        results["no mark_dynamic"] = t_nomark
        specs_nomark = _specs()
    else:
        t_nomark = t_all
        specs_nomark = specs_all

    print(f"corpus={len(states)} states, PROD preset, single thread")
    print("")
    print(f"{'arm':<20} {'median':>9} {'min':>9} {'stdev':>8} {'vs eager':>10}")
    base = statistics.median(t_eager)
    for label, samples in results.items():
        med = statistics.median(samples)
        print(
            f"{label:<20} {med:>8.2f}ms {min(samples):>8.2f}ms "
            f"{statistics.stdev(samples):>7.2f} {med / base:>9.3f}x"
        )

    print("")
    print(f"soft_fill_batched specializations:")
    print(f"  with mark_dynamic    : {specs_all}")
    print(f"  without mark_dynamic : {specs_nomark}")
    print(f"  dynamo limit         : {torch._dynamo.config.recompile_limit}")

    best = min(
        (statistics.median(v), k) for k, v in results.items() if k != "eager"
    )
    print("")
    print(f"fastest compiled arm: {best[1]} at {best[0]:.2f} ms/decide")


if __name__ == "__main__":
    main()