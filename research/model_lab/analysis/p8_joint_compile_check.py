"""Does the joint path now run the compiled kernel, and does it agree with eager?

The wiring in ``joint_optimizer.py`` routes both venue chains through
``resolve_compiled_soft_fill``. This checks the two things that matter:

  1. **Reachability** -- after construction, are the signatures the joint path
     actually requests present in ``_SOFT_FILL_WARMED_SIGNATURES``? If any are
     missing, the decide path is running eager and the whole compile stack is
     still idle.
  2. **Parity** -- do compiled and eager joint decisions agree? The compiled
     kernel reassociates float64 arithmetic, so exact equality is not expected;
     what must hold is that the actions agree to tight tolerance and that no
     non-finite values appear.

It also re-checks the thing that caused the original incident: a decide must
never request a compiled kernel for an unwarmed signature.

Run: OMP_NUM_THREADS=2 python3 research/model_lab/analysis/p8_joint_compile_check.py
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
import mm.mpc.mpc_tensor_batched as mpc_mod  # noqa: E402
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
        global_ob_rel=OrderbookRelative(bids=((2.0, 1.0), (4.0, 0.5)), asks=((2.0, 1.0), (4.0, 0.5))),
        global_active=active,
        global_inventory=3.0,
        global_cash=500.0,
        global_buy_vwap=base.buy_vwap * 0.999,
        global_sell_vwap=base.sell_vwap * 1.001,
        reference_price=base.sell_vwap,
    )
    for k, v in fields.items():
        object.__setattr__(base, k, v)
    return base


def _opt(*, compile_soft_fill: bool, variant: str = "a"):
    kw = dict(PROD)
    price, flow = _FixedPrice(), _FixedFlow()
    if variant == "b":
        # The second shape seen in production: grid_levels 20 (spread_step 10)
        # with a smaller sample count and component count.
        kw["spread_step"] = 10.0
        kw["n_samples"] = 16
        kw["n_components"] = 2
    n_local = kw["n_components"]
    return MonteCarloMPCTensorBatched(
        local_price_model=price,
        local_market_flow_model=flow,
        execution_model=_Exec(),
        global_price_model=price,
        global_market_flow_model=flow,
        reference_price_source="global",
        global_optimizer={"n_components": GLOBAL_N_COMPONENTS},
        local_optimizer={"n_components": n_local},
        fixed_mc_samples=True,
        torch_threads=int(os.environ.get("P8_THREADS", "1")),
        device="cpu",
        compile_soft_fill=compile_soft_fill,
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
    n_states = int(os.environ.get("P8_STATES", "6"))
    raw = build_states(n_states=n_states)
    # Production shape: global venue inactive (liquidity sourcing off).
    states = [_with_global(st, active=False) for st, _ in raw]

    if mpc_mod._COMPILE_SOFT_FILL_FAILED:
        print("inductor unavailable in this process; cannot verify the compiled path")
        return

    opt_c = _opt(compile_soft_fill=True)
    print(f"threads={opt_c.torch_threads}  corpus={len(states)} states")
    print(f"compile warm-up paid at construction: {opt_c.soft_fill_compile_warmup_ms:.0f} ms")
    print(f"warmed signatures: {len(mpc_mod._SOFT_FILL_WARMED_SIGNATURES)}")
    for sig in sorted(mpc_mod._SOFT_FILL_WARMED_SIGNATURES, key=str):
        print(f"  n_comp={sig[0]} n_samples={sig[1]} grid={sig[2]} "
              f"act={sig[3]} queue={sig[4]} depth_scale={sig[5]} "
              f"soft_temp={sig[6]} maker_fee={sig[7]}")

    # 1. Reachability: record what the decide path asks for.
    requested: list[dict] = []
    real_resolver = joint_mod.resolve_compiled_soft_fill

    def spy(torch_mod, **kw):
        requested.append(kw)
        return real_resolver(torch_mod, **kw)

    joint_mod.resolve_compiled_soft_fill = spy
    try:
        for st in states:
            opt_c.optimize_with_diagnostics(st)
    finally:
        joint_mod.resolve_compiled_soft_fill = real_resolver

    served_eager = {
        (kw["n_components"], float(kw.get("depth_scale", 1.0)))
        for kw in requested
        if real_resolver(
            torch,
            enabled=True,
            n_components=kw["n_components"],
            n_samples=kw["n_samples"],
            grid_levels=kw["grid_levels"],
            soft_fill_activation=kw["soft_fill_activation"],
            soft_fill_queue_model=kw["soft_fill_queue_model"],
            depth_scale=kw.get("depth_scale", 1.0),
            soft_temp=kw.get("soft_temp", 1.0),
            maker_fee=kw.get("maker_fee", 0.0),
        ) is mpc_mod.soft_fill_batched
    }
    print(f"\nsignatures the decide path asked for: "
          f"{sorted({(w['n_components'], float(w.get('depth_scale', 1.0))) for w in requested})}")
    print(f"served eager (unwarmed): {sorted(served_eager) if served_eager else 'none'}")

    # 2. Parity against eager.
    opt_e = _opt(compile_soft_fill=False)
    diffs, objs, nonfinite = [], [], 0
    for st in states:
        ac, dc = opt_c.optimize_with_diagnostics(st)
        ae, de = opt_e.optimize_with_diagnostics(st)
        vc, ve = _act_vec(ac), _act_vec(ae)
        diffs.append(float(np.max(np.abs(vc - ve))))
        objs.append(abs(float(dc.best.objective_value) - float(de.best.objective_value)))
        nonfinite += int(not np.all(np.isfinite(vc)))

    print(f"\nparity compiled vs eager over {len(states)} states:")
    print(f"  max |action diff|     = {max(diffs):.3e}")
    print(f"  max |objective diff|  = {max(objs):.3e}")
    print(f"  non-finite actions    = {nonfinite}")

    t_c = statistics.median(timeit(opt_c, states))
    t_e = statistics.median(timeit(opt_e, states))
    print(f"\neager   median {t_e:8.2f} ms/decide")
    print(f"compiled median {t_c:8.2f} ms/decide   speedup {t_e / t_c:.3f}x")

    # --- does the decide path settle, or add specializations forever? ---
    def _recompiles() -> int:
        return int(torch._dynamo.utils.counters.get("recompiles", 0) or 0)

    print("\nrecompiles across decides (joint path):")
    print(f"{'decide':>7} {'total':>8} {'delta':>6}")
    prev = _recompiles()
    series = []
    n_decides = int(os.environ.get("P8_DECIDES", "10"))
    for i in range(n_decides):
        for st in states:
            opt_c.optimize_with_diagnostics(st)
        now = _recompiles()
        series.append(now)
        print(f"{i + 1:>7} {now:>8} {now - prev:>6}")
        prev = now
    tail = series[-4:]
    print(f"last four decides: {tail}")

    # --- frame census: which code objects actually accumulate specializations ---
    # Reducing specialization axes by guesswork has been wrong twice today (a
    # ``side`` string branch that was not the frame hog, a stale reachability
    # probe), so measure per-code-object frame counts rather than assume.
    if os.environ.get("P8_CENSUS", "1") == "1":
        try:
            import torch._dynamo as dynamo

            print("")
            print(f"traced frames total: "
                  f"{dynamo.utils.counters['frames'].get('total', 0)}, "
                  f"recompiles: {dynamo.utils.counters.get('recompiles', 0)}")
            # Cache entries live in C++ on torch 2.x. ``_debug_get_cache_entry_list``
            # takes a code object and returns its compiled specializations, which is
            # how we name the worst axis instead of guessing at it.
            try:
                from torch._C._dynamo.eval_frame import _debug_get_cache_entry_list

                import mm.mpc.state_tensor as st_mod

                watched = {
                    "_materialize_side": st_mod._materialize_side,
                    "effective_depth_bid_tensor": st_mod.effective_depth_bid_tensor,
                    "effective_depth_ask_tensor": st_mod.effective_depth_ask_tensor,
                    "effective_depth_stacked_tensor": st_mod.effective_depth_stacked_tensor,
                    "soft_fill_batched": mpc_mod.soft_fill_batched,
                }
                rows = []
                for label, fn in watched.items():
                    try:
                        n = len(list(_debug_get_cache_entry_list(fn.__code__)))
                    except Exception:
                        n = -1
                    rows.append((n, label))
                print("  specializations per code object (limit is "
                      f"{torch._dynamo.config.recompile_limit}):")
                for n, label in sorted(rows, key=lambda r: -r[0]):
                    print(f"    {n:>3}  {label}")
            except Exception as inner:  # pragma: no cover - diagnostics only
                print(f"  census unavailable: {type(inner).__name__}: {inner}")
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"frame census skipped: {type(exc).__name__}: {exc}")

    settled = max(tail) - min(tail) == 0
    print("  -> fixed point reached; warm-up is bounded"
          if settled else "  -> STILL RECOMPILING late; each decide adds specializations")
    limit = torch._dynamo.config.recompile_limit
    print(f"  cache_size_limit={limit}, exceeded={max(series) > limit}")

    # Two shapes on one code object: the (64, grid10) preset that decides and
    # the (16, grid20) shape seen warming in the same pod. If alternating them
    # keeps adding frames, the "config is fixed" premise is false in practice.
    if os.environ.get("P8_TWO_PRESETS", "1") == "1":
        try:
            alt_a = _opt(compile_soft_fill=True, variant="a")
            alt_b = _opt(compile_soft_fill=True, variant="b")
            base = _recompiles()
            for _ in range(3):
                for st in states:
                    alt_a.optimize_with_diagnostics(st)
            alone = _recompiles() - base
            base2 = _recompiles()
            for i in range(4):
                opt = alt_a if i % 2 == 0 else alt_b
                for st in states:
                    opt.optimize_with_diagnostics(st)
            alternated = _recompiles() - base2
            print(f"\ntwo-shape check: single preset 3x -> +{alone} recompiles; "
                  f"alternating 4x -> +{alternated} recompiles")
            print("  -> bounded (each shape warms once)" if alternated <= alone + 4
                  else "  -> thrashing: each alternation adds specializations")
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"\ntwo-shape check skipped: {type(exc).__name__}: {exc}")

    # Two controllers in one process. Production is a multi-strategy worker:
    # worker_build_complete logs strategy_ids ["mpc-usdt-local","bitwana-sample"]
    # and actors ["bitwana-controller","mpc-controller"], and each warms its own
    # shapes. If the second controller's signature evicts the first from the
    # shared dynamo cache, the deciding controller silently drops to eager --
    # which is what the (n_components=2, n_samples=16, grid_levels=20) warm
    # event in the live pod turned out to be.
    if os.environ.get("P8_TWO_CONTROLLERS", "1") == "1":
        try:
            ctrl_a = _opt(compile_soft_fill=True, variant="a")
            ctrl_b = _opt(compile_soft_fill=True, variant="b")
            served = []

            real_get = mpc_mod._get_compiled_soft_fill

            def counting_get(torch_mod):
                served.append(1)
                return real_get(torch_mod)

            mpc_mod._get_compiled_soft_fill = counting_get
            try:
                for st in states:
                    ctrl_a.optimize_with_diagnostics(st)
                a_served = len(served)
                served.clear()
                for st in states:
                    ctrl_b.optimize_with_diagnostics(st)
                b_served = len(served)
            finally:
                mpc_mod._get_compiled_soft_fill = real_get

            print(f"\ntwo-controller check (one process, two presets):")
            print(f"  controller A served compiled kernel on {a_served}/{len(states)} decides")
            print(f"  controller B served compiled kernel on {b_served}/{len(states)} decides")
            print("  -> both still compiled" if a_served and b_served
                  else "  -> EVICTED: a controller fell back to eager")
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"\ntwo-controller check skipped: {type(exc).__name__}: {exc}")

    ok = not served_eager and nonfinite == 0 and max(diffs) < 1e-6 and settled
    print("\nVERDICT: " + ("joint path reaches the compiled kernel, agrees with eager, "
                          "and recompilation is bounded"
                          if ok else "check the numbers above"))


if __name__ == "__main__":
    main()