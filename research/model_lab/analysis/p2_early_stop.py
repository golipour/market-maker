"""Re-measure early stopping on the symmetry-broken (well-conditioned) baseline.

Why this script exists
----------------------
Early stopping was previously REJECTED, but that measurement was taken while
every sigmoid component was initialized at the same depth. That makes the
objective flat along a collapsed direction, so "best epoch" carried almost no
information about where the search was heading. Option A removed the flat
manifold, so the verdict has to be re-measured.

Measurement contract (what was wrong before)
--------------------------------------------
An early-stopping rule is a *prefix* of ONE optimization run. Two approaches
were tried here first and both were wrong:

1. Comparing ``optimize(adam_epochs=E)`` against ``optimize(adam_epochs=40)``
   compares two separate runs, not a prefix with the whole.
2. Driving the trace by repeatedly calling ``optimize_with_diagnostics`` with
   ``adam_epochs=1`` does not work: the method rebuilds ``x_torch.data`` from
   the warm-start initial point on entry, so every call restarts from the same
   place and the parameter never advances.

The approach that does work is to observe a single real run from outside by
wrapping the Adam optimizer's ``step``. The closure runs inside ``step`` and
returns that epoch's loss, so the wrapper sees the objective for every epoch
of one continuous run, in order, with the true parameter trajectory. Decoding
the action at each epoch uses the production decoder on the live parameter, so
the trace is the run's own path and no production code is modified.
"""

from __future__ import annotations

import json
import os
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


def _make_opt(epochs: int, **kw) -> MonteCarloMPCTensorBatched:
    return MonteCarloMPCTensorBatched(
        local_price_model=_FixedPrice(),
        local_market_flow_model=_FixedFlow(),
        execution_model=_Exec(),
        reference_price_source="local",
        fixed_mc_samples=True,
        torch_threads=1,
        adam_epochs=epochs,
        **kw,
    )


def _vec(a) -> np.ndarray:
    return np.concatenate(
        [
            np.atleast_1d(np.asarray(a.bid_L, dtype=float)),
            np.atleast_1d(np.asarray(a.bid_k, dtype=float)),
            np.atleast_1d(np.asarray(a.bid_x0, dtype=float)),
            np.atleast_1d(np.asarray(a.ask_L, dtype=float)),
            np.atleast_1d(np.asarray(a.ask_k, dtype=float)),
            np.atleast_1d(np.asarray(a.ask_x0, dtype=float)),
        ]
    )


def trace_run(st, base_kw, total_epochs: int, warm_calls: int):
    """Return (curve, diagnostics, wall_ms) for ONE continuous run.

    ``curve[e]`` holds that epoch's objective and the action decoded from the
    parameter at that epoch, taken from inside the real Adam loop.

    The patch is installed on the ``torch.optim.Adam`` *class*, not on an
    instance, because ``optimize_with_diagnostics`` constructs a fresh Adam on
    entry, so an instance-level wrapper would be discarded before the loop ran.
    """
    opt = _make_opt(total_epochs, **base_kw)
    for _ in range(warm_calls):
        opt.optimize_with_diagnostics(st)

    n = opt.n_components
    single_len = 3 * 2 * n + 3
    curve: list[dict] = []

    real_step = torch.optim.Adam.step

    def traced_step(self, closure=None):  # noqa: ANN001
        loss = real_step(self, closure)
        try:
            x = self.param_groups[0]["params"][0]
        except Exception:
            x = None
        vec = np.full(single_len, np.nan, dtype=float)
        if x is not None:
            with torch.no_grad():
                flat = x.detach().cpu().numpy()[:single_len].copy()
            try:
                act = opt._flat_to_effective_action(
                    flat,
                    mid_price=st.mid,
                    bid_quote_size_min=None,
                    bid_quote_size_max=None,
                    ask_quote_size_min=None,
                    ask_quote_size_max=None,
                )
                vec = _vec(act)
            except Exception:
                pass
        j = float(loss.detach().item()) if torch.is_tensor(loss) else float("nan")
        curve.append({"J_loss": -j, "vec": vec, "loss": j})
        return loss

    torch.optim.Adam.step = traced_step
    try:
        t0 = time.perf_counter()
        best_action, diag = opt.optimize_with_diagnostics(st)
        wall_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        torch.optim.Adam.step = real_step

    return curve, best_action, diag, wall_ms


def main() -> None:
    n_states = int(os.environ.get("P2_STATES", "40"))
    total_epochs = int(os.environ.get("P2_EPOCHS", "40"))
    n_components = int(os.environ.get("P2_NCOMP", "4"))
    n_samples = int(os.environ.get("P2_NSAMP", "64"))
    horizon = int(os.environ.get("P2_HORIZON", "2"))
    adam_lr = float(os.environ.get("P2_LR", "1.0"))
    spread_max = float(os.environ.get("P2_SPREADMAX", "200.0"))
    spread_step = float(os.environ.get("P2_SPREADSTEP", "20.0"))
    warm_calls = int(os.environ.get("P2_WARM", "3"))

    corpus = build_states(n_states=n_states)
    base_kw = dict(
        n_components=n_components,
        n_samples=n_samples,
        horizon=horizon,
        spread_max=spread_max,
        spread_step=spread_step,
        adam_lr=adam_lr,
    )

    rows = []
    for k, (st, meta) in enumerate(corpus):
        curve, best_action, diag, wall_ms = trace_run(st, base_kw, total_epochs, warm_calls)
        if len(curve) != total_epochs:
            print(f"[{k:3d}] WARNING traced {len(curve)} epochs, expected {total_epochs}")

        j = np.array([c["J_loss"] for c in curve])
        vecs = np.array([c["vec"] for c in curve])
        finite = np.isfinite(j)
        if not finite.all():
            print(f"[{k:3d}] WARNING non-finite objective at {int((~finite).sum())} epochs")

        j_full = float(j[finite].max()) if finite.any() else float("nan")
        imax = int(np.nanargmax(np.where(finite, j, -np.inf)))
        d_act = np.array([float(np.max(np.abs(v - vecs[imax]))) for v in vecs])

        rows.append(
            {
                "i": k,
                "index": meta["index"],
                "spread": meta["spread"],
                "j_full": j_full,
                "argmax_epoch": imax,
                "best_epoch_reported": int(diag.best_epoch),
                "j_reported": float(diag.best.objective_value),
                "wall_ms": wall_ms,
                "curve_j": j.tolist(),
                "curve_dact": d_act.tolist(),
            }
        )
        e20 = min(20, len(d_act) - 1)
        print(
            f"[{k:3d}] idx={meta['index']:6d} spread={meta['spread']:7.1f} "
            f"argmax_ep={imax:3d} (reported {int(diag.best_epoch):3d}) "
            f"J_scan={j_full: .6e} J_rep={float(diag.best.objective_value): .6e} "
            f"dAct@20={d_act[e20]:9.3f}  {wall_ms:6.1f}ms",
            flush=True,
        )

    print("\n=== where does the peak objective land? (one continuous run) ===")
    am = np.array([r["argmax_epoch"] for r in rows])
    for lo, hi in [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, total_epochs - 1)]:
        cnt = int(((am >= lo) & (am <= hi)).sum())
        print(f"  epochs {lo:2d}-{hi:2d}: {cnt:3d}/{len(rows)} ({100.0 * cnt / max(len(rows), 1):5.1f}%)")
    print(f"  median={np.median(am):.1f} mean={am.mean():.1f}")

    print("\n=== early-stop cost at epoch E (best-so-far prefix vs run's eventual best) ===")
    print("    ret   = J(best-so-far through E) / J(best over the whole run)")
    print("    dAct  = |action(best-so-far through E) - action(best over the whole run)|")
    print(f"{'E':>4} {'mean ret':>10} {'min ret':>10} {'#ret<0.99':>10} "
          f"{'med|dAct|':>11} {'p95|dAct|':>11} {'max|dAct|':>11} {'%time':>7}")
    stats = []
    for E in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, total_epochs - 1]:
        E = min(E, total_epochs - 1)
        rets, dacts = [], []
        for r in rows:
            cj = np.array(r["curve_j"])
            cda = np.array(r["curve_dact"])
            j_best = float(cj.max())
            # An early stop returns the best-so-far snapshot, so both the
            # objective and the action must come from the SAME epoch: the
            # best epoch within the prefix.
            i_prefix_best = int(np.argmax(cj[: E + 1]))
            rets.append(float(cj[i_prefix_best]) / j_best if j_best > 0 else 1.0)
            dacts.append(float(cda[i_prefix_best]))
        rets_a = np.array(rets)
        dacts_a = np.array(dacts)
        n99 = int((rets_a < 0.99).sum())
        print(
            f"{E:4d} {rets_a.mean():10.6f} {rets_a.min():10.6f} {n99:10d} "
            f"{np.median(dacts_a):11.4f} {np.percentile(dacts_a, 95):11.4f} "
            f"{dacts_a.max():11.4f} {100.0 * (E + 1) / total_epochs:6.1f}%"
        )
        stats.append(
            {
                "E": E,
                "mean_ret": float(rets_a.mean()),
                "min_ret": float(rets_a.min()),
                "n_below_099": n99,
                "median_d_act": float(np.median(dacts_a)),
                "p95_d_act": float(np.percentile(dacts_a, 95)),
                "max_d_act": float(dacts_a.max()),
                "pct_time": float(100.0 * (E + 1) / total_epochs),
            }
        )

    mean_ms = float(np.mean([r["wall_ms"] for r in rows]))
    print(f"\nmean decide for {total_epochs} epochs: {mean_ms:.1f}ms")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "p2_early_stop.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {
                "config": {**base_kw, "total_epochs": total_epochs, "n_states": n_states},
                "mean_decide_ms": mean_ms,
                "stats": stats,
                "rows": rows,
            },
            fh,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
