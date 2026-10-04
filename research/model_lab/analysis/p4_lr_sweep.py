"""Validate the lr=0.5 finding on a large, disjoint replay corpus.

The lr sweep in ``p3_lr_oscillation.py`` used 10 states starting at row 500.
This script re-runs the decisive comparison (lr=1.0 vs 0.5 vs 0.25) on a
much larger corpus drawn from a different, non-overlapping region of the
series, and reports both the objective quality and the stability metrics, so
the recommendation does not rest on a 10-state sample.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import replay_states  # noqa: E402

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


def _opt(epochs: int, lr: float):
    return MonteCarloMPCTensorBatched(
        local_price_model=_FixedPrice(),
        local_market_flow_model=_FixedFlow(),
        execution_model=_Exec(),
        n_components=4,
        n_samples=64,
        horizon=2,
        spread_max=200.0,
        spread_step=20.0,
        adam_lr=lr,
        reference_price_source="local",
        fixed_mc_samples=True,
        torch_threads=1,
        adam_epochs=epochs,
    )


def _vec(a):
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


def trace(st, lr, total_epochs, warm=3):
    opt = _opt(total_epochs, lr)
    for _ in range(warm):
        opt.optimize_with_diagnostics(st)
    opt.optimize_with_diagnostics(st)

    curve, vecs = [], []
    real_step = torch.optim.Adam.step

    def traced(self, closure=None):
        loss = real_step(self, closure)
        x = self.param_groups[0]["params"][0]
        n = 4
        sl = 3 * 2 * n + 3
        with torch.no_grad():
            flat = x.detach().cpu().numpy()[:sl].copy()
        try:
            a = opt._flat_to_effective_action(
                flat, mid_price=st.mid,
                bid_quote_size_min=None, bid_quote_size_max=None,
                ask_quote_size_min=None, ask_quote_size_max=None,
            )
            vecs.append(_vec(a))
        except Exception:
            vecs.append(np.full(6 * n, np.nan))
        curve.append(float(loss.detach().item()) if torch.is_tensor(loss) else float("nan"))
        return loss

    torch.optim.Adam.step = traced
    try:
        best_action, diag = opt.optimize_with_diagnostics(st)
    finally:
        torch.optim.Adam.step = real_step
    return np.array([-c for c in curve]), np.array(vecs), int(diag.best_epoch)


def main() -> None:
    n_states = int(os.environ.get("P4_STATES", "120"))
    start = int(os.environ.get("P4_START", "40000"))
    stride = int(os.environ.get("P4_STRIDE", "37"))
    total_epochs = int(os.environ.get("P4_EPOCHS", "40"))
    prod_epochs = int(os.environ.get("P4_PROD_EPOCHS", "20"))
    lrs = [float(x) for x in os.environ.get("P4_LRS", "1.0,0.5,0.25").split(",")]

    corpus = replay_states.build_states(n_states=n_states, start=start, stride=stride)
    print(f"corpus: {n_states} states from row {start}, stride {stride} (disjoint from p3)")
    print(f"epochs traced: {total_epochs}; production budget: {prod_epochs}\n")

    res = {lr: [] for lr in lrs}
    for k, (st, meta) in enumerate(corpus):
        for lr in lrs:
            j, vecs, be = trace(st, lr, total_epochs)
            res[lr].append(
                {
                    "J_peak": float(j.max()),
                    "J_at_prod": float(j.max()),
                    "J_best_in_window": float(j[:prod_epochs].max()),
                    "neg_frac": float((j < 0).mean()),
                    "max_step": float(np.max(np.abs(np.diff(j)))),
                    "best_epoch": be,
                    "vec_best": vecs[int(j.argmax())],
                    "vec_at_prod_best": vecs[int(np.argmax(j[:prod_epochs]))],
                }
            )
        if (k + 1) % 20 == 0:
            print(f"  ... {k + 1}/{n_states}", flush=True)

    base = lrs[0]
    summary = []
    for lr in lrs:
        R = res[lr]
        peak = np.array([r["J_peak"] for r in R])
        inwin = np.array([r["J_best_in_window"] for r in R])
        vf = np.array([r["vec_best"] for r in R])
        vw = np.array([r["vec_at_prod_best"] for r in R])
        dact = np.array([np.max(np.abs(a - b)) for a, b in zip(vf, vw)])
        summary.append(
            {
                "lr": lr,
                "mean_peak": float(peak.mean()),
                "mean_in_window": float(inwin.mean()),
                "loss_in_window": float(1.0 - inwin.mean() / peak.mean()),
                "neg_frac": float(np.mean([r["neg_frac"] for r in R])),
                "max_step": float(np.mean([r["max_step"] for r in R])),
                "median_best_epoch": float(np.median([r["best_epoch"] for r in R])),
                "median_dact": float(np.median(dact)),
                "p95_dact": float(np.percentile(dact, 95)),
            }
        )

    print(f"\n=== lr sweep on {n_states} disjoint real states, {prod_epochs}-epoch budget ===")
    print(
        f"{'lr':>6} {'mean peak J':>14} {'mean J in budget':>17} {'J lost':>9} "
        f"{'neg%':>6} {'max|dJ/de|':>11} {'med best ep':>12} {'med|dAct|':>10} {'p95|dAct|':>10}"
    )
    for s in summary:
        print(
            f"{s['lr']:6.3f} {s['mean_peak']:14.6e} {s['mean_in_window']:17.6e} "
            f"{100 * s['loss_in_window']:8.3f}% {100 * s['neg_frac']:6.1f} "
            f"{s['max_step']:11.3e} {s['median_best_epoch']:12.1f} "
            f"{s['median_dact']:10.4f} {s['p95_dact']:10.4f}"
        )

    bp = max(summary, key=lambda s: s["mean_peak"])
    print(f"\nbest mean peak objective: lr={bp['lr']} ({bp['mean_peak']:.6e})")
    for s in summary:
        print(f"  lr={s['lr']:5.3f}  {100 * s['mean_peak'] / bp['mean_peak']:6.2f}% of best peak")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "p4_lr_sweep.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {
                "config": {"n_states": n_states, "start": start, "stride": stride,
                           "epochs": total_epochs, "prod_epochs": prod_epochs},
                "summary": summary,
            },
            fh,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
