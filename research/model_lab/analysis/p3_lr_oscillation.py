"""Why does early stopping fail? Because the optimizer is in a limit cycle.

The per-epoch trace at the production preset shows the objective rising, then
collapsing to a large negative value, then recovering, repeating with a period
of roughly 20-25 epochs. The "best epoch" is therefore a phase of that cycle,
not a convergence point, which is exactly the situation where truncating the
run is unsafe: the cutoff lands on an arbitrary phase.

This script tests whether the cycle is caused by the learning rate. If it is,
the fix is not early stopping but damping the step size, which is both cheaper
and better than cutting the run short.
"""

from __future__ import annotations

import json
import os
import sys

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


def _opt(epochs: int, lr: float, **over):
    kw = dict(
        n_components=4,
        n_samples=64,
        horizon=2,
        spread_max=200.0,
        spread_step=20.0,
        adam_lr=lr,
    )
    kw.update(over)
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


def trace(st, base, total_epochs, warm=3):
    opt = _opt(total_epochs, base["adam_lr"], **{k: v for k, v in base.items() if k != "adam_lr"})
    for _ in range(warm):
        opt.optimize_with_diagnostics(st)
    opt.optimize_with_diagnostics(st)

    curve = []
    real_step = torch.optim.Adam.step

    def traced(self, closure=None):
        loss = real_step(self, closure)
        curve.append(float(loss.detach().item()) if torch.is_tensor(loss) else float("nan"))
        return loss

    torch.optim.Adam.step = traced
    try:
        _a, diag = opt.optimize_with_diagnostics(st)
    finally:
        torch.optim.Adam.step = real_step
    return np.array([-c for c in curve]), int(diag.best_epoch)


def main() -> None:
    lrs = [float(x) for x in os.environ.get("P3_LRS", "1.0,0.5,0.25,0.1,0.05,0.02,0.01").split(",")]
    n_states = int(os.environ.get("P3_STATES", "10"))
    total_epochs = int(os.environ.get("P3_EPOCHS", "40"))
    corpus = build_states(n_states=n_states)

    base = dict(adam_lr=1.0)
    print(f"corpus: {n_states} real states, {total_epochs} epochs per run\n")

    summary = []
    for lr in lrs:
        neg_frac, osc, bests, finals, peaks, drops = [], [], [], [], [], []
        for st, meta in corpus:
            j, be = trace(st, {**base, "adam_lr": lr}, total_epochs)
            neg_frac.append(float((j < 0).mean()))
            osc.append(float(j.max() - j.min()))
            bests.append(float(j.max()))
            finals.append(float(j[min(19, len(j) - 1)]))
            peaks.append(be)
            # largest single-epoch drop: a proxy for a discontinuity
            drops.append(float(np.max(np.abs(np.diff(j)))))
        row = {
            "lr": lr,
            "neg_frac": float(np.mean(neg_frac)),
            "range": float(np.mean(osc)),
            "peak_J": float(np.mean(bests)),
            "J_at_20": float(np.mean(finals)),
            "peak_over_20": float(np.mean(bests) / max(np.mean(finals), 1e-30)),
            "max_step": float(np.mean(drops)),
            "median_best_epoch": float(np.median(peaks)),
        }
        summary.append(row)
        print(
            f"lr={lr:5.3f}  neg%={100 * row['neg_frac']:5.1f}  "
            f"range={row['range']:.3e}  max|dJ/de|={row['max_step']:.3e}  "
            f"peakJ={row['peak_J']:.6e}  J@20={row['J_at_20']:.6e}  "
            f"peak/J@20={row['peak_over_20']:6.3f}  med_best_ep={row['median_best_epoch']:4.1f}",
            flush=True,
        )

    print("\nInterpretation:")
    print("  neg%  -> fraction of epochs spent in the heavily penalized region")
    print("  range -> spread of J across the run; large = oscillating, not converging")
    print("  peak/J@20 -> how much better the best epoch is than epoch 20")
    print("           ~1.0 means the run has effectively converged by epoch 20")
    print("           >>1.0 means epoch 20 is far from the best the run ever sees")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "p3_lr.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"summary": summary, "config": {"n_states": n_states, "epochs": total_epochs}}, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
