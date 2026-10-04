"""Re-run the lr / early-stopping study WITH production warm start.

Earlier sweeps used the default WarmStartParams(enabled=False), i.e. a cold
start on every decide. The shipped preset is warm: enabled, alpha 0.8, fixed
mode, preserve_optimizer_state=true.

That changes the problem in three ways that matter here:

1. The epoch-0 parameter is a blend of the previous decision's action and the
   symmetric default, so the optimizer no longer starts from a fixed point.
   Convergence statistics computed cold are not transferable.
2. preserve_optimizer_state=true keeps Adam's first and second moments across
   decides, so consecutive decisions are coupled: an oscillation on decide k
   seeds the trajectory on decide k+1.
3. Real decisions arrive in a sequence, one per interval, each warm started
   from the last. A cold-start corpus replays one state repeatedly, which is
   not what production does.

So the correct measurement replays the corpus as a time-ordered sequence with
a single optimizer instance, exactly as the live loop does.
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
from mm.mpc.warm_start import WarmStartParams  # noqa: E402


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


PROD_WARM = dict(enabled=True, alpha=0.8, alpha_mode="fixed", preserve_optimizer_state=True)
COLD_WARM = dict(enabled=False)


def _opt(epochs, lr, warm):
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
        warm_start=WarmStartParams.from_mapping(warm),
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


def trace_sequential(corpus, lr, warm, total_epochs):
    """Replay the corpus as a live decision sequence with one optimizer.

    A single optimizer instance is carried across states, so warm start and
    the preserved Adam moments behave exactly as in production. The per-epoch
    trace is captured from inside the real Adam loop.
    """
    opt = _opt(total_epochs, lr, warm)
    st_box = {"st": None}
    curve_box = {"curve": [], "vecs": []}
    real_step = torch.optim.Adam.step
    n = 4
    sl = 3 * 2 * n + 3

    def traced(self, closure=None):
        loss = real_step(self, closure)
        x = self.param_groups[0]["params"][0]
        with torch.no_grad():
            flat = x.detach().cpu().numpy()[:sl].copy()
        st = st_box["st"]
        try:
            a = opt._flat_to_effective_action(
                flat,
                mid_price=st.mid,
                bid_quote_size_min=None,
                bid_quote_size_max=None,
                ask_quote_size_min=None,
                ask_quote_size_max=None,
            )
            curve_box["vecs"].append(_vec(a))
        except Exception:
            curve_box["vecs"].append(np.full(6 * n, np.nan))
        curve_box["curve"].append(
            float(loss.detach().item()) if torch.is_tensor(loss) else float("nan")
        )
        return loss

    torch.optim.Adam.step = traced
    rows = []
    try:
        for k, (st, meta) in enumerate(corpus):
            curve_box["curve"] = []
            curve_box["vecs"] = []
            st_box["st"] = st
            _a, diag = opt.optimize_with_diagnostics(st)
            j = np.array([-c for c in curve_box["curve"]])
            rows.append(
                {
                    "i": k,
                    "index": meta["index"],
                    "spread": meta["spread"],
                    "J_best": float(diag.best.objective_value),
                    "best_epoch": int(diag.best_epoch),
                    "curve": j.tolist(),
                }
            )
    finally:
        torch.optim.Adam.step = real_step
    return rows


def main() -> None:
    n_states = int(os.environ.get("P5_STATES", "60"))
    start = int(os.environ.get("P5_START", "40000"))
    stride = int(os.environ.get("P5_STRIDE", "37"))
    total_epochs = int(os.environ.get("P5_EPOCHS", "40"))
    prod_epochs = int(os.environ.get("P5_PROD_EPOCHS", "20"))
    lrs = [float(x) for x in os.environ.get("P5_LRS", "1.0,0.5,0.25").split(",")]

    corpus = replay_states.build_states(n_states=n_states, start=start, stride=stride)
    print(f"corpus: {n_states} states from row {start}, stride {stride}")
    print("sequential replay (one optimizer, warm start carried)")
    print(f"traced {total_epochs} epochs; production budget {prod_epochs}\n")

    all_rows = {}
    for label, warm in [("cold", COLD_WARM), ("warm", PROD_WARM)]:
        for lr in lrs:
            key = f"{label}_lr{lr}"
            all_rows[key] = trace_sequential(corpus, lr, warm, total_epochs)
            print(f"  ran {key}", flush=True)

    print(f"\n=== objective quality, {prod_epochs}-epoch production budget ===")
    print(f"{'config':<14} {'mean J_best':>14} {'neg%':>7} {'max|dJ/de|':>11} "
          f"{'med best ep':>12} {'J in budget':>13}")
    summary = []
    for key, rows in all_rows.items():
        jb = np.array([r["J_best"] for r in rows])
        negs, steps, bes, inwin = [], [], [], []
        for r in rows:
            c = np.array(r["curve"])
            if c.size == 0:
                continue
            negs.append(float((c < 0).mean()))
            steps.append(float(np.max(np.abs(np.diff(c)))))
            bes.append(r["best_epoch"])
            inwin.append(float(c[: prod_epochs + 1].max()))
        row = {
            "key": key,
            "mean_J_best": float(jb.mean()),
            "neg_frac": float(np.mean(negs)) if negs else float("nan"),
            "max_step": float(np.mean(steps)) if steps else float("nan"),
            "median_best_epoch": float(np.median(bes)) if bes else float("nan"),
            "mean_J_in_budget": float(np.mean(inwin)) if inwin else float("nan"),
        }
        summary.append(row)
        print(
            f"{key:<14} {row['mean_J_best']:14.6e} {100 * row['neg_frac']:6.1f}% "
            f"{row['max_step']:11.3e} {row['median_best_epoch']:12.1f} "
            f"{row['mean_J_in_budget']:13.6e}"
        )

    print("\n=== warm vs cold at the same lr ===")
    for lr in lrs:
        w = next((s for s in summary if s["key"] == f"warm_lr{lr}"), None)
        c = next((s for s in summary if s["key"] == f"cold_lr{lr}"), None)
        if not w or not c:
            continue
        print(
            f"  lr={lr:5.3f}  warm J={w['mean_J_best']:.6e}  cold J={c['mean_J_best']:.6e}  "
            f"warm/cold={w['mean_J_best'] / c['mean_J_best']:.4f}  "
            f"warm neg%={100 * w['neg_frac']:5.1f}  cold neg%={100 * c['neg_frac']:5.1f}"
        )

    print("\n=== lr effect within each warm mode ===")
    for label in ("cold", "warm"):
        sub = [s for s in summary if s["key"].startswith(label + "_")]
        if not sub:
            continue
        best = max(sub, key=lambda s: s["mean_J_best"])
        print(f"  {label}: best lr={best['key'].split('lr')[-1]} ({best['mean_J_best']:.6e})")
        for s in sub:
            lost = 1 - s["mean_J_in_budget"] / s["mean_J_best"] if s["mean_J_best"] else float("nan")
            print(
                f"    lr={s['key'].split('lr')[-1]:>5}  "
                f"{100 * s['mean_J_best'] / best['mean_J_best']:6.2f}% of best  "
                f"neg%={100 * s['neg_frac']:5.1f}  J lost to budget={100 * lost:6.3f}%"
            )

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "p5_warmstart.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {
                "config": {
                    "n_states": n_states,
                    "start": start,
                    "stride": stride,
                    "epochs": total_epochs,
                    "prod_epochs": prod_epochs,
                    "prod_warm": PROD_WARM,
                },
                "summary": summary,
            },
            fh,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
