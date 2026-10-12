"""Causal prequential evaluation runner for discrete price and flow models.

Integrity contract:
    1. Zero lookahead: predict(y, t) is evaluated before update(y, t).
    2. Zero gap leakage: only steps where run[t+1] == run[t] are scored.
    3. Proper scoring rules: discrete NLL in nats, Brier score on zero hit.
    4. Strict normalization: assert log P <= 0.0 (no probability > 1.0).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from research.model_lab.analysis.stats1 import dm_test, moving_block_ci


def prequential_price(model: Any, y: dict, *, warm: int = 2000) -> dict[str, np.ndarray]:
    """Score next-step mid move prequentially."""
    N = y["N"]
    nll = np.full(N, np.nan)
    brier = np.full(N, np.nan)
    p0_arr = np.full(N, np.nan)
    valid_step = y["valid_step"]

    bind = getattr(model, "bind", None)
    if bind is not None:
        bind(y)

    # Unit check normalization on initial state
    init_p = model.predict(y, 0)
    check_norm = getattr(model, "assert_normalized", None)
    if check_norm is not None:
        check_norm(init_p)

    for t in range(N):
        if t >= warm and valid_step[t]:
            target = y["dmid_n1"][t]
            p = model.predict(y, t)
            ll = model.logpmf(target, p)
            assert ll <= 1e-7, f"Illegal probability > 1: log P = {ll:.6f} at t={t} in {model.name}"
            nll[t] = -ll
            p0 = p.get("p0", np.nan)
            p0_arr[t] = p0
            if math.isfinite(p0):
                is_zero = 1.0 if target == 0.0 else 0.0
                brier[t] = (p0 - is_zero) ** 2
        model.update(y, t)

    return {"nll": nll, "brier": brier, "p0": p0_arr}


def prequential_flow(model: Any, y: dict, *, warm: int = 2000) -> dict[str, dict[str, np.ndarray]]:
    """Score buy and sell lot counts prequentially."""
    N = y["N"]
    buy_nll = np.full(N, np.nan)
    sell_nll = np.full(N, np.nan)
    joint_nll = np.full(N, np.nan)
    buy_brier = np.full(N, np.nan)
    sell_brier = np.full(N, np.nan)
    valid_step = y["valid_step"]

    # Verify initial normalization
    check_norm = getattr(model, "assert_normalized", None)
    if check_norm is not None:
        check_norm(model.predict("buy"))
        check_norm(model.predict("sell"))

    for t in range(N):
        kb = y["bv_lots_n1"][t]
        ks = y["sv_lots_n1"][t]
        if t >= warm and valid_step[t]:
            pb = model.predict("buy")
            ps = model.predict("sell")
            lb = model.logpmf(int(kb), pb)
            ls = model.logpmf(int(ks), ps)
            assert lb <= 1e-7 and ls <= 1e-7, f"Illegal flow prob > 1 in {model.name}"
            buy_nll[t] = -lb
            sell_nll[t] = -ls
            joint_nll[t] = -(lb + ls)
            buy_brier[t] = (pb["p0"] - (1.0 if kb == 0 else 0.0)) ** 2
            sell_brier[t] = (ps["p0"] - (1.0 if ks == 0 else 0.0)) ** 2

        # Update model using realized target
        if valid_step[t]:
            model.update_side(int(kb), "buy")
            model.update_side(int(ks), "sell")

    return {
        "buy": {"nll": buy_nll, "brier": buy_brier},
        "sell": {"nll": sell_nll, "brier": sell_brier},
        "joint": {"nll": joint_nll},
    }


def summary(name: str, nll: np.ndarray, brier: np.ndarray | None = None) -> dict[str, Any]:
    v = nll[np.isfinite(nll)]
    if v.size == 0:
        return {"name": name, "n": 0}
    b = brier[np.isfinite(brier)] if brier is not None else np.array([])
    return {
        "name": name,
        "n": int(v.size),
        "mean_nll": float(v.mean()),
        "med_nll": float(np.median(v)),
        "p90_nll": float(np.percentile(v, 90)),
        "p99_nll": float(np.percentile(v, 99)),
        "mean_brier": float(b.mean()) if b.size > 0 else np.nan,
    }


def paired_comparison(base_nll: np.ndarray, cand_nll: np.ndarray) -> dict[str, Any] | None:
    mask = np.isfinite(base_nll) & np.isfinite(cand_nll)
    diff = cand_nll[mask] - base_nll[mask]
    if diff.size < 50:
        return None
    mean, lo, hi, n = moving_block_ci(diff, B=2000, seed=42)
    t_stat, p_val = dm_test(diff)
    return {
        "mean_diff": float(mean),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "dm_t": float(t_stat),
        "dm_p": float(p_val),
        "n": int(n),
    }


HDR = (
    f"  {'Model':<44} {'Mean NLL':>10} {'Med NLL':>9} {'p99 NLL':>10} "
    f"{'Brier(0)':>10} {'N':>8}"
)


def fmt_summary(s: dict[str, Any]) -> str:
    if not s.get("n"):
        return f"  {s['name']:<44} (no scored data)"
    brier_str = f"{s['mean_brier']:>10.4f}" if math.isfinite(s.get("mean_brier", np.nan)) else "       N/A"
    return (
        f"  {s['name']:<44} {s['mean_nll']:>10.4f} {s['med_nll']:>9.4f} "
        f"{s['p99_nll']:>10.3f} {brier_str} {s['n']:>8}"
    )
