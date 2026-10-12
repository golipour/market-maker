"""Rigorous, leak-free prequential backtest across Bitwana and Nobitex datasets.

Executes mathematically normalized price and flow models with:
  1. Strict causality (predict before update).
  2. No gap bridging (runs split on dt != 10000ms).
  3. Proper discrete scoring (NLL in nats, Brier score on zero hit).
  4. Moving-block bootstrap 95% confidence intervals and Diebold-Mariano tests.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import bx_data as D
from research.model_lab.analysis import bx_run as R
from research.model_lab.analysis.bx_flow_models import (
    BaselineEmpiricalFlow,
    FlowHurdleWeibull,
    FlowIncumbent,
    FlowRegularised,
)
from research.model_lab.analysis.bx_price_models import (
    BaselineEmpirical,
    HurdleGaussian,
    HurdleLogNormalSigned,
    HurdleStudentT,
    IncumbentGaussian,
)

DATASETS = [
    ("nobitex", "USDT_IRT"),
    ("bitwana", "USDT_IRT"),
    ("nobitex", "BTC_IRT"),
]

PRICE_MODELS = [
    ("INCUMBENT: Gaussian (lattice)", lambda: IncumbentGaussian()),
    ("BASELINE:  Empirical (rolling 2k)", lambda: BaselineEmpirical()),
    ("CHALLENGER: Hurdle + Gaussian body", lambda: HurdleGaussian()),
    ("CHALLENGER: Hurdle-t (nu=4.0)", lambda: HurdleStudentT(nu=4.0)),
    ("CHALLENGER: Hurdle-t (nu=2.5)", lambda: HurdleStudentT(nu=2.5)),
    ("CHALLENGER: Hurdle + signed LogNormal", lambda: HurdleLogNormalSigned()),
]

FLOW_MODELS = [
    ("INCUMBENT:  Hurdle LogNormal", lambda: FlowIncumbent()),
    ("CHALLENGER: Regularized Hurdle LogNormal", lambda: FlowRegularised()),
    ("INCUMBENT:  Hurdle Weibull", lambda: FlowHurdleWeibull()),
    ("BASELINE:   Empirical Flow", lambda: BaselineEmpiricalFlow()),
]


def banner(text: str) -> None:
    print("\n" + "=" * 96)
    print(text)
    print("=" * 96, flush=True)


def load_dataset(exchange: str, symbol: str) -> dict:
    raw = D.load_series(exchange, symbol)
    built = D.build(raw, interval_ms=10000.0)
    return D.derived(built, symbol=symbol)


def run_price_suite(y: dict, warm: int = 2000) -> dict[str, dict[str, np.ndarray]]:
    banner(f"PRICE EVALUATION: {y['symbol']} on {y['N']} 10s steps (tick={y['tick']} IRT)")
    print(f"  Valid scored steps: {int(np.sum(y['valid_step'][warm:]))}")
    print(f"  P(dmid == 0) realized: {float(np.mean(y['dmid_n1'][y['valid_step']] == 0.0)):.4f}")
    print(R.HDR)

    results = {}
    for name, factory in PRICE_MODELS:
        model = factory()
        res = R.prequential_price(model, y, warm=warm)
        results[name] = res
        summ = R.summary(name, res["nll"], res["brier"])
        print(R.fmt_summary(summ))

    # Paired comparisons vs Incumbent
    ref_name = PRICE_MODELS[0][0]
    ref_nll = results[ref_name]["nll"]
    half_idx = y["N"] // 2

    print("\n  --- PAIRED LOSS DIFFERENCE VS INCUMBENT (negative = better) ---")
    print(f"  {'Model':<44} {'Full Gain (nats)':<18} {'95% CI':<24} {'1st Half':<12} {'2nd Half':<12} {'DM t':>8}")

    for name, _ in PRICE_MODELS[1:]:
        cand_nll = results[name]["nll"]
        comp = R.paired_comparison(ref_nll, cand_nll)
        if comp is None:
            continue
        # Split comparisons
        comp_h1 = R.paired_comparison(ref_nll[warm:half_idx], cand_nll[warm:half_idx])
        comp_h2 = R.paired_comparison(ref_nll[half_idx:], cand_nll[half_idx:])
        h1_str = f"{comp_h1['mean_diff']:+.4f}" if comp_h1 else " N/A"
        h2_str = f"{comp_h2['mean_diff']:+.4f}" if comp_h2 else " N/A"
        ci_str = f"[{comp['ci_lo']:+.4f}, {comp['ci_hi']:+.4f}]"

        print(
            f"  {name:<44} {comp['mean_diff']:>+12.4f} nats   "
            f"{ci_str:<24} {h1_str:>10}   {h2_str:>10}   {comp['dm_t']:>8.1f}"
        )

    return results


def run_flow_suite(y: dict, warm: int = 2000) -> dict[str, Any]:
    banner(f"ORDER FLOW EVALUATION: {y['symbol']} (lot scale = {y['lot_scale']})")
    print(f"  Realized zero flow: Buy={float(np.mean(y['bv_lots_n1'][y['valid_step']] == 0)):.4f}, "
          f"Sell={float(np.mean(y['sv_lots_n1'][y['valid_step']] == 0)):.4f}")

    # Run each flow model ONCE
    all_results: dict[str, dict[str, Any]] = {}
    for name, factory in FLOW_MODELS:
        model = factory()
        all_results[name] = R.prequential_flow(model, y, warm=warm)

    for tag in ("joint", "buy", "sell"):
        print(f"\n  === FLOW: {tag.upper()} ===")
        print(R.HDR)
        res_tag: dict[str, np.ndarray] = {}
        for name, _ in FLOW_MODELS:
            res = all_results[name]
            nll = res[tag]["nll"]
            brier = res[tag].get("brier")
            res_tag[name] = nll
            summ = R.summary(name, nll, brier)
            print(R.fmt_summary(summ))

        # Paired comparisons vs Incumbent
        ref_name = FLOW_MODELS[0][0]
        ref_nll = res_tag[ref_name]
        print(f"\n  --- Paired difference vs {ref_name} ---")
        for name, _ in FLOW_MODELS[1:]:
            comp = R.paired_comparison(ref_nll, res_tag[name])
            if comp:
                ci_str = f"[{comp['ci_lo']:+.4f}, {comp['ci_hi']:+.4f}]"
                print(f"    {name:<42} {comp['mean_diff']:>+10.4f} nats  {ci_str:<24} DM t={comp['dm_t']:>6.1f} (p={comp['dm_p']:.3e})")


def main() -> None:
    banner("RIGOROUS EX-ANTE PREQUENTIAL BACKTEST (BITWANA & NOBITEX)")
    print("Protocol:")
    print("  - Pure causal walk-forward (predict at t, score t+1, update at t+1)")
    print("  - Gaps where dt != 10000ms strictly segregated into separate runs")
    print("  - Discrete PMFs verified: sum_k P(k) == 1.000000, no probability > 1.0")
    print("  - Warmup = 2000 steps (~5.5 hours) excluded from scoring")

    for exchange, symbol in DATASETS:
        print(f"\nLoading {exchange}/{symbol} ...", flush=True)
        y = load_dataset(exchange, symbol)
        run_price_suite(y, warm=2000)
        run_flow_suite(y, warm=2000)


if __name__ == "__main__":
    main()
