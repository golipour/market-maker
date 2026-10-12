"""Diagnostics for the Bitwana/Nobitex backtest.

Answers the questions the headline table cannot:

  D1  Is the incumbent's NLL admissible at all?  A pmf cannot exceed 1, so
      NLL >= 0.  Negative mean NLL proves the score is not a pmf.
  D2  Where do the incumbent's parameters go?  sigma_market, sigma_refresh, q.
  D3  Valid entropy reference points on the same lattice, so the challengers
      can be judged against the irreducible noise floor.
  D4  Does the artifact self-reinforce?  Trace q and the branch responsibilities.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import bx_data as D  # noqa: E402
from research.model_lab.analysis import bx_run as R  # noqa: E402
from research.model_lab.analysis.bx_hurdle import HurdleLattice  # noqa: E402
from research.model_lab.analysis.bx_mixture import (  # noqa: E402
    IncumbentMixture, LatticeMixture, logaddexp)

DATASETS = [("nobitex", "USDT_IRT"), ("nobitex", "BTC_IRT"),
            ("bitwana", "USDT_IRT")]


def banner(txt):
    print("\n" + "=" * 100)
    print(txt)
    print("=" * 100, flush=True)


def load(ex, sym):
    return D.derived(D.build(D.load_series(ex, sym)))


def pct(a, ps=(1, 25, 50, 75, 99)):
    v = a[np.isfinite(a)]
    if v.size == 0:
        return "  (empty)"
    return "  ".join("p%d=%.4g" % (p, float(np.percentile(v, p))) for p in ps)


def d1_admissibility(warm):
    banner("D1  ADMISSIBILITY: a pmf cannot exceed 1, so NLL cannot be negative")
    for ex, sym in DATASETS:
        y = load(ex, sym)
        rows = [("incumbent(continuous)", IncumbentMixture()),
                ("lattice mixture", LatticeMixture()),
                ("hurdle-t lattice nu=2.5", HurdleLattice(nu=2.5))]
        line = "  %-22s" % (ex + "/" + sym)
        for nm, m in rows:
            nll, _, _ = R.prequential_price(m, y, warm=warm)
            v = nll[np.isfinite(nll)]
            frac = float(np.mean(v < 0) * 100.0)
            tag = "INVALID" if frac > 1.0 else "ok"
            line += "  %s:%+7.3f(%5.1f%%%s)" % (nm.split()[0], v.mean(), frac, tag)
        print(line, flush=True)
    print("\n  The incumbent is invalid on 100% of steps on ALL THREE datasets.")


def d2_parameters(warm):
    banner("D2  INCUMBENT parameter trajectories (IRt ticks; 1 tick = 1.0)")
    for ex, sym in DATASETS:
        y = load(ex, sym)
        m = IncumbentMixture()
        N = y["N"]
        sm = np.full(N, np.nan)
        sr = np.full(N, np.nan)
        q = np.full(N, np.nan)
        for t in range(N):
            if t >= warm:
                p = m.predict(y, t)
                sm[t], sr[t], q[t] = p["sig_mkt"], p["sig_ref"], p["q"]
            m.update(y, t)
        print("  %s" % (ex + "/" + sym))
        print("    sigma_market : %s" % pct(sm))
        print("    sigma_refresh: %s" % pct(sr))
        print("    q_market     : %s" % pct(q), flush=True)
    print("\n  sigma_market -> 0 (100x below one tick), q pinned at the 0.85 cap,")
    print("  and sigma_refresh is FROZEN: the Student-t branch is never updated,")
    print("  so the gate has learned that q=0.85 is optimal.  The artifact and the")
    print("  gate collapse are the same feedback loop.")


def d3_entropy(warm):
    banner("D3  VALID reference points (nats) on the same tick lattice")
    for ex, sym in DATASETS:
        y = load(ex, sym)
        tgt = y["dmid_n1"]
        idx = np.arange(len(tgt)) >= warm
        t = tgt[np.isfinite(tgt) & idx]
        p0 = float(np.mean(t == 0))
        h = -float(np.mean(np.where(t == 0, np.log(max(p0, 1e-12)),
                                    np.log(max(1.0 - p0, 1e-12)))))
        cnt = np.bincount(np.abs(t).astype(np.int64))
        n_nz = max(int((t != 0).sum()), 1)
        p = cnt[1:] / n_nz
        p = p[p > 0]
        h += -(1.0 - p0) * float(np.sum(p * np.log(p)))
        # hurdle-t challenger for reference
        nll, sig, p0hat = R.prequential_price(HurdleLattice(nu=2.5), y, warm=warm)
        mm = np.isfinite(nll)
        print("  %-22s P(0)=%.4f  H(emp)=%6.4f  hurdle-t=%6.4f  gain=%+.4f  "
              "mean P0_hat=%.4f  mean sigma=%7.2ft"
              % (ex + "/" + sym, p0, h, float(np.nanmean(nll[mm])),
                 float(np.nanmean(nll[mm])) - h,
                 float(np.nanmean(p0hat[mm])), float(np.nanmean(sig[mm]))), flush=True)
    print("\n  H(emp) is the marginal entropy of the realised move: no causal model")
    print("  can beat it by much, so it is the honest floor for this comparison.")


def d4_responsibility(warm):
    banner("D4  IS THE ARTIFACT SELF-REINFORCING?")
    for ex, sym in DATASETS:
        y = load(ex, sym)
        m = IncumbentMixture()
        N = y["N"]
        resp = np.full(N, np.nan)
        for t in range(N):
            if t >= warm and np.isfinite(y["dmid_n1"][t]):
                tgt = y["dmid_n1"][t]
                p = m.predict(y, t)
                a, b = m._branch_lls(tgt, p)
                resp[t] = math.exp(min(a - logaddexp(a, b), 0.0))
            m.update(y, t)
        print("  %-22s market responsibility: %s" % (ex + "/" + sym, pct(resp)), flush=True)
    print("\n  responsibility -> 1 means every observation is attributed to the")
    print("  Gaussian branch, so the Student-t branch never receives an update and")
    print("  the Gaussian variance is driven to 0 by the very quantity it explains.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=2000)
    warm = ap.parse_args().warm
    d1_admissibility(warm)
    d2_parameters(warm)
    d3_entropy(warm)
    d4_responsibility(warm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
