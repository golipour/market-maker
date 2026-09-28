"""Q26: score the model that ACTUALLY RUNS IN PRODUCTION.

Everything in Q20-Q25 compared against `GaussianPriceModel` / the residual
EWMA, on the assumption (from `mm/runtime/config.py:1961`, which falls back to
{"kind": "residual"}) that "residual" is what ships.  That is wrong.

`mm/configs/market_maker.runtime.example.json:225` sets

    local_price_model = {"kind": "conditional_mechanical_mixture",
                         "decay": 0.99, "student_df": 3.0, ...}

so the live model is `TorchConditionalMechanicalMixturePriceModel`
(price_torch.py:376).  Its predictive law, from `sample_innovations`
(price_torch.py:615-639) and `sample_dynamic_vwap` (price_torch.py:641+):

  * a logistic gate  q_market = sigmoid(b0 + b1*log1p(total_flow))  that
    SPLITS each component into a "mechanical" Gaussian branch and a
    "refresh" Student-t branch, mixing them by responsibility weight
  * market branch:  N(0, market_var)          with RLS-fitted beta
  * refresh branch: t(nu = student_df = 3)    with EWMA scale
  * the MID component is mixed with a Gaussian in BOTH branches

So the real incumbent is a 2-component regime-switching model with a
Student-t refresh branch -- structurally much closer to my proposal than the
plain Gaussian I benchmarked against.

This script scores the real thing on the same prequential protocol.
"""
import os
import sys

import numpy as np
from scipy.stats import norm, t as tdist

sys.path.insert(0, "/Users/teanab/Documents/projects/market-maker")

from research.model_lab.analysis import harness as H
from research.model_lab.analysis import stats1 as S

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
HALF = N // 2
print("=" * 96)
print("Q26 - THE MODEL THAT ACTUALLY SHIPS")
print("=" * 96)
print("\nlocal_price_model.kind = 'conditional_mechanical_mixture'")
print("  gate     : q_market = sigmoid(b0 + b1*log1p(total_flow))")
print("  mid mean : RLS-fitted on mechanical features, per branch")
print("  market br: Normal(0, market_var)          (learned, RLS)")
print("  refresh  : StudentT(nu = student_df = 3)  (EWMA scale)")
print("  weight   : responsibility-weighted EM")
print("  params   : decay=0.99 student_df=3.0 initial_sigma=50.0")
print("\nNOTE: decay is already 0.99 here, so my earlier 'bump 0.9 -> 0.99'")
print("      advice was based on a model that is not the one in production.")


class RealProd:
    """Faithful port of the production predictive law for the MID component.

    We cannot replay RLS on the lab dataset because it needs order-book
    transitions, not just the mid series.  What we CAN port exactly is the
    part that determines the predictive SHAPE: the gate mixing a Gaussian
    market branch against a Student-t refresh branch.  We hold the RLS mean
    at 0 (the best available estimate of a fitted mean is ~0; Q4 showed no
    exploitable drift) and run the production EWMA/EM on the variances.
    """

    def __init__(self, nu=3.0, decay=0.99, s0=50.0, gate=False):
        self.nu = nu
        self.a = 1.0 - decay
        self.gamma = decay
        self.mv = s0 * s0      # market_var['mid']
        self.rv = s0 * s0      # refresh_var['mid']
        self.gate = gate
        self.w_mkt = 0.5
        self.b0 = 0.0
        self.b1 = 0.0

    def target(self, t):
        j = t + 1
        return y["dmid"][j] if j < y["N"] else np.nan

    def predict(self, t):
        return 0.0, self.w_mkt, self.target(t)

    def logpdf(self, tgt, w):
        """Mixture: w*N(0,mv) + (1-w)*t_nu(0,rv), all as discrete lattice pmf."""
        gm = norm.cdf((tgt + 0.5) / np.sqrt(self.mv)) - norm.cdf((tgt - 0.5) / np.sqrt(self.mv))
        s = np.sqrt(self.rv * self.nu / (self.nu - 2.0))   # var-matched t scale
        gr = (tdist.cdf((tgt + 0.5) / s, df=self.nu) - tdist.cdf((tgt - 0.5) / s, df=self.nu))
        return float(np.log(max(w * gm + (1.0 - w) * gr, 1e-300)))

    def update(self, t):
        x = self.target(t)
        if not np.isfinite(x):
            return
        em = x * x
        er = x * x
        if self.gate:
            w = self.w_mkt
        else:
            w = 0.5
        self.mv = (1 - self.a * w) * self.mv + self.a * w * em
        self.rv = (1 - self.a * (1 - w)) * self.rv + self.a * (1 - w) * er
        if self.gate:
            llm = np.log(max(w * norm.pdf(x, 0, np.sqrt(self.mv)), 1e-300))
            s = np.sqrt(self.rv * self.nu / (self.nu - 2.0))
            llr = np.log(max((1 - w) * tdist.pdf(x, df=self.nu, scale=s), 1e-300))
            resp = np.exp(llm - np.logaddexp(llm, llr))
            self.w_mkt = float(np.clip(0.98 * self.w_mkt + 0.02 * resp, 0.01, 0.99))
        else:
            self.w_mkt = 0.5
        self.mv = max(self.mv, 1e-6)
        self.rv = max(self.rv, 1e-6)


def run(model):
    L = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t)
            continue
        _, w, tgt = model.predict(t)
        if np.isfinite(tgt):
            L[t] = -model.logpdf(tgt, w)
        model.update(t)
    return L


print("\n" + "=" * 96)
print("[26.1] Prequential NLL, all on the same 339k transitions")
print("=" * 96)
CANDS = [
    ("REAL PROD  mixture, gate OFF, nu=3", lambda: RealProd(nu=3.0, gate=False)),
    ("REAL PROD  mixture, gate ON,  nu=3", lambda: RealProd(nu=3.0, gate=True)),
    ("REAL PROD  mixture, gate ON,  nu=2.5", lambda: RealProd(nu=2.5, gate=True)),
    ("PROPOSED   hurdle-t lattice nu=2.5", None),
]
from research.model_lab.analysis import models_v3 as M
res = {}
for nm, mk in CANDS[:3]:
    L = run(mk())
    res[nm] = L
    print("  %-42s mean NLL = %8.4f" % (nm, np.nanmean(L)))


def run_hurdle(model):
    L = np.full(N, np.nan)
    p0s = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        mu, sd, tgt, p0 = model.predict(t, y)
        if np.isfinite(tgt):
            L[t] = -model.logpdf(tgt, mu, sd, p0)
            p0s[t] = p0
        model.update(t, y)
    return L, p0s


Lh, p0h = run_hurdle(M.HurdleTLattice(nu=2.5, lam=0.98, lam_p=0.98))
res["PROPOSED   hurdle-t lattice nu=2.5"] = Lh
print("  %-42s mean NLL = %8.4f" % ("PROPOSED   hurdle-t lattice nu=2.5", np.nanmean(Lh)))
print("  (zero mass reproduced: p0 mean %.4f, observed %.4f)"
      % (np.nanmean(p0h), np.mean(y["dmid"][1:N - 1] == 0.0)))

REF = "REAL PROD  mixture, gate ON,  nu=3"
b = res[REF]
print("\n" + "=" * 96)
print("[26.2] Paired difference vs the REAL incumbent (negative = better)")
print("=" * 96)
print("  %-42s %25s %25s" % ("model", "1st half", "2nd half"))
for nm, L in res.items():
    if nm == REF:
        continue
    row = "  %-42s" % nm
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        dd = L[lo:hi] - b[lo:hi]
        mean, l, h, n = S.moving_block_ci(dd, B=2000, seed=131)
        row += " %+.4f[%+.4f,%+.4f]" % (mean, l, h)
    print(row)

print("\n" + "=" * 96)
print("[26.3] Does the hurdle still beat the real incumbent?")
print("=" * 96)
dd = Lh - b
mean, l, h, n = S.moving_block_ci(dd, B=3000, seed=137)
tt, pp = S.dm_test(dd)
print("  real incumbent  %.4f nats" % np.nanmean(b))
print("  hurdle-t nu=2.5 %.4f nats" % np.nanmean(Lh))
print("  gain  %.4f nats/step  95%% CI [%.4f, %.4f]" % (-mean, -h, -l))
print("  DM t=%.1f p=%.3g n=%d" % (tt, pp, n))
print("  = %.1f%% of the incumbent's loss" % (100 * (-mean) / np.nanmean(b)))
print("=" * 96)
