"""Q28: the last missing piece -- conditional (non-zero-only) variance.

Q27 left an anomaly: production+zero-atom scores 4.14, but a bare hurdle-t
scores 3.52.  The hurdle should be WORSE, since it has no gate and no regime
mixing.  The explanation is the variance recursion.

Production updates with the raw second moment, INCLUDING the zeros:

    var <- (1 - a*w)*var + a*w*x^2          for every x, zero or not

so 35% of the driving observations are exactly 0 and they drag the scale
DOWN toward sqrt(0.35)*true_sigma.  The hurdle tracks

    m2 <- E[x^2 | x != 0]                   only non-zero shocks

The fix is small and surgical: keep everything production does, and simply
exclude zeros from the variance update.  This is the decisive test.
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
print("Q28 - CONDITIONAL (NON-ZERO-ONLY) VARIANCE")
print("=" * 96)


class ProdVar:
    """Production mixture with a switchable variance estimator.

    cond=False : var <- (1-a*w)*var + a*w*x^2            (production)
    cond=True  : var <- (1-a*w)*var + a*w*x^2            but SKIPPED when x==0,
                  with the weight renormalised so the effective memory is
                  preserved among non-zero shocks.
    """

    def __init__(self, nu=3.0, decay=0.99, s0=50.0, p0_0=0.35,
                 hurdle=True, cond=False):
        self.nu = nu
        self.a = 1.0 - decay
        self.mv = s0 * s0
        self.rv = s0 * s0
        self.w = 0.5
        self.hurdle = hurdle
        self.cond = cond
        self.p0n, self.p0d = p0_0, p0_0

    def target(self, t):
        j = t + 1
        return y["dmid"][j] if j < y["N"] else np.nan

    def _body(self, tgt):
        gm = (norm.cdf((tgt + 0.5) / np.sqrt(self.mv))
              - norm.cdf((tgt - 0.5) / np.sqrt(self.mv)))
        s = np.sqrt(self.rv * self.nu / (self.nu - 2.0))
        gr = (tdist.cdf((tgt + 0.5) / s, df=self.nu)
              - tdist.cdf((tgt - 0.5) / s, df=self.nu))
        return self.w * gm + (1.0 - self.w) * gr

    def logpdf(self, tgt):
        if self.hurdle:
            p0 = min(max(self.p0n / self.p0d, 1e-6), 1.0 - 1e-6)
            if tgt == 0:
                return float(np.log(p0))
            return float(np.log(max((1.0 - p0) * self._body(tgt), 1e-300)))
        return float(np.log(max(self._body(tgt), 1e-300)))

    def update(self, t):
        x = self.target(t)
        if not np.isfinite(x):
            return
        w = self.w
        # conditional: only non-zero shocks inform the scale, and the
        # weight is renormalised by (1-p0) to keep the same memory horizon
        if not (self.cond and x == 0.0):
            self.mv = max((1 - self.a * w) * self.mv + self.a * w * x * x, 1e-6)
            self.rv = max((1 - self.a * (1 - w)) * self.rv
                          + self.a * (1 - w) * x * x, 1e-6)
            self.mv = max((1 - self.a * w) * self.mv + self.a * w * x * x, 1e-6)
            self.rv = max((1 - self.a * (1 - w)) * self.rv
                          + self.a * (1 - w) * x * x, 1e-6)
        s = np.sqrt(self.rv * self.nu / (self.nu - 2.0))
        llm = np.log(max(w * norm.pdf(x, 0, np.sqrt(self.mv)), 1e-300))
        llr = np.log(max((1 - w) * tdist.pdf(x, df=self.nu, scale=s), 1e-300))
        resp = float(np.exp(llm - np.logaddexp(llm, llr)))
        self.w = float(np.clip(0.98 * self.w + 0.02 * resp, 0.01, 0.99))
        if self.hurdle:
            z = 1.0 if x == 0.0 else 0.0
            self.p0n = (1 - self.a) * self.p0n + self.a * z
            self.p0d = (1 - self.a) * self.p0d + self.a


def run(model):
    L = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t)
            continue
        tgt = model.target(t)
        if np.isfinite(tgt):
            L[t] = -model.logpdf(tgt)
        model.update(t)
    return L


CANDS = [
    ("A. PROD as-is (no atom, raw var)", dict(hurdle=False, cond=False)),
    ("B. + zero atom (raw var)", dict(hurdle=True, cond=False)),
    ("C. + zero atom + conditional var", dict(hurdle=True, cond=True)),
    ("D. zero atom + cond var, nu=2.5", dict(hurdle=True, cond=True, nu=2.5)),
]
print("\n[28.1] Prequential NLL -- isolating each change")
print("-" * 96)
res = {}
for nm, kw in CANDS:
    L = run(ProdVar(**kw))
    res[nm] = L
    print("  %-40s mean NLL = %8.4f" % (nm, np.nanmean(L)))

b = res["A. PROD as-is (no atom, raw var)"]
print("\n[28.2] Cumulative gain vs the real incumbent")
print("-" * 96)
prev = b
for nm in [c[0] for c in CANDS[1:]]:
    L = res[nm]
    dd = L - prev
    mean, l, h, n = S.moving_block_ci(dd, B=2000, seed=163)
    ddt = L - b
    mt, lt, ht, _ = S.moving_block_ci(ddt, B=2000, seed=167)
    row = "  %-40s step %+0.4f   cumulative %+0.4f [%+.4f,%+.4f]" % (
        nm, -mean, -mt, -ht, -lt)
    print(row)
    prev = L

print("\n[28.3] Final candidate vs incumbent, both halves")
print("-" * 96)
FIN = "C. + zero atom + conditional var"
dd = res[FIN] - b
mean, l, h, n = S.moving_block_ci(dd, B=3000, seed=173)
tt, pp = S.dm_test(dd)
for lo, hi, lab in ((WARM, HALF, "1st half"), (HALF, N - 2, "2nd half")):
    d2 = res[FIN][lo:hi] - b[lo:hi]
    m2, l2, h2, _ = S.moving_block_ci(d2, B=2000, seed=179)
    print("  %-12s %+.4f [%+.4f,%+.4f]" % (lab, m2, l2, h2))
print()
print("  total gain %.4f nats  95%% CI [%.4f, %.4f]  DM t=%.1f p=%.3g n=%d"
      % (-mean, -h, -l, tt, pp, n))
print("  %.1f%% of incumbent loss, %.0f nats/day/path"
      % (100 * (-mean) / np.nanmean(b), -mean * 8640))
print("=" * 96)
