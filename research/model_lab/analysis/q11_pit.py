"""Q11: proper PIT calibration of the winner, including the zero branch.

Q10's "PIT deciles" were computed on the raw standardised residual, which is
dominated by the zero mass (35% of outcomes sit at z = 0) and therefore
cannot show whether the CDF is calibrated.  The correct diagnostic is the
PROBABILITY INTEGRAL TRANSFORM

    u = F(y | I_t)   with   u ~ U(0,1)  if and only if F is calibrated.

So this script forms the model's own CDF at the realised outcome, both as a
whole and split by branch, and tests uniformity.  This is the only
calibration statement that is actually meaningful for a density forecast, and
it is the gate the roadmap requires before promotion.
"""
import os
import sys

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import models_v3 as M  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
print("=" * 84)
print("Q11 - PIT CALIBRATION OF THE WINNER (B: hurdle-t, nu=2.5, alpha online)")
print("=" * 84)


def t_cdf_np(x, nu):
    return M.t_cdf(x, nu)


T = M.t_cdf


m = M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98)
NEG = np.full(N, -np.inf)
NEG[-1] = np.nan
PIT = np.full(N, np.nan)
BR0 = np.zeros(N, bool)
for t in range(N - 2):
    if t < WARM:
        m.update(t, y)
        continue
    mu, sd, tgt, p0 = m.predict(t, y)
    s = max(sd, 1e-9)
    p0 = min(max(p0, 1e-9), 1 - 1e-9)
    if tgt == 0:
        # PIT of a DISCRETE outcome is  F(y-) + v*(F(y) - F(y-)), and
        # v is itself Uniform(0,1) under a calibrated forecast.  Using F(y)
        # alone is wrong and inflates the mean PIT toward p0.
        u_ = np.random.default_rng(t).random()
        F_lo = p0
        F_hi = p0
    else:
        F_lo = p0 + (1.0 - p0) * t_cdf_np((tgt - 0.5 - mu) / s, m.nu)
        F_hi = p0 + (1.0 - p0) * t_cdf_np((tgt + 0.5 - mu) / s, m.nu)
        u_ = np.random.default_rng(t).random()
    PIT[t] = min(max(F_lo + u_ * (F_hi - F_lo), 0.0), 1.0)
    BR0[t] = (tgt == 0)
    m.update(t, y)

ok = np.isfinite(PIT)
u = PIT[ok]
print("\n[11.1] Overall PIT uniformity (n = %d)" % len(u))
print("  mean u = %.4f   (want 0.5)" % u.mean())
ks = st.kstest(u, 'uniform')
print("  KS test vs Uniform(0,1): D=%.5f  p=%.4g" % (ks.statistic, ks.pvalue))
cnt, edges = np.histogram(u, bins=10, range=(0, 1))
exp = len(u) / 10.0
print("  decile counts (want %.0f each):" % exp)
print("   ", " ".join("%d" % c for c in cnt))
chi2 = float(((cnt - exp) ** 2 / exp).sum() / 9.0)
print("  chi2/9 = %.2f  (want ~1)" % chi2)

print("\n[11.2] PIT by branch")
for nm, msk in (("zero move  (y==0)", BR0[ok]), ("non-zero move", ~BR0[ok])):
    if msk.sum() < 100:
        continue
    uu = u[msk]
    k2 = st.kstest(uu, 'uniform')
    print("  %-18s n=%7d  mean=%.4f  KS D=%.5f p=%.4g  frac in [0,0.35] = %.4f"
          % (nm, msk.sum(), uu.mean(), k2.statistic, k2.pvalue,
             np.mean(uu < 0.35)))

print("\n[11.3] The zero branch is the weak point: PIT values for y==0 are")
print("  concentrated below p0, so their uniformity is governed entirely by")
print("  how well p0 is estimated.  Mean PIT | y==0 should be ~p0/2.")
u0 = u[BR0[ok]]
print("  mean PIT on the zero branch = %.4f  (p0 ~ 0.346, so ~0.173 expected)"
      % u0.mean())

print("\n[11.4] Coverage of the predictive INTERVAL (the MPC-relevant object)")
for lev in (0.50, 0.80, 0.90, 0.95):
    a = (1 - lev) / 2
    lo_q, hi_q = T(a, m.nu), T(1 - a, m.nu)
    cov = np.zeros(N, bool)
    mm = M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98)
    for t in range(N - 2):
        if t < WARM:
            mm.update(t, y)
            continue
        mu, sd, tgt, p0 = mm.predict(t, y)
        s = max(sd, 1e-9)
        cov[t] = ((tgt - mu) / s >= lo_q) and ((tgt - mu) / s <= hi_q)
        mm.update(t, y)
    c = cov[ok].mean()
    print("  nominal %.2f -> empirical %.4f  (ratio %.3f)"
          % (lev, c, c / lev))
print("\n" + "=" * 84)
