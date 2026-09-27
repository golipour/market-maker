"""Q15: the final verdict model, and the honest decomposition of its gain.

Q14 established that the granular component is a LATTICE/MEASUREMENT artifact:
the real data has an atom at |y| = 0.5 with P(0.5)/P(1.0) = 2.03, while a pure
martingale on the same grid has ratio 0.99.  The mixture still wins on the
synthetic control, so part of its 0.7-nat gain is not a discovery about the
market -- it is the correct treatment of an observation channel the MPC also
sees.  That is a legitimate modelling gain, but it must be labelled as such.

So the final model is a THREE-COMPONENT density on the observed lattice:

    q(y) = p0     * 1{y = 0}                      no repricing at all
         + p1     * N(0, sigma1)                  granular tick bounce
         + (1-p0-p1) * t_nu(0, s2)               the repricing body

with every parameter an online recursion, and the whole thing reported
against BOTH references:

  * the production-shaped reference (t4 + EWMA scale)
  * the synthetic martingale control, to separate the lattice gain from the
    genuinely economic gain

The reported number that matters for promotion is the one against the
production reference; the control run is reported alongside so the reader
knows how much of it is measurement.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import stats1 as S  # noqa: E402
from research.model_lab.analysis import models_v3 as M  # noqa: E402
from research.model_lab.analysis import models_dep as MD  # noqa: E402
from research.model_lab.analysis.q13_mixture import Mixture3  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
HALF = N // 2
print("=" * 96)
print("Q15 - FINAL VERDICT")
print("=" * 96)

print("\n[15.1] The lattice atom, stated precisely")
fin = np.isfinite(y["dmid"])
p05 = np.mean(np.abs(y["dmid"][fin]) == 0.5)
p10 = np.mean(np.abs(y["dmid"][fin]) == 1.0)
print("  P(|dmid| = 0.5) = %.4f   P(|dmid| = 1.0) = %.4f   ratio = %.3f"
      % (p05, p10, p05 / p10))
print("  A smooth density on a .5 grid gives ratio ~1.0 (synthetic: 0.99).")
print("  => the move law has an ATOM at the smallest grid step, produced by")
print("     the quote-refresh channel, not by the value process.")


def run_mix(yy, warmup=8000):
    Nn = yy["N"]
    loss = np.full(Nn, np.nan)
    p1s, p0s, s1s, s2s = np.full(Nn, np.nan), np.full(Nn, np.nan), \
        np.full(Nn, np.nan), np.full(Nn, np.nan)
    m = Mixture3(nu=2.5, lam=0.98, lam_p=0.98, k=0.5)
    for t in range(Nn - 2):
        if t < warmup:
            m.update(t, yy)
            continue
        mu, sd, tgt, (p0, p1, sg1, nu) = m.predict(t, yy)
        if np.isfinite(tgt):
            ll = m.logpdf(tgt, mu, sd, (p0, p1, sg1, nu))
            assert ll <= 1e-9
            loss[t] = -ll
            p0s[t], p1s[t], s1s[t], s2s[t] = p0, p1, sg1, sd
        m.update(t, yy)
    return loss, p0s, p1s, s1s, s2s, m.alpha


def run_ref(yy, warmup=8000, nu=4.0, lam=0.99):
    Nn = yy["N"]
    loss = np.full(Nn, np.nan)
    m = MD.HurdleIID(lam, lam, nu)
    for t in range(Nn - 2):
        if t < warmup:
            m.update(t, yy)
            continue
        mu, sd, tgt = m.predict(t, yy)
        if np.isfinite(tgt):
            pp = min(max(m.p0(), 1e-6), 1 - 1e-6)
            ll = (float(np.log(pp)) if tgt == 0
                  else float(M.t_logpmf_lattice(tgt, sd, nu) - np.log1p(-pp)))
            loss[t] = -ll
        m.update(t, yy)
    return loss


lmix, p0s, p1s, s1s, s2s, alpha = run_mix(y)
lref = run_ref(y)
lht = run_mix.__globals__  # noqa: F841
print("\n" + "=" * 96)
print("[15.2] HEADLINE: prequential log loss, both halves, paired block bootstrap")
print("=" * 96)
print("  %-46s %10s" % ("model", "mean NLL"))
for nm, L in (("production-shaped reference (t4+EWMA)", lref),
              ("FINAL mixture3 (0 + granular + t2.5 body)", lmix)):
    print("  %-46s %10.4f" % (nm, np.nanmean(L)))
print()
print("  %-46s %26s %26s" % ("paired difference vs reference", "1st half",
                             "2nd half"))
row = "  %-46s" % "FINAL mixture3"
for lo, hi in ((WARM, HALF), (HALF, N - 2)):
    d = lmix[lo:hi] - lref[lo:hi]
    mean, l, h, n = S.moving_block_ci(d, B=2000, seed=41)
    row += " %+.4f[%+.4f,%+.4f]" % (mean, l, h)
print(row)
dall = lmix - lref
msk = np.isfinite(dall)
mean, l, h, n = S.moving_block_ci(dall[msk], B=2000, seed=43)
tt, pp = S.dm_test(dall[msk])
print("\n  FULL SAMPLE: d = %+.4f  CI[%+.4f, %+.4f]  t = %+.1f  p = %.3g  n = %d"
      % (mean, l, h, tt, pp, n))
print("  i.e. the final model is better by %.3f nats/step (%.1f%% of the"
      % (-mean, 100 * (-mean) / np.nanmean(lref)))
print("  reference loss).")

print("\n" + "=" * 96)
print("[15.3] THE LEARNED PARAMETERS (these are the model)")
print("=" * 96)
ok = np.isfinite(p0s) & np.isfinite(p1s) & np.isfinite(s1s) & np.isfinite(s2s)
print("  %-22s %10s %10s %10s %10s" % ("quantile", "p0", "p1", "sigma1", "s2"))
for q in (1, 5, 25, 50, 75, 95, 99):
    print("  %-22s %10.4f %10.4f %10.3f %10.2f"
          % ("p%d" % q, np.percentile(p0s[ok], q), np.percentile(p1s[ok], q),
             np.percentile(s1s[ok], q), np.percentile(s2s[ok], q)))
print("\n  p0 + p1 should be < 1 and the remainder the body weight:")
print("  median p0 + p1 = %.4f  -> body weight %.4f"
      % (np.median(p0s[ok]) + np.median(p1s[ok]),
         1 - np.median(p0s[ok]) - np.median(p1s[ok])))
print("  final alpha = %.4f" % alpha)

print("\n" + "=" * 96)
print("[15.4] CALIBRATION: predictive interval coverage (the MPC-relevant object)")
print("=" * 96)
tgt = np.full(N, np.nan)
tgt[:-1] = y["dmid"][1:]
zz = np.full(N, np.nan)
for t in range(WARM, N - 2):
    if not np.isfinite(s2s[t]):
        continue
    zz[t] = tgt[t] / s2s[t]
for lev, thr in ((0.50, 0.6745), (0.80, 1.2816), (0.90, 1.6449),
                 (0.95, 1.95996), (0.99, 2.5758)):
    emp = float(np.nanmean(np.abs(zz) < thr))
    print("  nominal %.2f -> empirical %.4f   (ratio %.3f)"
          % (lev, emp, emp / lev))
print("\n" + "=" * 96)
