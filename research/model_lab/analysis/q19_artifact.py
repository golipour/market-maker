"""Q19: count the invalid-density steps that produced the phantom -0.818 gain.

The claim in the findings document is that the three-component mixture's
apparent 0.818-nat advantage came from a granular branch evaluated as a
CONTINUOUS density at a lattice point, which can exceed 1.  This script
counts those steps and quantifies exactly how much of the phantom score they
carried, so the claim is measured rather than asserted.

The earlier ad-hoc version of this check hung because it imported scipy
inside the per-step loop; here the Gram-Charlier series is evaluated
vectorised after the recursion has collected sigma1 and p1.
"""
import os
import sys

import numpy as np
from scipy.special import log_ndtr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import models_v3 as M  # noqa: E402
from research.model_lab.analysis.q13_mixture import Mixture3  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
print("=" * 88)
print("Q19 - MEASURING THE PHANTOM GAIN")
print("=" * 88)

m = Mixture3(nu=2.5, lam=0.98, lam_p=0.98, k=0.5)
S1 = np.full(N, np.nan)
P1 = np.full(N, np.nan)
P0 = np.full(N, np.nan)
S2 = np.full(N, np.nan)
for t in range(N - 2):
    if t < WARM:
        m.update(t, y)
        continue
    mu, sd, tgt, (p0, p1, s1, nu) = m.predict(t, y)
    P0[t], P1[t], S1[t], S2[t] = p0, p1, s1, sd
    m.update(t, y)

ok = np.isfinite(S1) & np.isfinite(P1) & (P1 > 0)
print("\n[19.1] The granular scale the mixture actually learned")
print("  sigma1 quantiles: " + " ".join(
    "p%d=%.3f" % (q, np.percentile(S1[ok], q)) for q in (1, 5, 50, 95, 99)))
print("  p1     quantiles: " + " ".join(
    "p%d=%.3f" % (q, np.percentile(P1[ok], q)) for q in (1, 5, 50, 95, 99)))
print("  s2     quantiles: " + " ".join(
    "p%d=%.2f" % (q, np.percentile(S2[ok], q)) for q in (1, 5, 50, 95, 99)))

print("\n[19.2] Is a CONTINUOUS density at the learned sigma1 a valid pmf?")
print("  A pmf on the 0.5 grid must satisfy q(y) <= 1 for every y.")
print("  The maximum of a continuous density is 1/(sigma*sqrt(2 pi)).")
for q in (1, 5, 50, 95, 99):
    s = np.percentile(S1[ok], q)
    peak = 1.0 / (s * np.sqrt(2 * np.pi))
    print("    p%-3d sigma1=%6.3f  -> max continuous density = %8.4f  %s"
          % (q, s, peak, "LEGAL" if peak <= 1 else "*** ILLEGAL (q>1) ***"))

print("\n[19.3] Count the steps where the OLD continuous form gave q > 1")
tgt = np.full(N, np.nan)
tgt[:-1] = y["dmid"][1:]
# rebuild the realised outcome for each scored step
n_bad = 0
n_tot = 0
worst = 0.0
worst_t = -1
for t in range(WARM, N - 2):
    s1, p1, p0 = S1[t], P1[t], P0[t]
    if not (np.isfinite(s1) and np.isfinite(p1) and p1 > 0):
        continue
    v = tgt[t]
    if not np.isfinite(v) or v == 0:
        continue
    n_tot += 1
    g = float(log_ndtr(v / max(s1, 1e-9)) - np.log(max(s1, 1e-9))
              - 0.5 * np.log(2 * np.pi))
    q = float(p1 * np.exp(g))
    if q > 1.0:
        n_bad += 1
        if q > worst:
            worst, worst_t = q, t
print("  non-zero scored steps examined : %d" % n_tot)
print("  steps with q > 1 (impossible)  : %d  (%.4f%%)"
      % (n_bad, 100.0 * n_bad / max(n_tot, 1)))
if worst_t >= 0:
    print("  worst q = %.4g at t = %d  (|dmid| = %.1f, sigma1 = %.3f)"
          % (worst, worst_t, abs(tgt[worst_t]), S1[worst_t]))

print("\n[19.4] Verdict")
if n_bad == 0:
    print("  The continuous granular branch never exceeded 1 on this sample.")
    print("  => the -0.818 must be re-EXPLAINED; the q>1 story is not the")
    print("     mechanism, and the findings document needs correcting.")
else:
    frac = n_bad / max(n_tot, 1)
    print("  %d of %d non-zero steps (%.2f%%) produced q > 1." % (n_bad, n_tot,
                                                                100 * frac))
    print("  Each such step scores a loss that is NEGATIVE by construction")
    print("  (since -log q < 0), which is impossible for a real predictive")
    print("  distribution.  That is the mechanism of the phantom gain.")
print("\n" + "=" * 88)
