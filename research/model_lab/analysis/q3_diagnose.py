"""Q3: why did the ARCH mean not help, and why did the GARCH scale break?

Two anomalies from Q2:
  (1) D2 (fixed ARCH mean) beat D1 by only -0.0004 nats, even though Q1
      measured ACF1(standardised innovation) = -0.189 with t = -112.
  (2) D3/D4 (GARCH scale) were *worse* by +0.42 nats and wildly
      miscalibrated (z sd 6.86 where 1.0 is required).

Hypothesis for (1): the hurdle already removes the dependence.  The negative
ACF of the standardised innovation is carried by the ZERO-MASS / sign-flip
structure, and once the point mass at zero is modelled, the continuous part
has little residual mean dependence.  If true, the roadmap's claim that there
is no exploitable directional signal survives even the dependence-aware test --
which would be a much stronger statement.

Hypothesis for (2): the recursion is mis-specified.  standardising the SHOCK
and then taking its EWMA gives sigma^2 in units of the OLD sigma, so the
recursion is not a variance recursion at all; it compounds.  The fix is to keep
the scale in observable units and add the shock variance in absolute terms.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import stats1 as S  # noqa: E402
from research.model_lab.analysis import preq as P  # noqa: E402
from research.model_lab.analysis import models_dep as M  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
print("=" * 84)
print("Q3 - DIAGNOSING THE TWO ANOMALIES")
print("=" * 84)

# ---- reconstruct the causal EWMA scale used in Q1 ---------------------------
lam = 0.99
sd = np.full(N, np.nan)
s, w = 0.0, 0.0
for t in range(N):
    if w > 1.0:
        sd[t] = np.sqrt(s / w)
    x = y["dmid"][t]
    if np.isfinite(x):
        s = s * lam + x * x
        w = w * lam + 1.0
z = np.where(np.isfinite(sd) & (sd > 0), y["dmid"] / sd, np.nan)

print("\n[3.1] Where does the negative ACF of z actually live?")
print("  Decompose the standardised innovation by whether the move was zero.")
is0 = y["dmid"] == 0
print("\n  (a) ALL buckets:")
r, n = S.acf(z, 1)
print("      ACF1(z) = %+.4f  (n=%d)" % (r, n))
print("\n  (b) CONDITIONAL on dmid_t != 0 and dmid_{t-1} != 0 (continuous part):")
# mask of "both endpoints non-zero", NaN-padded to the length-N contract
cont = (~is0).astype(float)
prev_cont = np.concatenate([[np.nan], cont[:-1]])
both = (cont * prev_cont) > 0
r, n = S.acf(z, 1, mask=both)
print("      ACF1(z) = %+.4f  (n=%d)" % (r, n))
print("\n  (c) CONDITIONAL on dmid_t != 0:")
r, n = S.acf(z, 1, mask=(~is0))
print("      ACF1(z) = %+.4f  (n=%d)" % (r, n))
print("\n  (d) The SIGN process, which is what an ARCH mean actually uses:")
sgn = np.sign(y["dmid"])
r, n = S.acf(sgn, 1)
print("      ACF1(sign(dmid)) = %+.4f  (n=%d)" % (r, n))
r, n = S.acf(sgn, 2)
print("      ACF2(sign(dmid)) = %+.4f  (n=%d)" % (r, n))
print("\n  (e) MAGNITUDE process |z| restricted to consecutive non-zero moves:")
az = np.abs(z)
r, n = S.acf(az, 1, mask=both)
print("      ACF1(|z| | both nonzero) = %+.4f  (n=%d)" % (r, n))
r, n = S.acf(az ** 2, 1, mask=both)
print("      ACF1(z^2 | both nonzero) = %+.4f  (n=%d)" % (r, n))

print("\n" + "-" * 84)
print("[3.2] Is the GARCH scale simply mis-specified? (unit check)")
print("  A correct scale recursion in observable units:")
print("     s2_t = (1-a)*s2_{t-1} + a*x_t^2          <- EWMA, stable, this is D0")
print("     s2_t = (1-a-b)*s2_{t-1} + a*x_t^2 + b*s2_{t-1}   <- GARCH, needs 2 knobs")
print("  The D3 recursion standardised the shock first, so its accumulator is")
print("  in units of 1/sigma^2 and compounds without bound.  Test:")
for lg in (0.90, 0.95, 0.97, 0.99):
    m2 = M.GarchScale(lam=lg, c=0.0, lam_p=0.99, nu=4.0)
    r = P.Runner(y, warmup=8000).run(m2)
    ss = r.sd[8000:]
    fin = np.isfinite(ss)
    print("     lam_g=%.2f  median sd=%9.2f  p99 sd=%12.2f  mean NLL=%.4f"
          % (lg, np.median(ss[fin]), np.percentile(ss[fin], 99),
             np.nanmean(r.loss[8000:])))

print("\n" + "=" * 84)
