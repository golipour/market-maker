"""Q1: Is the conditional transition law independent across time?

The incumbent family (EWMA scale, zero mean) uses the PAST only through a
single scalar variance statistic. Its implicit assumption is

    E[ L_t | F_t ] depends on the past only via sigma_t
    i.e. the standardised innovation  dmid_{t+1} / sigma_t  is iid  and
        independent of  dmid_t / sigma_t.

This script tests that assumption directly, at three levels:

  A. unconditional dependence of the move on its own past
  B. dependence AFTER standardising by the EWMA scale  (the actual assumption)
  C. dependence of the SQUARE of the standardised innovation on its own past
     (i.e. is the scale even correctly tracked?)

If (B) is zero but (C) is not, the incumbent is under-dispersed and a
conditional-scale correction is worth real nats. If (B) is non-zero, the
predictive density itself must carry serial dependence (GARCH-type, or a
predictive distribution built from a state that persists).
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import stats1 as S  # noqa: E402

np.set_printoptions(suppress=True, linewidth=150)

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
dm = y["dmid"]

print("=" * 78)
print("Q1 - SERIAL DEPENDENCE OF THE TRANSITION LAW")
print("=" * 78)
print("N = %d" % N)

# ---- build the causal EWMA scale (zero mean, Student-t scale) ---------------
# scale_t^2 = EWMA of dmid_t^2, using only dmid up to time t.
lam = 0.99
sd = np.full(N, np.nan)
s, w = 0.0, 0.0
for t in range(N):
    if w > 1.0:
        sd[t] = np.sqrt(s / w)
    x = dm[t]
    if np.isfinite(x):
        s = s * lam + x * x
        w = w * lam + 1.0
ok = np.isfinite(sd) & (sd > 0)
z = np.where(ok, dm / sd, np.nan)
zz = z[np.isfinite(z)]
print("standardised innovation: n=%d  mean=%+.4f  sd=%.4f  (want 0, ~0.7 for t4)"
      % (len(zz), zz.mean(), zz.std()))

print("\n[A] Unconditional ACF of dmid (raw move, no scaling)")
for k in (1, 2, 3, 5, 10, 20, 60):
    r, n = S.acf(dm, k)
    print("    lag %-3d  %+.4f   (n=%d)" % (k, r, n))

print("\n[B] ACF of the STANDARDISED innovation  dmid_t / sd_t")
print("    <- this is exactly the incumbent's iid assumption.  Want ~0.")
for k in (1, 2, 3, 5, 10, 20, 60):
    r, n = S.acf(z, k)
    t = r * np.sqrt(max(n - k - 1, 1))
    p = 2 * stats.t.sf(abs(t), df=max(1, n - 2 * k - 2)) if np.isfinite(r) else np.nan
    print("    lag %-3d  %+.4f   t=%+7.1f  p=%.3g   (n=%d)" % (k, r, t, p, n))

print("\n[C] ACF of the SQUARED standardised innovation")
print("    <- tests whether the EWMA scale actually tracks conditional variance")
z2 = z ** 2
for k in (1, 2, 3, 5, 10, 20, 60):
    r, n = S.acf(z2, k)
    print("    lag %-3d  %+.4f   (n=%d)" % (k, r, n))

print("\n[D] Volatility-of-volatility: is the scale itself predictable?")
print("    If |z_t| is predictable from |z_{t-1}|, the scale should be a")
print("    filtered quantity, not a plain EWMA.")
az = np.abs(z)
for k in (1, 2, 3, 5):
    r, n = S.acf(az, k)
    print("    ACF(|z|) lag %-3d  %+.4f   (n=%d)" % (k, r, n))

print("\n[E] Decay of the squared-innovation autocorrelation (GARCH signature)")
ac = np.array([S.acf(z2, k)[0] for k in range(1, 121)])
half = np.nanargmax(np.abs(ac) < 0.5 * ac[0]) if np.isfinite(ac[0]) else -1
print("    ACF1(z^2)=%+.4f  half-life to <50%% of that: %d steps"
      % (ac[0], half))
print("    sum(ACF) over 120 lags = %+.3f  (=> integrated persistence)"
      % np.nansum(ac))

print("\n" + "=" * 78)
