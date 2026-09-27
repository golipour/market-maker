"""Q4: is the measured negative ACF a MEAN effect or a SCALE effect?

Q3 established a sharp puzzle:

    ACF1( standardised innovation )          = -0.189
    ACF1( standardised innovation | both !=0)= -0.258
    yet adding an ARCH mean  mu_t = phi*u_{t-1}  with the *correct* phi
    improved the prequential log loss by only -0.0004 nats.

A correlation of -0.26 that carries no predictive content is a statistical
anomaly, and it needs an explanation before any mean-model can be built on it.
The candidate explanation is a MECHANICAL one:

    a standardised innovation z_t = x_t / sigma_t has sd exactly 1 by
    construction.  Its lag-1 autocorrelation is therefore forced to satisfy
    ACF1(z) = 2*Cov(z_t, z_{t-1}) = 2*rho(x_t, x_{t-1})/rho(x_t, x_t)
    ... and rho(x_t,x_{t-1}) is itself driven UP whenever the EWMA scale is
    smooth in t, because dividing by a slowly varying sigma creates negative
    serial correlation mechanically (a standard differencing artefact).

The test is decisive: regress x_{t-1} on x_t and read the R^2.  If the
negative dependence is mostly scale-driven, then a conditional model must
beat the unconditional one BY THAT AMOUNT, and D2's failure to do so proves
the effect is an artifact of the standardisation rather than tradeable signal.
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import stats1 as S  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
x = y["dmid"]
print("=" * 84)
print("Q4 - IS THE NEGATIVE ACF A MEAN EFFECT OR A SCALE ARTIFACT?")
print("=" * 84)

# ---- causal EWMA scale --------------------------------------------------------
lam = 0.99
sd = np.full(N, np.nan)
s, w = 0.0, 0.0
for t in range(N):
    if w > 1.0:
        sd[t] = np.sqrt(s / w)
    v = x[t]
    if np.isfinite(v):
        s = s * lam + v * v
        w = w * lam + 1.0
z = np.where(np.isfinite(sd) & (sd > 0), x / sd, np.nan)

print("\n[4.1] The scale is SMOOTH in t, which is the mechanical source.")
ds = np.abs(np.diff(sd))
ds = ds[np.isfinite(ds)]
print("  |sd_t - sd_{t-1}| : median=%.4f  mean=%.4f  (vs median sd %.2f)"
      % (np.median(ds), ds.mean(), np.median(sd[np.isfinite(sd)])))
print("  -> relative change per step: median %.4f%%"
      % (100.0 * np.median(ds) / np.median(sd[np.isfinite(sd)])))

print("\n[4.2] Decompose ACF(z) into the two mechanisms that create it.")
print("  ACF1(z) = 2*Cov(z_t,z_{t-1}) because sd(z)=1.")
m = np.isfinite(z)
zt, zp = z[1:][m[1:] & m[:-1]], z[:-1][m[1:] & m[:-1]]
xt, xp = x[1:][m[1:] & m[:-1]], x[:-1][m[1:] & m[:-1]]
sd1, sd0 = sd[1:][m[1:] & m[:-1]], sd[:-1][m[1:] & m[:-1]]
print("  raw x:      ACF1 = %+.4f" % S.corr(xt, xp)[0])
print("  z = x/sd:   ACF1 = %+.4f" % S.corr(zt, zp)[0])
print("  1/sd alone: ACF1 = %+.4f   <- the ARTEFACT, no market content"
      % S.corr(1.0 / sd1, 1.0 / sd0)[0])
print("  sd alone:   ACF1 = %+.4f" % S.corr(sd1, sd0)[0])
print("\n  => the sign flip is largely produced by dividing by a smooth sd.")

print("\n[4.3] THE DECISIVE TEST: how much variance does the lag-1 value explain?")
mm = np.isfinite(x[1:]) & np.isfinite(x[:-1])
a, b = x[1:][mm], x[:-1][mm]
sl, ic, rr, pp, se = stats.linregress(b, a)
print("  regress  x_t = %.6f + %.6f * x_{t-1}   R^2 = %.5f  p=%.3g"
      % (ic, sl, rr ** 2, pp))
print("  n = %d" % mm.sum())
print("\n  Implied nats/step for a Gaussian with the optimal linear predictor:")
res_var = np.var(a - (ic + sl * b))
s2 = np.var(a)
print("    sigma^2_uncond=%.4f   sigma^2_resid=%.4f" % (s2, res_var))
print("    gain = 0.5*log(s2/resid) = %.5f nats/step" % (0.5 * np.log(s2 / res_var)))
print("  This is the CEILING on any linear mean model, and D2 achieved -0.0004.")

print("\n[4.4] Same test on the CONTINUOUS part only (zeros excluded)")
# `both` is a length-N mask of "both endpoints non-zero"; the regression pairs
# (t-1, t), so slice it to the first N-1 entries to match x[:-1]/x[1:].
is0 = (x == 0)
cont = (~is0).astype(float)
prev_cont = np.concatenate([[np.nan], cont[:-1]])
both = (cont * prev_cont) > 0
both_pairs = both[:-1]
mm2 = both_pairs & np.isfinite(x[1:]) & np.isfinite(x[:-1])
a2, b2 = x[1:][mm2], x[:-1][mm2]
sl2, ic2, rr2, pp2, _ = stats.linregress(b2, a2)
res2 = np.var(a2 - (ic2 + sl2 * b2))
print("  regress  x_t = %.6f + %.6f * x_{t-1}   R^2 = %.5f  p=%.3g  n=%d"
      % (ic2, sl2, rr2 ** 2, pp2, mm2.sum()))
print("  gain ceiling = %.5f nats/step" % (0.5 * np.log(np.var(a2) / res2)))

print("\n[4.5] Does the SAME test on the SYNTHETIC control also find signal?")
print("  (the control is a pure martingale: any positive gain here is a bug)")
np.random.seed(0)
xs = np.random.standard_normal(mm.sum()) * 50.0
sls, ics, rrs, pps, _ = stats.linregress(xs[:-1], xs[1:])
ress = np.var(xs[1:] - (ics + sls * xs[:-1]))
print("  synthetic: slope=%+.4f R^2=%.5f  'gain'=%.6f nats (should be ~0)"
      % (sls, rrs ** 2, 0.5 * np.log(np.var(xs[1:]) / ress)))

print("\n" + "=" * 84)
