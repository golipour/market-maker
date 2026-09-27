"""Q13: the non-zero move is a MIXTURE, not a Student-t.

Q12 found the decisive structural fact.  The non-zero |dmid| distribution has
quantiles

    p10=0.5  p25=1.0  p50=9.0  p75=49.0  p90=107.5  p99=224.0

and 18.4% of non-zero moves are EXACTLY 0.5 tick, 4.0% exactly 1.5, 2.4%
exactly 2.5.  No Student-t has that shape: the p25/p50 ratio is 9, which no
single scale family produces.  It is a mixture of

  (i)  a "no real information" component of tiny tick-granular moves, and
  (ii) a heavy-tailed body of genuine repricing.

This is the Roll-bounce signature showing up inside the non-zero branch.  So
the model needs THREE components, not two:

    q(y) = p0 * 1{y=0}
         + p1 * G(y; 0, sigma_tiny)          # granular micro-adjustments
         + (1 - p0 - p1) * t_nu(y; 0, s)     # the repricing body

with sigma_tiny on the tick scale (order 1 IRT) and p1 estimated online.
This script fits and scores that against the best two-component model.
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

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
print("=" * 92)
print("Q13 - THREE-COMPONENT MIXTURE (zero + granular + heavy body)")
print("=" * 92)

x = y["dmid"]
nz = x[np.isfinite(x) & (x != 0)]
print("\n[13.1] Shape of the non-zero branch")
print("  P(|y| = 0.5) = %.4f   P(|y| = 1.0) = %.4f   P(|y| = 1.5) = %.4f"
      % (np.mean(np.abs(nz) == 0.5), np.mean(np.abs(nz) == 1.0),
         np.mean(np.abs(nz) == 1.5)))
print("  P(|y| <= 2)  = %.4f" % np.mean(np.abs(nz) <= 2.0))
print("  => %.1f%% of non-zero moves are within 2 ticks of zero."
      % (100 * np.mean(np.abs(nz) <= 2.0)))
print("  These cannot come from a body with median |y| = 9 under ANY")
print("  single Student-t.  They are a separate component.")


class Mixture3(M.HurdleT):
    """zero + granular Gaussian + heavy Student-t body.

        q(y) = p0 * 1{y=0}
             + p1 * N(y; 0, sig1)
             + (1 - p0 - p1) * t_nu(y; 0, s2)

    sig1 is estimated from the SMALL moves only (|y| <= k ticks), s2 from the
    large ones, and p1 from the share of mass in the granular window.  All
    three are EWMA recursions on realised data, so nothing is fitted in batch
    and the split is causal (it uses only the realised value at t).
    """

    name = "mixture3"

    def __init__(self, nu=2.5, lam=0.98, lam_p=0.98, k=2.0, **kw):
        super().__init__(nu=nu, lam=lam, lam_p=lam_p, **kw)
        self.k = k
        self.s1s, self.s1w = 0.0, 0.0
        self.s2s, self.s2w = 0.0, 0.0
        self.p1s, self.p1w = 0.0, 0.0
        self.alpha1 = 1.0

    def sig1(self):
        return max(np.sqrt(self.s1s / self.s1w), 0.3) if self.s1w > 1.0 else 1.0

    def p1(self):
        return float(np.clip(self.p1s / self.p1w, 1e-4, 0.9)) \
            if self.p1w > 1.0 else 0.0

    def predict(self, t, y):
        p0 = self.p0(y, t)
        p1 = min(self.p1(), 0.95 * (1.0 - p0))
        sig1 = self.sig1() * self.alpha1
        s2 = self.scale(y, t)
        return 0.0, s2, self.target(y, t), (p0, p1, sig1, self.nu)

    def update(self, t, y):
        xv = self.target(y, t)
        if not np.isfinite(xv):
            return
        c = self._cell(y, t)
        s = self._get(self.S, c)
        z = self._get(self.Z, c)
        z[0] = z[0] * self.lam_p + float(xv == 0)
        z[1] = z[1] * self.lam_p + 1.0
        if xv == 0:
            self.p1s = self.p1s * self.lam_p
            self.p1w = self.p1w * self.lam_p + 1.0
            return
        s[0] = s[0] * self.lam + xv * xv
        s[1] = s[1] * self.lam + 1.0
        if abs(xv) <= self.k:
            self.s1s = self.s1s * self.lam + xv * xv
            self.s1w = self.s1w * self.lam + 1.0
            self.p1s = self.p1s * self.lam + 1.0
            self.p1w = self.p1w * self.lam + 1.0
        else:
            self.p1s = self.p1s * self.lam
            self.p1w = self.p1w * self.lam + 1.0
        if self.calibrate:
            cur = max(self._raw_scale(y, t), 1e-9)
            cc = self._get(self.C, 0)
            cc[0] = cc[0] * self.lam_cal + (xv / cur) ** 2
            cc[1] = cc[1] * self.lam_cal + 1.0
            if 1.0 / (1.0 - self.lam_cal) > 20.0 and cc[0] / cc[1] > 0.05:
                tgt = self.nu / (self.nu - 2.0) if self.nu > 2.0 else 1.0
                self.alpha = float(np.clip(
                    np.sqrt((cc[0] / cc[1]) / tgt), 0.5, self.cal_cap))
            if self.s1w > 20.0 and self.s1s / self.s1w > 0:
                # calibrate the granular component too
                self.alpha1 = float(np.clip(
                    np.sqrt((self.s1s / self.s1w)), 0.5, 4.0))

    def logpdf(self, tgt, mu, sd, pars):
        p0, p1, sig1, nu = pars
        p0 = min(max(p0, 1e-9), 1 - 1e-9)
        p1 = min(max(p1, 0.0), 0.95 * (1 - p0))
        if tgt == 0:
            return np.log(p0)
        rem = max(1.0 - p0 - p1, 1e-9)
        from scipy.special import log_ndtr
        g = float(log_ndtr(tgt / max(sig1, 1e-9))
                  - np.log(max(sig1, 1e-9)) - 0.5 * np.log(2 * np.pi))
        b = M.t_logpmf_lattice(tgt, max(sd, 1e-9), nu)
        return float(np.logaddexp(np.log(p1) + g,
                                   np.log(rem) + b))


def run(model):
    loss = np.full(N, np.nan)
    sd = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        out = model.predict(t, y)
        m_, s_, tgt, pars = out
        if np.isfinite(tgt):
            ll = model.logpdf(tgt, m_, s_, pars)
            assert ll <= 1e-9, "impossible density at t=%d ll=%r" % (t, ll)
            loss[t] = -ll
            sd[t] = s_
        model.update(t, y)
    return loss, sd


print("\n" + "=" * 92)
CANDS = []
for k in (0.5, 1.0, 1.5, 2.0, 3.0):
    CANDS.append(("mixture3 k=%.0f nu=2.5" % k,
                  lambda k=k: Mixture3(nu=2.5, lam=0.98, lam_p=0.98, k=k)))
CANDS.append(("mixture3 k=1 nu=3.0",
              lambda: Mixture3(nu=3.0, lam=0.98, lam_p=0.98, k=1.0)))
CANDS.append(("mixture3 k=1 nu=2.0",
              lambda: Mixture3(nu=2.0, lam=0.98, lam_p=0.98, k=1.0)))
CANDS.append(("mixture3 k=1 nu=1.5",
              lambda: Mixture3(nu=1.5, lam=0.98, lam_p=0.98, k=1.0)))
CANDS.append(("mixture3 k=1 lam=0.995",
              lambda: Mixture3(nu=2.5, lam=0.995, lam_p=0.995, k=1.0)))
CANDS.append(("mixture3 k=1 no calib",
              lambda: Mixture3(nu=2.5, lam=0.98, lam_p=0.98, k=1.0,
                               calibrate=False)))
CANDS.append(("BASELINE hurdle-t nu=2.5",
              lambda: M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98)))

res = {}
print("%-30s %10s %10s %10s" % ("model", "mean", "median", "p99"))
for nm, mk in CANDS:
    L, sd = run(mk())
    res[nm] = (L, sd)
    fin = L[np.isfinite(L)]
    print("%-30s %10.4f %10.4f %10.4f"
          % (nm, fin.mean(), np.percentile(fin, 50), np.percentile(fin, 99)))

BASE = "BASELINE hurdle-t nu=2.5"
HALF = N // 2
print("\n" + "-" * 92)
print("Paired difference vs the two-component baseline, BY HALF")
print("-" * 92)
b = res[BASE][0]
print("  %-30s %26s %26s" % ("model", "1st half", "2nd half"))
for nm, (L, sd) in res.items():
    row = "  %-30s" % nm
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        d = L[lo:hi] - b[lo:hi]
        mean, l, h, n = S.moving_block_ci(d, B=1200, seed=23)
        row += " %+.4f[%+.4f,%+.4f]" % (mean, l, h)
    print(row)

print("\n" + "-" * 92)
print("CALIBRATION of the continuous branch (z sd ~ 1, cov ~ nominal)")
print("-" * 92)
tgt = np.full(N, np.nan)
tgt[:-1] = y["dmid"][1:]
print("  %-30s %14s %14s" % ("model", "z sd", "90% cov"))
for nm, (L, sd) in res.items():
    m = (np.arange(N) >= WARM) & np.isfinite(sd) & (sd > 0) & np.isfinite(tgt)
    m &= (tgt != 0)
    zz = tgt[m] / sd[m]
    print("  %-30s %14.3f %14.4f"
          % (nm, zz.std(), np.mean(np.abs(zz) < 1.645)))
print("\n" + "=" * 92)
