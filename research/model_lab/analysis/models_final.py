"""Final model family: correctly-specified, calibrated, serial-dependence aware.

Q1-Q4 established, on 347,640 buckets:

  * the conditional MEAN is not exploitable (ceiling 0.024 nats, realised 0.0004)
  * a hurdle at zero is worth 0.895 nats
  * |z| has ACF1 = +0.236 on consecutive non-zero moves  -> the SCALE carries
    the real serial dependence, and it must be modelled as a filtered
    quantity, not a plain EWMA

So the winning model is:  hurdle  x  Student-t  x  conditional scale, with

  * the scale a proper ARCH-in-variance recursion in OBSERVABLE units
        s2_t = (1-a-b) s2_{t-1} + a x_t^2 + b s2_{t-1}
    (GARCH, 2 knobs, stable, and it can express mean-reversion in variance
     via b < 0 which is what ACF1(z^2)=+0.043 would NOT want -- so we scan)
  * the hurdle probability p0_t conditional on a small set of state bins
  * an explicit, online-estimated CALIBRATION scalar, because the roadmap
    showed that a sharper-than-truth density wins NLL and loses MPC value
  * optional state conditioning on staleness / trade occurrence / spread
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

LOG2PI = float(np.log(2.0 * np.pi))


def _lt(y, mu, sd, nu):
    sd = max(float(sd), 1e-9)
    z = (y - mu) / sd
    lc = (gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
          - 0.5 * (np.log(nu * np.pi) + np.log(sd)))
    return float(lc - 0.5 * (nu + 1.0) * np.log1p(z * z / nu))


class GarchHurdle:
    """Hurdle x Student-t with a GARCH scale and a conditional zero mass.

    All recursions are causal.  `predict` builds the density from state that
    only saw realised values up to and including t; `update` folds in the move
    realised over [t, t+1].
    """

    name = "garch-hurdle"

    def __init__(self, a=0.10, b=0.85, nu=4.0, lam_p=0.99, s0=50.0,
                 floor=1.0, p0_0=0.35, p0_bins=None, p0_edges=None,
                 use_stale=False, use_ocur=False, calibrate=True,
                 cal_lam=0.99, s0_units=True):
        self.a, self.b, self.nu, self.lam_p = a, b, nu, lam_p
        self.s0, self.floor, self.p0_0 = s0, floor, p0_0
        self.s2 = s0 * s0
        self.use_stale, self.use_ocur = use_stale, use_ocur
        self.calibrate, self.cal_lam = calibrate, cal_lam
        self.alpha = 1.0
        self.u_sum, self.u_w = 0.0, 0.0
        self.zs, self.zw = {}, {}
        self._p0_edges = p0_edges
        self._p0_bins = p0_bins
        self.last_mu = 0.0
        self.last_sd = s0

    # ---- state binning for the conditional zero mass ------------------------
    def _bin(self, y, t):
        if not (self.use_stale or self.use_ocur) or self._p0_edges is None:
            return 0
        sp = y["spread"][t]
        st = y["stale_ms"][t]
        oc = y["ocur"][t]
        i = 0
        for e in self._p0_edges:
            if np.isfinite(sp) and sp >= e:
                i += 1
        j = 0
        if self.use_stale and np.isfinite(st):
            j = 1 if st < 1000.0 else 2
        k = 1 if (self.use_ocur and oc > 0) else 0
        return i * 9 + j * 2 + k

    def p0(self, y=None, t=None):
        if y is None:
            v = self.zs.get(0)
            return (v / self.zw[0]) if self.zw.get(0, 0) > 0 else self.p0_0
        b = self._bin(y, t)
        if self.zw.get(b, 0) > 1.0:
            return self.zs[b] / self.zw[b]
        if self.zw.get(0, 0) > 0:
            return self.zs[0] / self.zw[0]
        return self.p0_0

    def target(self, y, t, h=1):
        j = t + h
        return y["dmid"][j] if j < y["N"] else np.nan

    def predict(self, t, y):
        """Returns (mu, sd, target, p0) -- target third, matching preq.Runner."""
        sd = max(np.sqrt(max(self.s2, 1e-12)) * self.alpha, self.floor)
        self.last_mu, self.last_sd = 0.0, sd
        return 0.0, sd, self.target(y, t), self.p0(y, t)

    def update(self, t, y):
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        a, b = self.a, self.b
        self.s2 = (1.0 - a - b) * self.s2 + a * x * x + b * self.s2
        if not np.isfinite(self.s2) or self.s2 <= 0:
            self.s2 = self.s0 * self.s0
        bn = self._bin(y, t)
        self.zs[bn] = self.zs.get(bn, 0.0) * self.lam_p + float(x == 0)
        self.zw[bn] = self.zw.get(bn, 0.0) * self.lam_p + 1.0
        if self.calibrate:
            sd = max(np.sqrt(self.s2), self.floor)
            u = (x / sd) / max(self.alpha, 1e-6)
            self.u_sum = self.u_sum * self.cal_lam + u * u
            self.u_w = self.u_w * self.cal_lam + 1.0
            if self.u_w > 50.0:
                m2 = self.u_sum / self.u_w
                if m2 > 0.05:
                    self.alpha = float(np.clip(1.0 / np.sqrt(m2), 0.5, 3.0))

    def logpdf(self, tgt, mu, sd, p0):
        p0 = float(min(max(p0, 1e-6), 1 - 1e-6))
        if tgt == 0:
            return np.log(p0)
        return _lt(abs(tgt), 0.0, sd, self.nu) - np.log1p(-p0)


def scan_state(y):
    """Report how the zero mass and the robust scale split on the state vars."""
    N = y["N"]
    dm = y["dmid"]
    sp, st, oc = y["spread"], y["stale_ms"], y["ocur"]
    print("  %-34s %8s %9s %9s" % ("cell", "n", "P(dmid=0)", "iqr_sd"))
    edges = [0, 7, 199, 1e9]
    names = ["sp<7", "7<=sp<199", "sp>=199"]
    for i in range(3):
        lo, hi = edges[i], edges[i + 1]
        for j, (jlo, jhi, jn) in enumerate([(0, 1000, "st<1s"),
                                            (1000, 1e12, "st>=1s")]):
            for k in (0, 1):
                m = (sp >= lo) & (sp < hi) & (st >= jlo) & (st < jhi) & (oc == k)
                m &= np.isfinite(dm)
                if m.sum() < 500:
                    continue
                v = dm[m]
                lab = "%s %s %s" % (names[i], jn, "trade" if k else "quiet")
                q1, q3 = np.percentile(v, [25, 75])
                print("  %-34s %8d %9.4f %9.2f"
                      % (lab, m.sum(), np.mean(v == 0), (q3 - q1) / 1.349))
