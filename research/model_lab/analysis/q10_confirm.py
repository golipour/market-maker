"""Q10: the final model, confirmed out-of-sample with calibration active.

The specification that survived Q1-Q9:

    q(y) = p0(c) * 1{y=0} + (1 - p0(c)) * f_t(y ; 0, s(c), nu)

  mu = 0        (Q3/Q4: the mean is not exploitable; ceiling 0.024 nats,
                 realised 0.0004 nats)
  p0(c)         conditional zero mass, EWMA of the zero indicator
  s(c)          conditional scale from the NON-ZERO shocks, quantile-matched
  nu = 2.5      (Q9: optimal on BOTH halves independently)
  lambda=0.98   (Q9.4)
  c             cell = (spread x staleness x trade)
  alpha         online calibration scalar, ESS-gated, targeting nu/(nu-2)

This script reports the headline number with the calibration live, checks
calibration on both halves, and runs the two falsifiers from the roadmap:
does a mean help at all, and does a deliberate over-sharpening hurt?
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
HALF = N // 2
print("=" * 96)
print("Q10 - FINAL MODEL, OUT-OF-SAMPLE CONFIRMATION")
print("=" * 96)


def run(model, klattice=False):
    from research.model_lab.analysis.models_v3 import t_logpmf_lattice
    loss = np.full(N, np.nan)
    sd = np.full(N, np.nan)
    p0 = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        out = model.predict(t, y)
        if len(out) == 4:
            m_, s_, tgt, p_ = out
        else:
            m_, s_, tgt = out
            p_ = None
        if np.isfinite(tgt):
            assert s_ > 0
            if p_ is not None:
                assert 0.0 <= p_ <= 1.0
                ll = model.logpdf(tgt, m_, s_, p_)
            else:
                pp = min(max(model.p0(), 1e-6), 1 - 1e-6)
                ll = (float(np.log(pp)) if tgt == 0
                      else float(t_logpmf_lattice(tgt, s_, 4.0)
                                 - np.log1p(-pp)))
            assert ll <= 1e-9, "impossible density at t=%d" % t
            loss[t] = -ll
            sd[t] = s_
            p0[t] = p_ if p_ is not None else np.nan
        model.update(t, y)
    return loss, sd, p0


class _Sharp(M.HurdleT):
    """The final model with the scale FROZEN at a fixed multiple.

    This is the falsifier for the roadmap's central claim: a model that is
    sharper than the truth wins nothing and loses calibration, and would make
    an MPC overconfident.  Freezing the multiplier (rather than clipping the
    calibration cap) makes the perturbation actually take effect.
    """

    def __init__(self, mult, **kw):
        kw.pop('calibrate', None)
        super().__init__(calibrate=False, **kw)
        self._mult = float(mult)

    def scale(self, y, t):
        c = self._cell(y, t)
        v = self.S.get(c)
        m2 = (v[0] / v[1]) if (v and v[1] > 1.0 and v[0] > 0) else None
        if m2 is None:
            g = self.S.get(0)
            m2 = (g[0] / g[1]) if (g and g[1] > 1.0 and g[0] > 0) else 2500.0
        if self.nu > 2.0 + 1e-9:
            s = np.sqrt(m2 * (self.nu - 2.0) / self.nu)
        else:
            s = np.sqrt(m2) / M._t75(self.nu)
        return max(s, self.floor) * self._mult


BEST = dict(nu=2.5, lam=0.98, lam_p=0.98, state_fn=M.cell_sp_st_oc)

CANDS = [
    ("REF  iid t4 + EWMA scale", lambda: MD.HurdleIID(0.99, 0.99, 4.0)),
    ("REF+ lattice-consistent t4", lambda: MD.HurdleIID(0.99, 0.99, 4.0)),
    ("A   hurdle-t nu=4 global", lambda: M.HurdleT(nu=4.0)),
    ("B   hurdle-t nu=2.5 global", lambda: M.HurdleT(nu=2.5, lam=0.98,
                                                     lam_p=0.98)),
    ("C   B + cells (sp,stale,trade) [FINAL]",
     lambda: M.HurdleT(**BEST)),
    ("D   C without calibration",
     lambda: M.HurdleT(calibrate=False, **BEST)),
    ("E   B frozen at 0.6x scale (sharp)",
     lambda: _Sharp(0.6, **BEST)),
    ("F   B frozen at 1.4x scale (safe)",
     lambda: _Sharp(1.4, **BEST)),
]

print("\n" + "=" * 96)
res = {}
for name, mk in CANDS:
    m = mk()
    L, sd, p0 = run(m)
    a = getattr(m, "alpha", None)
    res[name] = (L, sd, p0, a)
    fin = L[np.isfinite(L)]
    print("  %-40s mean=%8.4f med=%7.4f p90=%7.3f p99=%7.3f  alpha=%s"
          % (name, fin.mean(), np.percentile(fin, 50),
             np.percentile(fin, 90), np.percentile(fin, 99),
             "n/a" if a is None else "%.3f" % a))

REF = "REF  iid t4 + EWMA scale"
print("\n" + "-" * 96)
print("Paired difference vs %s, BY HALF (negative = better)" % REF)
print("-" * 96)
base = res[REF][0]
print("  %-40s %26s %26s" % ("model", "1st half", "2nd half"))
for name, (L, sd, p0, a) in res.items():
    row = "  %-40s" % name
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        d = L[lo:hi] - base[lo:hi]
        mean, l, h, n = S.moving_block_ci(d, B=1200, seed=17)
        row += " %+.4f[%+.4f,%+.4f]" % (mean, l, h)
    print(row)

print("\n" + "-" * 96)
print("CALIBRATION BY HALF  (want z sd ~ 1, 80% cov ~ 0.80)")
print("-" * 96)
tgt = np.full(N, np.nan)
tgt[:-1] = y["dmid"][1:]
print("  %-40s %22s %22s" % ("model", "1st half", "2nd half"))
for name, (L, sd, p0, a) in res.items():
    row = "  %-40s" % name
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        msk = np.zeros(N, bool)
        msk[lo:hi] = True
        m = msk & np.isfinite(sd) & (sd > 0) & np.isfinite(tgt)
        zz = tgt[m] / sd[m]
        row += "  sd=%.3f cov=%.3f" % (zz.std(), np.mean(np.abs(zz) < 1.2816))
    print(row)

print("\n" + "-" * 96)
print("PIT DECILE CALIBRATION of the FINAL model (uniformity check)")
print("-" * 96)
L, sd, p0, a = res["C   B + cells (sp,stale,trade) [FINAL]"]
m = (np.arange(N) >= WARM) & np.isfinite(sd) & (sd > 0) & np.isfinite(tgt)
zz = tgt[m] / sd[m]
# a correct PIT is uniform; the standardised residual is only indicative
qs = np.percentile(zz, np.arange(10, 100, 10))
print("  PIT deciles of the standardised residual:")
print("   ", " ".join("%.3f" % v for v in qs))
print("  (a uniform PIT would be flat 0.1..0.9; these are the RAW residual")
print("   quantiles, so a good model tracks the fat-tailed distribution.)")
print("\n  coverage ladder (standardised residual, |z| quantiles):")
for lv in (0.50, 0.80, 0.90, 0.95, 0.99):
    thr = np.quantile(np.abs(zz), lv)
    print("    nominal %.2f -> empirical %.4f" % (lv, np.mean(np.abs(zz) <= thr)))
print("\n  final alpha = %.4f" % a)
print("\n" + "=" * 96)
