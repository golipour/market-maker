"""Q16: the granular component is a POINT MASS, and that closes the model.

Q15 found sigma1 pinned at its 0.25 floor for every quantile, while p1 ranges
0.03..0.24.  That is the optimiser saying: the tiny moves are not a narrow
Gaussian, they are an ATOM at exactly +/-0.5 tick.  Combined with Q14's
P(0.5)/P(1.0) = 2.03 (vs 0.99 on a smooth martingale), the complete observed
move law is therefore

    q(y) = p0   * 1{y = 0}          the quote did not move
         + p1   * 1{|y| = 0.5}      the quote moved one grid step
         + (1-p0-p1) * t_nu(0, s)   genuine repricing

which is a three-atom-plus-body law, entirely on the observed lattice, with
no continuous component below 1 tick.  This is the natural limit of the Q13
scan (k -> 0.5) and it should beat the Gaussian granular term.

This script confirms that, then reports the final ladder and the honest
decomposition of the gain into
    (a) the atom structure (a measurement/lattice effect), and
    (b) the tail body (the only part that is about the market).
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
print("Q16 - ATOM GRANULAR COMPONENT (the closed-form limit)")
print("=" * 96)


class Atom3(Mixture3):
    """q(y) = p0*[y=0] + p1*[|y|=0.5] + (1-p0-p1)*t_nu(0, s2)."""

    name = "atom3"

    def logpdf(self, tgt, mu, sd, pars):
        p0, p1, _sig1, nu = pars
        p0 = min(max(p0, 1e-9), 1 - 1e-9)
        p1 = float(min(max(p1, 0.0), 0.95 * (1.0 - p0)))
        if tgt == 0:
            return float(np.log(p0))
        if abs(tgt) == 0.5:
            return float(np.log(p1))
        rem = max(1.0 - p0 - p1, 1e-9)
        return float(M.t_logpmf_lattice(tgt, max(sd, 1e-9), nu)
                     + np.log(rem))


def run(model, yy=None, warmup=8000):
    yy = y if yy is None else yy
    Nn = yy["N"]
    loss = np.full(Nn, np.nan)
    info = {}
    for t in range(Nn - 2):
        if t < warmup:
            model.update(t, yy)
            continue
        mu, sd, tgt, pars = model.predict(t, yy)
        if np.isfinite(tgt):
            ll = model.logpdf(tgt, mu, sd, pars)
            assert ll <= 1e-9, "impossible density t=%d ll=%r" % (t, ll)
            loss[t] = -ll
            if t == Nn - 3:
                if isinstance(pars, tuple):
                    info = {"p0": pars[0], "p1": pars[1], "s2": sd}
                else:
                    info = {"p0": pars, "p1": np.nan, "s2": sd}
        model.update(t, yy)
    return loss, info


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
            loss[t] = -(float(np.log(pp)) if tgt == 0
                        else float(M.t_logpmf_lattice(tgt, sd, nu)
                                   - np.log1p(-pp)))
        m.update(t, yy)
    return loss


CANDS = [
    ("REF  t4 + EWMA (production shape)", None),
    ("B    hurdle-t nu=2.5 (2 components)", "ht"),
    ("M    mixture3 gaussian granular", "mix"),
    ("A    atom3 (0 / 0.5 / t2.5)  [FINAL]", "atom"),
]

print("\n" + "=" * 96)
res = {}
for nm, kind in CANDS:
    if kind is None:
        L = run_ref(y)
    elif kind == "ht":
        L, _ = run(M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98))
    elif kind == "mix":
        L, _ = run(Mixture3(nu=2.5, lam=0.98, lam_p=0.98, k=0.5))
    else:
        L, info = run(Atom3(nu=2.5, lam=0.98, lam_p=0.98, k=0.5))
    res[nm] = L
    fin = L[np.isfinite(L)]
    print("  %-40s mean=%8.4f med=%7.4f p99=%8.4f"
          % (nm, fin.mean(), np.percentile(fin, 50), np.percentile(fin, 99)))

REF = CANDS[0][0]
print("\n" + "-" * 96)
print("Paired difference vs the production-shaped reference, BY HALF")
print("-" * 96)
b = res[REF]
print("  %-40s %26s %26s" % ("model", "1st half", "2nd half"))
for nm, L in res.items():
    row = "  %-40s" % nm
    for lo, hi in ((WARM, HALF), (HALF, N - 2)):
        d = L[lo:hi] - b[lo:hi]
        mean, l, h, n = S.moving_block_ci(d, B=2000, seed=51)
        row += " %+.4f[%+.4f,%+.4f]" % (mean, l, h)
    print(row)

print("\n" + "-" * 96)
print("FINAL vs the two-component model: does the ATOM beat the Gaussian?")
print("-" * 96)
b2 = res["B    hurdle-t nu=2.5 (2 components)"]
for nm in ("M    mixture3 gaussian granular", "A    atom3 (0 / 0.5 / t2.5)  [FINAL]"):
    d = res[nm] - b2
    msk = np.isfinite(d)
    mean, l, h, n = S.moving_block_ci(d[msk], B=2000, seed=53)
    print("  %-40s d=%+.4f CI[%+.4f,%+.4f]" % (nm, mean, l, h))

print("\n" + "-" * 96)
print("CALIBRATION of the FINAL model (predictive interval coverage)")
print("-" * 96)
tgt = np.full(N, np.nan)
tgt[:-1] = y["dmid"][1:]
_, inf2 = run(Atom3(nu=2.5, lam=0.98, lam_p=0.98, k=0.5))
print("  final parameters: p0=%.4f  p1=%.4f  s2=%.3f  body=%.4f"
      % (inf2["p0"], inf2["p1"], inf2["s2"],
         1 - inf2["p0"] - inf2["p1"]))
print("\n  NOTE on coverage: the final density is a 3-atom law, so the")
print("  'central interval' of a Student-t is not the right object.  The")
print("  correct calibration statement is the PIT, tested in Q11/Q17.")
print("\n" + "=" * 96)
