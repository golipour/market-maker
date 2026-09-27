"""Q14: the granular component is a LATTICE artifact, not a market structure.

Q13 found the optimum at the smallest admissible window k = 0.5, i.e. the
"granular" component captures exactly the |y| = 0.5 move and nothing else.
That is a strong hint: the tiny moves are the half-tick grid of the mid
itself, not a separate economic process.  If so, the right model is not a
mixture at all but the CORRECT LATTICE TREATMENT of a single continuous law:

    y = round to the .5 grid applied to a continuous latent move,
    q(y) = F( (y + h)/2 ) - F( (y - h)/2 )   with h = 0.25 on the .5 grid

which is what the lattice-corrected Student-t already does.  But Q13's
mixture still beat it by 0.7 nats, so the mixture is capturing something
real.  This script separates the two explanations:

  (i)  LATTICE: the gain should come entirely from the exact +/-0.5 mass and
       should vanish if the data had no sub-tick structure.  Test: build a
       SYNTHETIC control that is a true continuous random walk observed on a
       .5 grid.  If the mixture also wins big there, the gain is lattice.
  (ii) ECONOMIC: if the mixture wins on the real data but NOT on the
       synthetic control, the granular component is real market structure.

This is the decisive test, and it is the same synthetic-control logic the
roadmap used for the ACF finding.
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
print("=" * 92)
print("Q14 - IS THE GRANULAR COMPONENT REAL, OR A LATTICE ARTIFACT?")
print("=" * 92)

print("\n[14.1] The exact structure of the small moves on the real data")
x = y["dmid"]
fin = np.isfinite(x)
print("  P(dmid == 0)   = %.4f" % np.mean(x[fin] == 0))
print("  P(|dmid| == 0.5)= %.4f" % np.mean(np.abs(x[fin]) == 0.5))
print("  P(|dmid| == 1.0)= %.4f" % np.mean(np.abs(x[fin]) == 1.0))
print("  P(|dmid| == 1.5)= %.4f" % np.mean(np.abs(x[fin]) == 1.5))
print("  P(|dmid| == 2.0)= %.4f" % np.mean(np.abs(x[fin]) == 2.0))
print("\n  Ratio P(0.5)/P(1.0) = %.3f" % (np.mean(np.abs(x[fin]) == 0.5)
                                         / np.mean(np.abs(x[fin]) == 1.0)))
print("  For a smooth symmetric density on a .5 grid, this ratio is ~1.0.")
print("  A large excess at 0.5 means the move distribution has an ATOM")
print("  at the smallest grid step, which a continuous law cannot produce.")
print("\n  Fraction of all moves that are 0 or +/-0.5: %.4f"
      % np.mean((np.abs(x[fin]) <= 0.5)))


def run_on(model, yy, warmup=8000):
    Nn = yy["N"]
    loss = np.full(Nn, np.nan)
    for t in range(Nn - 2):
        if t < warmup:
            model.update(t, yy)
            continue
        out = model.predict(t, yy)
        if len(out) == 4:
            m_, s_, tgt, pars = out
            if np.isfinite(tgt):
                ll = model.logpdf(tgt, m_, s_, pars)
                if ll <= 1e-9:
                    loss[t] = -ll
        model.update(t, yy)
    return loss


def run_ref(yy, warmup=8000):
    Nn = yy["N"]
    loss = np.full(Nn, np.nan)
    m = MD.HurdleIID(0.99, 0.99, 4.0)
    for t in range(Nn - 2):
        if t < warmup:
            m.update(t, yy)
            continue
        mu, sd, tgt = m.predict(t, yy)
        if np.isfinite(tgt):
            pp = min(max(m.p0(), 1e-6), 1 - 1e-6)
            if tgt == 0:
                loss[t] = -float(np.log(pp))
            else:
                loss[t] = -float(M.t_logpmf_lattice(tgt, sd, 4.0)
                                 - np.log1p(-pp))
        m.update(t, yy)
    return loss


print("\n" + "=" * 92)
print("[14.2] SYNTHETIC CONTROL: a pure martingale observed on a .5 grid")
print("=" * 92)
rng = np.random.default_rng(0)
latent = np.cumsum(rng.standard_normal(N) * 50.0)
ys = {k: np.full(N, np.nan) for k in ("dmid", "mid", "spread", "stale_ms", "ocur")}
ys["N"] = N
ys["dmid"] = np.round(np.diff(np.concatenate([[0.0], latent])) * 2.0) / 2.0
ys["mid"] = latent
ys["spread"] = np.full(N, 49.0)
ys["stale_ms"] = np.full(N, 1500.0)
ys["ocur"] = np.zeros(N)
fin = np.isfinite(ys["dmid"])
print("  synthetic: P(0)=%.4f  P(|0.5|)=%.4f  P(|1.0|)=%.4f"
      % (np.mean(ys["dmid"][fin] == 0),
         np.mean(np.abs(ys["dmid"][fin]) == 0.5),
         np.mean(np.abs(ys["dmid"][fin]) == 1.0)))
print("  => a true continuous RW on a .5 grid has NO atom at 0.5 beyond")
print("     what the grid creates, and a smooth 0.5/1.0 ratio.")

lref = run_ref(ys)
lmix = run_on(Mixture3(nu=2.5, lam=0.98, lam_p=0.98, k=0.5), ys)
lh = run_on(M.HurdleT(nu=2.5, lam=0.98, lam_p=0.98), ys)
print("\n  %-34s %10s" % ("model on SYNTHETIC control", "mean NLL"))
for nm, L in (("reference t4 + EWMA", lref),
              ("hurdle-t nu=2.5", lh),
              ("mixture3 k=0.5", lmix)):
    print("  %-34s %10.4f" % (nm, np.nanmean(L)))
mm = np.isfinite(lmix) & np.isfinite(lref)
dd = lmix[mm] - lref[mm]
mean, lo, hi, n = S.moving_block_ci(dd, B=1000, seed=31)
print("\n  mixture3 - reference on the SYNTHETIC control: %+.4f CI[%+.4f,%+.4f]"
      % (mean, lo, hi))
print("  On the REAL data the same comparison was about -0.7 nats.")
print()
if hi < 0:
    print("  => The mixture ALSO wins on data with NO microstructure content.")
    print("     The gain is therefore a LATTICE/MEASUREMENT artifact, exactly")
    print("     like the ACF1 finding, and it does NOT reflect tradable")
    print("     structure.  It is still correct to model -- the MPC consumes")
    print("     the same lattice -- but it is not a discovery about the market.")
else:
    print("  => The mixture does NOT win on the synthetic control, so the")
    print("     granular component reflects real market structure.")
print("\n" + "=" * 92)
