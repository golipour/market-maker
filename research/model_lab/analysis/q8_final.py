"""Q8: the decisive model comparison.

Fixed contract for every candidate:
  * causal: the density at t uses only realised values up to t
  * prequential: score against the move over [t, t+1] BEFORE updating
  * paired block-bootstrap CIs against the reference
  * PIT calibration and hurdle calibration reported for every candidate
  * an explicit assertion that -log q >= 0, so no candidate can win by
    reporting an impossible density

The ladder:
  R0  iid Student-t, EWMA scale              (incumbent shape)
  R1  + hurdle                              (the measured 0.895 nat)
  R2  + conditional p0 by spread
  R3  + conditional p0 and scale by (spread, staleness)
  R4  + (spread, staleness, trade)
  R5  R4 with the lattice correction
  R6  R5 without calibration  (isolates what calibration buys)
  R7  R5 with a Gaussian continuous part     (isolates the tail index)
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
print("=" * 100)
print("Q8 - DECISIVE MODEL COMPARISON")
print("=" * 100)
print("N = %d  warmup = %d  target: dmid over [t, t+1]" % (N, WARM))


def run(model):
    """Prequential loop: predict -> score -> update.

    v3 models return (mu, sd, target, p0) and own their density.  The DepBase
    reference returns (mu, sd, target); it is scored on the SAME
    lattice-corrected footing so the comparison isolates model structure and
    not discretisation (its continuous density exceeds 1 on a lattice).
    """
    from research.model_lab.analysis.models_v3 import t_logpmf_lattice
    loss = np.full(N, np.nan)
    mu = np.full(N, np.nan)
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
            assert s_ > 0, "sd<=0 at t=%d" % t
            if p_ is not None:
                assert 0.0 <= p_ <= 1.0, "p0 out of range at t=%d: %r" % (t, p_)
                ll = model.logpdf(tgt, m_, s_, p_)
            else:
                pp = min(max(model.p0(), 1e-6), 1 - 1e-6)
                ll = (float(np.log(pp)) if tgt == 0
                      else float(t_logpmf_lattice(tgt, s_, 4.0)
                                 - np.log1p(-pp)))
            assert ll <= 1e-9, ("log q > 0 at t=%d tgt=%r sd=%r ll=%r"
                                % (t, tgt, s_, ll))
            mu[t], sd[t] = m_, s_
            p0[t] = p_ if p_ is not None else np.nan
            loss[t] = -ll
        model.update(t, y)
    return loss, mu, sd, p0


CANDS = [
    ("R0 iid t4, EWMA scale", lambda: MD.HurdleIID(0.99, 0.99, 4.0), "d3"),
    ("R1 hurdle-t (global p0)", lambda: M.HurdleT(nu=4.0), "v3"),
    ("R2 + cond p0 (spread)", lambda: M.HurdleT(nu=4.0, state_fn=M.cell_sp), "v3"),
    ("R3 + cond p0,sd (sp,stale)",
     lambda: M.HurdleT(nu=4.0, state_fn=M.cell_sp_st), "v3"),
    ("R4 + cond p0,sd (sp,stale,trade)",
     lambda: M.HurdleT(nu=4.0, state_fn=M.cell_sp_st_oc), "v3"),
    ("R5 R4 + lattice correction",
     lambda: M.HurdleTLattice(nu=4.0, state_fn=M.cell_sp_st_oc), "v3"),
    ("R6 R5 without calibration",
     lambda: M.HurdleTLattice(nu=4.0, state_fn=M.cell_sp_st_oc,
                              calibrate=False), "v3"),
    ("R7 R4 with nu=2.5 (fatter)",
     lambda: M.HurdleT(nu=2.5, state_fn=M.cell_sp_st_oc), "v3"),
    ("R8 R4 with nu=6 (thinner)",
     lambda: M.HurdleT(nu=6.0, state_fn=M.cell_sp_st_oc), "v3"),
    ("R9 R4 with nu=8 (thinner)",
     lambda: M.HurdleT(nu=8.0, state_fn=M.cell_sp_st_oc), "v3"),
]

print("\n" + "=" * 100)
res = {}
for name, mk, kind in CANDS:
    L, mu, sd, p0 = run(mk())
    a = None
    m = mk()
    if hasattr(m, "alpha"):
        a = m.alpha
    res[name] = (L, mu, sd, p0, a)
    fin = L[np.isfinite(L)]
    print("  %-42s n=%7d mean=%8.4f med=%7.4f p90=%7.3f p99=%7.3f alpha=%s"
          % (name, len(fin), fin.mean(), np.percentile(fin, 50),
             np.percentile(fin, 90), np.percentile(fin, 99),
             "n/a" if a is None else "%.3f" % a))

REF = "R0 iid t4, EWMA scale"
print("\n" + "-" * 100)
print("Paired difference vs %s   (negative = better)" % REF)
print("-" * 100)
base = res[REF][0]
for name, (L, mu, sd, p0, a) in res.items():
    m = np.isfinite(L) & np.isfinite(base)
    dlt = L[m] - base[m]
    mean, lo, hi, n = S.moving_block_ci(dlt, B=1500, seed=13)
    tt, pp = S.dm_test(dlt)
    v = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not resolved")
    print("  %-42s d=%+.4f CI[%+.4f,%+.4f] t=%+8.1f  %s"
          % (name, mean, lo, hi, tt, v))

print("\n" + "-" * 100)
print("CALIBRATION  (want z mean ~ 0, z sd ~ 1.0, 80% cov ~ 0.80)")
print("-" * 100)
tgt = np.full(N, np.nan)
tgt[:-1] = y["dmid"][1:]
print("  %-42s %8s %8s %9s %10s %10s" % ("model", "z mean", "z sd", "80% cov",
                                         "p0 nom", "p0 emp"))
for name, (L, mu, sd, p0, a) in res.items():
    m = np.isfinite(mu) & np.isfinite(sd) & (sd > 0) & np.isfinite(tgt)
    if m.sum() < 1000:
        continue
    zz = (tgt[m] - mu[m]) / sd[m]
    cov = float(np.mean(np.abs(zz) < 1.2816))
    pm = m & np.isfinite(p0)
    print("  %-42s %+8.4f %8.4f %9.4f %10.4f %10.4f"
          % (name, zz.mean(), zz.std(), cov,
             np.mean(p0[pm]) if pm.any() else np.nan,
             np.mean(tgt[m] == 0)))
print("\n" + "=" * 100)
