"""Q5: the final candidate set, prequentially scored and calibrated.

Runner contract here is the 4-tuple (mu, sd, p0, target); the model's own
`logpdf` is used, so the hurdle is handled inside the density exactly as the
model defines it (symmetric Student-t on |move| conditional on non-zero).

Scoring order per step: predict -> score -> update.  Nothing that requires the
realised future enters the density.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from research.model_lab.analysis import harness as H  # noqa: E402
from research.model_lab.analysis import stats1 as S  # noqa: E402
from research.model_lab.analysis import models_final as MF  # noqa: E402
from research.model_lab.analysis import models_dep as MD  # noqa: E402

d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]
WARM = 8000
print("=" * 96)
print("Q5 - FINAL CANDIDATE SET (prequential + calibration)")
print("=" * 96)
print("N = %d  warmup = %d" % (N, WARM))

print("\n[5.1] State structure: where does the conditional law actually vary?")
MF.scan_state(y)


def run4(model):
    loss = np.full(N, np.nan)
    mu = np.full(N, np.nan)
    sd = np.full(N, np.nan)
    p0 = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        # NOTE: GarchHurdle.predict returns (mu, sd, TARGET, p0).
        m_, s_, tgt, p_ = model.predict(t, y)
        if np.isfinite(tgt):
            mu[t], sd[t], p0[t] = m_, s_, p_
            # A normalised density has q <= 1, so -log q >= 0.  If p0 or sd
            # were swapped with the target this assertion fires immediately
            # instead of silently producing a negative mean loss.
            assert 0.0 <= p_ <= 1.0, "p0 out of range: %r" % (p_,)
            assert s_ > 0.0, "sd must be positive: %r" % (s_,)
            loss[t] = -model.logpdf(tgt, m_, s_, p_)
        model.update(t, y)
    return loss, mu, sd, p0


def run3(model):
    """For the DepBase models (3-tuple + logpdf(tgt, mu, sd))."""
    loss = np.full(N, np.nan)
    mu = np.full(N, np.nan)
    sd = np.full(N, np.nan)
    for t in range(N - 2):
        if t < WARM:
            model.update(t, y)
            continue
        m, s, tgt = model.predict(t, y)
        if np.isfinite(tgt):
            mu[t], sd[t] = m, s
            loss[t] = -model.logpdf(tgt, m, s)
        model.update(t, y)
    return loss, mu, sd, np.full(N, np.nan)


EDGES = [0, 7, 199, 1e9]

cands = [
    ("REF iid Student-t EWMA", lambda: MD.IIDStudent(0.99, 4.0), 3),
    ("REF+ hurdle (global p0)", lambda: MD.HurdleIID(0.99, 0.99, 4.0), 3),
    ("GARCH a=.10 b=.85, global p0",
     lambda: MF.GarchHurdle(a=0.10, b=0.85, nu=4.0), 4),
    ("GARCH a=.10 b=.85, cond p0(sp,st,oc)",
     lambda: MF.GarchHurdle(a=0.10, b=0.85, nu=4.0, p0_edges=EDGES,
                            use_stale=True, use_ocur=True), 4),
    ("GARCH a=.05 b=.90, cond p0",
     lambda: MF.GarchHurdle(a=0.05, b=0.90, nu=4.0, p0_edges=EDGES,
                            use_stale=True, use_ocur=True), 4),
    ("GARCH a=.20 b=.70, cond p0",
     lambda: MF.GarchHurdle(a=0.20, b=0.70, nu=4.0, p0_edges=EDGES,
                            use_stale=True, use_ocur=True), 4),
    ("GARCH a=.10 b=0 (EWMA-equiv), cond p0",
     lambda: MF.GarchHurdle(a=0.10, b=0.0, nu=4.0, p0_edges=EDGES,
                            use_stale=True, use_ocur=True), 4),
    ("GARCH a=.10 b=.85, cond p0, NO calib",
     lambda: MF.GarchHurdle(a=0.10, b=0.85, nu=4.0, p0_edges=EDGES,
                            use_stale=True, use_ocur=True, calibrate=False), 4),
]

print("\n" + "=" * 96)
res = {}
for name, mk, kind in cands:
    L, mu, sd, p0 = (run4(mk()) if kind == 4 else run3(mk()))
    res[name] = (L, mu, sd, p0, mk().__dict__.get("alpha"))
    fin = L[np.isfinite(L)]
    print("  %-42s n=%7d mean=%8.4f med=%7.4f p90=%8.3f p99=%8.3f"
          % (name, len(fin), fin.mean(), np.percentile(fin, 50),
             np.percentile(fin, 90), np.percentile(fin, 99)))

REF = "REF+ hurdle (global p0)"
print("\n" + "-" * 96)
print("Paired difference vs %s  (negative = better)" % REF)
print("-" * 96)
base = res[REF][0]
for name, (L, mu, sd, p0, alpha) in res.items():
    m = np.isfinite(L) & np.isfinite(base)
    dlt = L[m] - base[m]
    mean, lo, hi, n = S.moving_block_ci(dlt, B=1500, seed=11)
    tt, pp = S.dm_test(dlt)
    v = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not resolved")
    extra = "" if alpha is None else "  alpha=%.3f" % alpha
    print("  %-42s d=%+.4f CI[%+.4f,%+.4f] t=%+7.1f %s%s"
          % (name, mean, lo, hi, tt, v, extra))

print("\n" + "-" * 96)
print("CALIBRATION  (want z mean ~ 0, z sd ~ 1, 80% cov ~ 0.80)")
print("-" * 96)
print("  %-42s %8s %8s %9s %8s" % ("model", "z mean", "z sd", "80% cov", "p0 sd"))
tgt = np.full(N, np.nan)
tgt[:-1] = y["dmid"][1:]
for name, (L, mu, sd, p0, alpha) in res.items():
    m = np.isfinite(mu) & np.isfinite(sd) & (sd > 0) & np.isfinite(tgt)
    if m.sum() < 1000:
        continue
    zz = (tgt[m] - mu[m]) / sd[m]
    cov = float(np.mean(np.abs(zz) < 1.2816))
    p0s = np.nanstd(p0[m]) if np.isfinite(p0[m]).any() else np.nan
    print("  %-42s %+8.4f %8.4f %9.4f %8.4f"
          % (name, zz.mean(), zz.std(), cov, p0s))

print("\n" + "-" * 96)
print("HURDLE QUALITY: is p0 itself well calibrated?")
print("-" * 96)
print("  %-42s %10s %10s" % ("model", "nominal", "empirical"))
for name, (L, mu, sd, p0, alpha) in res.items():
    if not np.isfinite(p0).any():
        continue
    m = np.isfinite(p0) & np.isfinite(tgt)
    nom = float(np.mean(p0[m]))
    emp = float(np.mean(tgt[m] == 0))
    print("  %-42s %10.4f %10.4f" % (name, nom, emp))
print("\n" + "=" * 96)
