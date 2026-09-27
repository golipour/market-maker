"""Phase 1: the measurement model.

How much of the observed 'randomness' is (a) our own 10s bucketing, (b) the
bid-ask bounce, (c) feed staleness?  This partitions the entropy floor.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from scipy import stats
from research.model_lab.analysis import harness as H

np.set_printoptions(suppress=True, linewidth=150)
d = H.load()
f = H.core_frame(d)
y = H.derived(f)
N = y["N"]


def cc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def ac(a, lag, mask=None):
    """corr(a[t], a[t+lag]) on the full series, restricted by `mask` at t."""
    a = np.asarray(a, dtype=float)
    if lag == 0:
        return np.nan
    if mask is None:
        m = np.isfinite(a[:-lag]) & np.isfinite(a[lag:])
    else:
        m = np.asarray(mask, dtype=bool)[: len(a) - lag]
        m = m & np.asarray(mask, dtype=bool)[lag:]
        m = m & np.isfinite(a[:-lag]) & np.isfinite(a[lag:])
    return float(np.corrcoef(a[:-lag][m], a[lag:][m])[0, 1]), int(m.sum())


print("=" * 78)
print("PHASE 1 - MEASUREMENT MODEL / ENTROPY PARTITION")
print("=" * 78)
print("N = %d contiguous 10s buckets (%.1f days)" % (N, N / 8640.0))

print("\n[1.1] Grid of the observation")
print("  P(bid integer) = %.4f" % np.mean(np.abs(y['best_bid'] - np.round(y['best_bid'])) < 0.5))
print("  P(spread odd)  = %.4f  -> mid lives on a .5 grid" % np.mean(y['spread'] % 2 == 1))

print("\n[1.2] Discreteness floor of the mid move")
s = np.abs(y['dmid'])
sm = s[np.isfinite(s)]
print("  |dmid| P(=0)=%.4f nonzero med=%.0f p90=%.0f p99=%.0f max=%.0f"
      % (np.mean(sm == 0), np.median(sm[sm > 0]), np.percentile(sm[sm > 0], 90),
         np.percentile(sm[sm > 0], 99), sm.max()))
print("  dmid sd=%.2f ticks = %.2f bps per 10s (mid~%.0f)"
      % (np.nanstd(y['dmid']), np.nanstd(y['dmid']) / np.median(y['mid']) * 1e4, np.median(y['mid'])))

print("\n[1.3] Bid-ask bounce: MA(1) on dmid")
x, z = H.lag(y['dmid'], 1), y['dmid']
m = np.isfinite(x) & np.isfinite(z)
sl, ic, rr, pv, _ = stats.linregress(x[m], z[m])
print("  dmid_t = %+.3f %+.4f dmid_(t-1)  R2=%.4f p=%.1e" % (ic, sl, rr ** 2, pv))
res = z[m] - (ic + sl * x[m])
print("  resid sd %.2f vs uncond %.2f -> explains %.2f%%"
      % (res.std(), z[m].std(), 100 * (1 - res.var() / z[m].var())))
gain = float(stats.norm.logpdf(z[m], ic + sl * x[m], res.std()).mean()
             - stats.norm.logpdf(z[m], z[m].mean(), z[m].std()).mean())
print("  NLL gain from the MA(1) MEAN alone: %.4f nats/step" % gain)
print("  -> Roll: cov(dmid_t, dmid_t-1) = -eta^2 => eta = %.1f IRT (median half-spread %.1f)"
      % (np.sqrt(abs(np.cov(x[m], z[m], ddof=0)[0, 1])), np.median(y['spread']) / 2))

print("\n[1.4] ACF profile of dmid (lags 1..12)")
acs = [ac(y['dmid'], k)[0] for k in range(1, 13)]
print("  " + "  ".join("L%d=%+.3f" % (k, a) for k, a in enumerate(acs, 1)))

print("\n[1.5] Staleness of the book at the decision instant")
st = y['stale_ms']
print("  p50=%.0fms p90=%.0fms p99=%.0fms" % (np.median(st), np.percentile(st, 90), np.percentile(st, 99)))

print("\n[1.6] Does the bounce depend on staleness? (quote-revision test)")
srt = np.argsort(st)
for i, g in enumerate(np.array_split(srt, 4)):
    msk = np.zeros(N, dtype=bool)
    msk[g] = True
    c1, k1 = ac(y['dmid'], 1, msk)
    ci, _ = cc(y['queue_imb'][g], y['dmid_n1'][g])
    print("  stale Q%d (med %5.0fms) n=%6d: ACF1(dmid)=%+.4f  corr(imb,dmid_n1)=%+.4f"
          % (i + 1, np.median(st[g]), k1, c1, ci))

print("\n[1.7] Decomposing dmid into quote moves vs level moves")
db, da = y['dbid'], y['dask']
both = (db != 0) & (da != 0)
onlyb = (db != 0) & (da == 0)
onlya = (db == 0) & (da != 0)
none = (db == 0) & (da == 0)
ok = np.isfinite(db) & np.isfinite(da)
print("  P(both)=%.4f onlybid=%.4f onlyask=%.4f neither=%.4f"
      % (np.mean(both[ok]), np.mean(onlyb[ok]), np.mean(onlya[ok]), np.mean(none[ok])))
for nm, msk in [("onlybid", onlyb), ("onlyask", onlya), ("both", both), ("neither", none)]:
    g = msk & np.isfinite(y['dmid'])
    print("    %-8s dmid med %+7.1f  sd %6.1f  n=%d"
          % (nm, np.median(y['dmid'][g]), np.std(y['dmid'][g]), g.sum()))

print("\n[1.8] Is the negative ACF a price effect or a quote-revision effect?")
step = y['dbid_step']
moved = np.isfinite(step) & (step > 0)
print("  P(best bid stepped)     = %.4f" % np.mean(moved[np.isfinite(step)]))
print("  |d bid| when stepped: med=%.0f p90=%.0f p99=%.0f"
      % (np.median(step[moved]), np.percentile(step[moved], 90), np.percentile(step[moved], 99)))
c1, k1 = ac(y['dmid'], 1)
c1s, k1s = ac(step, 1)
cq, _ = cc(step, y['dmid'])
print("  ACF1(dmid)        = %+.4f  (n=%d)" % (c1, k1))
print("  ACF1(|d bid|)     = %+.4f  (n=%d)  <- large = persistent repricing" % (c1s, k1s))
print("  corr(|d bid|,dmid)= %+.4f" % cq)
for thr in (1, 10, 50, 100, 200):
    msk = moved & (np.nan_to_num(step) > thr)
    c, kk = ac(y['dmid'], 1, msk)
    print("    ACF1(dmid | |dbid| > %3d) = %+.4f  n=%d" % (thr, c, kk))

print("\n[1.9] Volatility clustering")
for nm, a in [("|dmid|", np.abs(y['dmid'])), ("dmid^2", y['dmid'] ** 2), ("|dmid| rel", np.abs(y['dmid']) / y['mid'] * 1e4)]:
    cs = [ac(a, k)[0] for k in (1, 2, 5, 10, 20)]
    print("  %-11s ACF: " % nm + "  ".join("L%d=%+.3f" % (k, c) for k, c in zip((1, 2, 5, 10, 20), cs)))

print("\n[1.10] Zero-mass of the move: a discrete hurdle must be modelled explicitly")
nz = y['dmid'][np.isfinite(y['dmid'])]
print("  P(dmid=0) = %.4f" % np.mean(nz == 0))
print("  Among dmid==0: P(bid also unchanged)=%.4f P(ask also unchanged)=%.4f"
      % (np.mean(db[nz == 0] == 0), np.mean(da[nz == 0] == 0)))
