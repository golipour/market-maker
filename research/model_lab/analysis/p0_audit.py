"""Phase 0: data audit on the 10s Nobitex USDT_IRT series."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from scipy import stats
from research.model_lab.analysis import harness as H

np.set_printoptions(suppress=True, linewidth=140)
d = H.load()
f = H.core_frame(d)
y = H.derived(f)

print("=" * 78)
print("PHASE 0 — DATA AUDIT")
print("=" * 78)
print(f"raw rows                 : {len(d['t_end']):,}")
print(f"contiguous 10s rows      : {len(f['mid']):,}")
print(f"gaps skipped             : {f['n_gaps']} (max gap {f['max_gap_ms']/1000:.0f}s)")
span = (d['t_end'][-1] - d['t_end'][0]) / 86400000.0
print(f"span                     : {span:.2f} days  {d['ts'][0]} -> {d['ts'][-1]}")

mid = f['mid']; spread = f['spread']
print("\n--- price level / grid ---")
print(f"mid  min/med/max        : {mid.min():.0f} / {np.median(mid):.0f} / {mid.max():.0f}")
uq = np.unique(np.round(np.diff(np.unique(mid))))
print(f"distinct mid increments  : {uq[:12]}  (n={len(uq)})")
uq2 = np.unique(np.diff(np.unique(f['best_bid'])))
print(f"distinct bid increments  : {uq2[:12]}")
print(f"tick size (mode)         : {np.bincount(uq2.astype(int)).argmax() if uq2.min()>=1 else 'nonunit'}")

print("\n--- spread ---")
sp = f['spread']
print(f"spread min/p10/med/p90/max: {sp.min():.0f} / {np.percentile(sp,10):.0f} / {np.median(sp):.0f} / {np.percentile(sp,90):.0f} / {sp.max():.0f}")
print(f"relative spread bps med  : {np.median(sp/mid)*1e4:.1f}   p90 {np.percentile(sp/mid,90)*1e4:.1f}")
print(f"P(spread==0)             : {(sp==0).mean():.4f}")
q = np.quantile(sp, [0.28, 0.84])
print(f"spread terciles: tight<{q[0]:.0f} ({np.mean(sp<q[0]):.0%}), mid ({np.mean((sp>=q[0])&(sp<q[1])):.0%}), wide>={q[1]:.0f}")

print("\n--- flow ---")
tot = y['tot']
print(f"P(total==0)              : {(tot==0).mean():.4f}")
pos = tot[tot > 0]
print(f"total>0  n={len(pos):,}  med={np.median(pos):.0f} p90={np.percentile(pos,90):.0f} p99={np.percentile(pos,99):.0f} max={pos.max():.0f}")
print(f"buy  P(=0)               : {(y['bv']==0).mean():.4f}")
print(f"sell P(=0)               : {(y['sv']==0).mean():.4f}")
print(f"P(buy=0 & sell=0)        : {((y['bv']==0)&(y['sv']==0)).mean():.4f}")
print(f"corr(bv,sv)              : {np.corrcoef(y['bv'],y['sv'])[0,1]:.4f}")
print(f"trade_count: mean={y['tc'].mean():.2f} med={np.median(y['tc']):.0f} p99={np.percentile(y['tc'],99):.0f}")
print(f"log1p(total) skew/kurt  : {stats.skew(np.log1p(tot)):.3f} / {stats.kurtosis(np.log1p(tot)):.3f}")

print("\n--- staleness (book age at decision time) ---")
st = y['stale_ms']
print(f"stale_ms min/p10/med/p90/max: {st.min():.0f} / {np.percentile(st,10):.0f} / {np.median(st):.0f} / {np.percentile(st,90):.0f} / {st.max():.0f}")
print(f"P(stale==0)              : {(st==0).mean():.4f}")
print(f"P(stale>1000ms)          : {(st>1000).mean():.4f}   P(>5000ms) {(st>5000).mean():.4f}")
print(f"corr(stale, |dmid| next) : {np.corrcoef(st[:-1], np.abs(y['dmid']))[0,1]:.4f}")
print(f"corr(stale, total next)  : {np.corrcoef(st[:-1], tot[1:])[0,1]:.4f}")

print("\n--- microprice / imbalance ---")
me = y['micro_minus_mid']
print(f"microprice-mid: med={np.median(me):.2f} sd={me.std():.2f}  (half-spread med {np.median(sp)/2:.1f})")
print(f"corr(micro-mid, dmid next) : {np.corrcoef(me[:-1], y['dmid'])[0,1]:.4f}")
qi = y['queue_imb']
print(f"queue_imb (bidqty-askqty)/(sum): med={np.median(qi):.3f} sd={qi.std():.3f}")
print(f"corr(imb, dmid next)     : {np.corrcoef(y['imb'][:-1], y['dmid'])[0,1]:.4f}")
print(f"corr(imb, |dmid| next)   : {np.corrcoef(y['imb'][:-1], np.abs(y['dmid']))[0,1]:.4f}")
print(f"corr(imb, total next)    : {np.corrcoef(y['imb'][:-1], tot[1:])[0,1]:.4f}")
print(f"corr(stale, imb)         : {np.corrcoef(st, qi)[0,1]:.4f}  <-- staleness inflates apparent imbalance")

print("\n--- depth ---")
print(f"bid_depth_levels uniq    : {np.unique(f['bid_depth_levels'])[:8]}")
print(f"ask_depth_levels uniq    : {np.unique(f['ask_depth_levels'])[:8]}")
print(f"best_bid_qty  p10/med/p90/max: {np.percentile(f['best_bid_qty'],10):.0f}/{np.median(f['best_bid_qty']):.0f}/{np.percentile(f['best_bid_qty'],90):.0f}/{f['best_bid_qty'].max():.0f}")
print(f"best_ask_qty  p10/med/p90/max: {np.percentile(f['best_ask_qty'],10):.0f}/{np.median(f['best_ask_qty']):.0f}/{np.percentile(f['best_ask_qty'],90):.0f}/{f['best_ask_qty'].max():.0f}")
print(f"ratio ask_qty/bid_qty med: {np.median(f['best_ask_qty']/np.maximum(f['best_bid_qty'],1e-9)):.2f}  <-- asymmetric venue")
print(f"corr(ask_qty/bid_qty, dmid next): {np.corrcoef((f['best_ask_qty']/np.maximum(f['best_bid_qty'],1e-9))[:-1], y['dmid'])[0,1]:.4f}")
