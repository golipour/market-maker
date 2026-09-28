"""Q30: redo the zero-atom comparison ON THE BOOK DATA, with robust statistics.

Two corrections to the Q28 conclusion:

  1. Q28 scored on the CSV.  The live book is a different sample: it has a
     0.3124 zero rate (vs 0.3507) and, crucially, a much thinner empirical
     tail (max |dmid| 951 vs 3563).  The CSV's tail was doing a lot of the
     work in that comparison, so it must be re-run here.

  2. Q29 proved the MEAN NLL is unusable on this data: 84.8% of the total
     loss sits in the top decile, and 10% of steps underflow to the
     1e-300 clamp floor at exactly -log(1e-300) = 690.78 nats.  Ranking
     models by the mean here just counts underflows.  So we compare on
     ROBUST statistics (median, trimmed mean) and report the tail
     separately instead of pretending it is a mean.

Variants tested, all on the real production model and the real book:

  A  production as-is
  B  production + zero atom                     (the proposal)
  C  production + zero atom, Student-t branch forced on
  D  production + zero atom + Student-t branch forced on, nu = 2.5
"""
import os
import sys

import numpy as np
from scipy.stats import norm, t as tdist

sys.path.insert(0, "/Users/teanab/Documents/projects/market-maker")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.model_lab.analysis import stats1 as S
from research.model_lab.analysis.price_model_real import (
    load_production_price_model,
)

ProdModel = load_production_price_model()

import mm.integrations.trade_stats as TS

TS._exchange_info_cache[("nobitex", "USDT_IRT")] = {
    "tick_size": 1.0, "lot_size": 0.01, "base_currency": "IRT",
    "quote_currency": "USDT",
}
normalize_orderbook_from_raw = TS.normalize_orderbook_from_raw

import csv
import json

csv.field_size_limit(10 ** 9)
from research.model_lab.analysis import book_io as B

obs = []
FLOW = []
with open(B.BOOK_CSV, newline="") as fh:
    for r in csv.DictReader(fh):
        try:
            b = json.loads(r["bids"])
            a = json.loads(r["asks"])
        except (ValueError, TypeError):
            continue
        if not b or not a:
            continue
        bl = [[l["price"], l["qty"]] for l in b]
        al = [[l["price"], l["qty"]] for l in a]
        try:
            obs.append(normalize_orderbook_from_raw(
                "USDT_IRT", "nobitex", {"bids": bl, "asks": al},
                t_ms=int(r["t_end"])))
        except Exception:
            continue
        FLOW.append((float(r["buy_vol"] or 0.0), float(r["sell_vol"] or 0.0)))

NB = len(obs)
FLOW = np.asarray(FLOW, dtype=float)
MIDS = np.array([o.mid for o in obs], dtype=float)
TGT = np.full(NB, np.nan)
TGT[1:] = np.diff(MIDS)
WARM = 8000
H = 0.5
Step = tuple
print("=" * 96)
print("Q30 - ZERO ATOM ON THE BOOK DATA, ROBUST STATISTICS")
print("=" * 96)
print("\nbook rows %d | warm %d | scored %d" % (NB, WARM, NB - WARM))
print("P(dmid=0) = %.4f   sigma(dmid) = %.2f   max|dmid| = %.1f"
      % (np.mean(TGT[1:] == 0.0), np.nanstd(TGT[1:]), np.nanmax(np.abs(TGT[1:]))))


def lattice_gauss(y, mu, var):
    sd = np.sqrt(max(var, 1e-12))
    return norm.cdf((y + H - mu) / sd) - norm.cdf((y - H - mu) / sd)


def lattice_t(y, mu, var, nu):
    sd = np.sqrt(max(var, 1e-12))
    s = sd * np.sqrt((nu - 2.0) / nu)
    return tdist.cdf((y + H - mu) / s, df=nu) - tdist.cdf((y - H - mu) / s, df=nu)


def replay(hurdle=False, force_t=False, nu=3.0, w_floor=0.0, p0_0=0.31, decay=0.99):
    """One prequential pass.

    force_t : floor the branch weight at w_floor<1 so the Student-t refresh
              branch cannot be switched off (Q29 found it pins to 1.0).
    """
    m = ProdModel(
        market_flow_model=None, reference_vwap_vol=1000.0, shape_alpha=0.5,
        decay=0.99, initial_sigma=50.0, initial_log_sigma=0.25,
        student_df=3.0, point_mode="sample",
    )
    L = np.full(NB, np.nan)
    p0n, p0d = p0_0, p0_0
    for t in range(1, NB):
        pr = m.get_params()
        b = np.asarray(pr["market_beta"]["mid"], dtype=float)
        mk_var = float(pr["market_sigma"]["mid"] or 1.0)
        rf_mu = float(pr["refresh_mean"]["mid"] or 0.0)
        rf_var = float(pr["refresh_sigma"]["mid"] or 1.0)
        w = pr.get("market_resp")
        w = 0.5 if w is None else float(w)
        if force_t:
            w = min(w, 1.0 - w_floor)
        y = TGT[t]
        if t >= WARM and np.isfinite(y):
            mu_m = b[0]
            dens = (w * lattice_gauss(y, mu_m, mk_var)
                    + (1.0 - w) * lattice_t(y, rf_mu, rf_var, nu))
            if hurdle:
                p0 = min(max(p0n / p0d, 1e-6), 1.0 - 1e-6)
                q = p0 if y == 0.0 else (1.0 - p0) * dens
            else:
                q = dens
            L[t] = -np.log(max(q, 1e-300))
        if hurdle:
            z = 1.0 if y == 0.0 else 0.0
            p0n = (1 - decay) * p0n + decay * z
            p0d = (1 - decay) * p0d + decay
        prev = Step((obs[t - 1].t_ms, obs[t - 1], FLOW[t - 1, 0], FLOW[t - 1, 1]))
        cur = Step((obs[t].t_ms, obs[t], FLOW[t, 0], FLOW[t, 1]))
        try:
            m.update(prev, cur)
        except Exception:
            continue
    return L


VAR = [
    ("A  production as-is", dict()),
    ("B  + zero atom", dict(hurdle=True)),
    ("C  + zero atom + t-branch on 5%", dict(hurdle=True, force_t=True, w_floor=0.05)),
    ("D  + zero atom + t-branch on 20%", dict(hurdle=True, force_t=True, w_floor=0.20)),
    ("E  + zero atom + t-branch 20%, nu=2.5", dict(hurdle=True, force_t=True, w_floor=0.20, nu=2.5)),
    ("F  t-branch 20%, no zero atom", dict(force_t=True, w_floor=0.20)),
]
print()
print("[30.1] ROBUST comparison (median + 10% trimmed mean)")
print("-" * 96)
print("  %-40s %9s %10s %10s %9s" % ("variant", "median", "trim10%", "p99", "clamped"))
res = {}
for nm, kw in VAR:
    L = replay(**kw)
    res[nm] = L
    v = L[np.isfinite(L)]
    trim = np.mean(np.sort(v)[:int(0.9 * v.size)])
    clamped = np.mean(v >= 690.0)
    print("  %-40s %9.4f %10.4f %10.2f %8.2f%%"
          % (nm, np.median(v), trim, np.percentile(v, 99), 100 * clamped))

print()
print("[30.2] Median-NLL difference vs production (negative = better)")
print("-" * 96)
b = res["A  production as-is"]
for nm, _ in VAR[1:]:
    L = res[nm]
    ok = np.isfinite(L) & np.isfinite(b)
    d = L[ok] - b[ok]
    med = np.median(d)
    # block bootstrap on the MEDIAN difference
    idx = np.flatnonzero(ok)
    bl = int(np.sqrt(len(idx)))
    nb = len(idx) // bl
    blocks = np.array([d[i * bl:(i + 1) * bl] for i in range(nb)])
    boots = np.median(blocks, axis=1)
    rng = np.random.default_rng(271)
    samp = np.median(blocks[rng.integers(0, nb, size=(2000, nb))], axis=1)
    lo, hi = np.percentile(samp, [2.5, 97.5])
    print("  %-40s median d = %+0.4f  95%% CI [%+.4f,%+.4f]" % (nm, med, lo, hi))

print()
print("[30.3] Tail behaviour, stated separately from the mean")
print("-" * 96)
for nm, _ in VAR:
    v = res[nm][np.isfinite(res[nm])]
    print("  %-40s p99.9 %8.2f   max %8.2f   share of loss in top 1%%: %5.1f%%"
          % (nm, np.percentile(v, 99.9), v.max(),
             100 * v[v >= np.percentile(v, 99)].sum() / v.sum()))
print("=" * 96)
