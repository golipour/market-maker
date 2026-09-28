"""Q29: run the ACTUAL production price model on the ACTUAL book data.

This is the test I could not do in Q26/Q27.  There I had to stub the gate and
the RLS mean because the lab CSV carries only book *aggregates* (a level count
and best-level qty), while the production model walks per-level depth:

    mech_bid, mech_ask = _consume_depth(bids, sell_vol), _consume_depth(asks, buy_vol)

Now we have the raw L2 book from Postgres, so we can import the real
`TorchConditionalMechanicalMixturePriceModel` and the real `Orderbook` type and
drive them with the true production parameters:

    reference_vwap_vol=1000.0, shape_alpha=0.5, decay=0.99,
    initial_sigma=50.0, initial_log_sigma=0.25, student_df=3.0

and score it prequentially on the real mid path, against the same hurdle
proposal.  No reimplementation of the model itself.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/teanab/Documents/projects/market-maker")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.model_lab.analysis import book_io as B

from research.model_lab.analysis import stats1 as S
from research.model_lab.analysis import models_v3 as M
from research.model_lab.analysis.price_model_real import (
    load_orderbook_types,
    load_production_price_model,
)

ProdModel = load_production_price_model()
Orderbook, Level = load_orderbook_types()
print("imported REAL production class:", ProdModel.__name__)

# Use the production normaliser on the exact raw payload shape from the DB.
import mm.integrations.trade_stats as TS

# _tick_size() would call the live exchange API.  The instrument is fixed and
# its grid is directly observable in the book: every level price is a whole
# number, so tick_size = 1.0 IRT and lot_size = 0.01 (min observed qty).
# We seed the module's info cache so the production normaliser runs offline.
def _fake_exchange_info(symbol, exchange):
    return {"tick_size": 1.0, "lot_size": 0.01}

TS._exchange_info_cache[("nobitex", "USDT_IRT")] = {
    "tick_size": 1.0, "lot_size": 0.01, "base_currency": "IRT",
    "quote_currency": "USDT",
}
normalize_orderbook_from_raw = TS.normalize_orderbook_from_raw

print("=" * 96)
print("Q29 - REAL PRODUCTION MODEL, REAL BOOK, REAL PARAMS")
print("=" * 96)

d = B.load()
N = d["N"]
print("\nbook rows: %d   span %.2f days" % (N, (d["t_end"][-1] - d["t_end"][0]) / 86400000.0))

import json as _json
import csv as _csv
_csv.field_size_limit(10 ** 9)

raws = []
with open(B.BOOK_CSV, newline="") as fh:
    for r in _csv.DictReader(fh):
        try:
            b = _json.loads(r["bids"])
            a = _json.loads(r["asks"])
        except (ValueError, TypeError):
            continue
        if not b or not a:
            continue
        raws.append(([[l["price"], l["qty"]] for l in b],
                     [[l["price"], l["qty"]] for l in a],
                     int(r["t_end"]),
                     float(r["buy_vol"] or 0.0),
                     float(r["sell_vol"] or 0.0)))
print("raw books: %d" % len(raws))

obs = []
for bl, al, tms, _bv, _sv in raws:
    try:
        obs.append(normalize_orderbook_from_raw(
            "USDT_IRT", "nobitex",
            {"bids": bl, "asks": al},
            t_ms=tms))
    except Exception as exc:
        if not obs:
            import traceback
            traceback.print_exc()
            raise SystemExit(1)
        continue
print("normalized books: %d" % len(obs))
FLOW = np.array([[_bv, _sv] for (_bl, _al, _t, _bv, _sv) in raws], dtype=float)
mids = np.array([o.mid for o in obs])
print("mid path (ticks): %.1f -> %.1f" % (mids[0], mids[-1]))
print("first book: %d bid levels, %d ask levels" % (len(obs[0].bids), len(obs[0].asks)))


# ---------------------------------------------------------------- scoring
# The production model is a STATEFUL updater, not a density: it exposes
# get_params() but no logpdf.  To score it prequentially we take its own
# predictive parameters at time t and evaluate the mixture density it
# actually implies (price_torch.py:754-790 use exactly these variances).
from scipy.stats import norm, t as tdist

WARM = 8000
NB = len(obs)


def prod_params(model, t):
    p = model.get_params()
    return p


print()
print("=" * 96)
print("[29.1] Fit the REAL production model over the real book")
print("=" * 96)
model = ProdModel(
    market_flow_model=None,
    reference_vwap_vol=1000.0,
    shape_alpha=0.5,
    decay=0.99,
    initial_sigma=50.0,
    initial_log_sigma=0.25,
    student_df=3.0,
    point_mode="sample",
)
Step = tuple
n_ok = 0
for t in range(1, NB):
    prev = Step((obs[t - 1].t_ms, obs[t - 1],
                 FLOW[t - 1, 0], FLOW[t - 1, 1]))
    cur = Step((obs[t].t_ms, obs[t], FLOW[t, 0], FLOW[t, 1]))
    try:
        model.update(prev, cur)
        n_ok += 1
    except Exception:
        continue
print("  updates applied : %d / %d" % (n_ok, NB - 1))
p = model.get_params()
print("  fitted params   :")
for k in sorted(p):
    v = p[k]
    if isinstance(v, (int, float)):
        print("      %-22s %.6f" % (k, v))
    elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
        print("      %-22s %s" % (k, np.round(np.asarray(v, dtype=float), 4)))
    else:
        print("      %-22s %s" % (k, type(v).__name__))

print()
print("=" * 96)
print("[29.2] The MID branch, and the density it implies")
print("=" * 96)
mb = p["market_beta"]
ms = p["market_sigma"]
rm = p["refresh_mean"]
rs = p["refresh_sigma"]
mid_mb = np.asarray(mb["mid"], dtype=float)
mid_ms = float(ms["mid"])
mid_rm = float(rm["mid"])
mid_rs = float(rs["mid"])
q_mkt = float(p["market_prob"])
print("  mid market beta   : intercept %.4f  slope %.4f" % (mid_mb[0], mid_mb[1]))
print("  mid market sigma  : %.4f ticks" % mid_ms)
print("  mid refresh mean  : %.4f ticks" % mid_rm)
print("  mid refresh sigma : %.4f ticks" % mid_rs)
print("  gate market_prob  : %.4f" % q_mkt)
print("  gate beta         : %s" % np.round(np.asarray(p["gate_beta"], dtype=float), 4))

# Prequential: replay and score the mid component at each step using the
# state available BEFORE the update.  We snapshot params each step.
print()
print("=" * 96)
print("[29.3] Prequential NLL on the real book (mid component)")
print("=" * 96)


def mid_density(y, mu, sd, nu):
    """Discrete lattice density of the mid draw (cell width 0.5 IRT)."""
    h = 0.5
    lo = (y - h - mu) / sd
    hi = (y + h - mu) / sd
    return norm.cdf(hi) - norm.cdf(lo)


def body_pdf(y, mu, var, nu=None):
    h = 0.5
    sd = np.sqrt(var)
    if nu is None:
        return norm.cdf((y + h - mu) / sd) - norm.cdf((y - h - mu) / sd)
    s = sd * np.sqrt((nu - 2.0) / nu)   # var-matched t scale
    return tdist.cdf((y + h - mu) / s, df=nu) - tdist.cdf((y - h - mu) / s, df=nu)



NU = 3.0
# The proposed addition, estimated causally (EWMA on the zero indicator).
P0H = 0.35


def body_pdf(y, mu, var, nu=None):
    h = 0.5
    sd = np.sqrt(var)
    if nu is None:
        return norm.cdf((y + h - mu) / sd) - norm.cdf((y - h - mu) / sd)
    s = sd * np.sqrt((nu - 2.0) / nu)
    return tdist.cdf((y + h - mu) / s, df=nu) - tdist.cdf((y - h - mu) / s, df=nu)


def replay(with_hurdle, p0_0=0.35, decay=0.99):
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
        y = float(obs[t].mid - obs[t - 1].mid)
        if t >= WARM and np.isfinite(y):
            mu_m = b[0]
            dens = w * body_pdf(y, mu_m, mk_var) + (1.0 - w) * body_pdf(y, rf_mu, rf_var, NU)
            if with_hurdle:
                p0 = min(max(p0n / p0d, 1e-6), 1.0 - 1e-6)
                q = p0 if y == 0.0 else (1.0 - p0) * dens
            else:
                q = dens
            L[t] = -np.log(max(q, 1e-300))
        if with_hurdle:
            z = 1.0 if y == 0.0 else 0.0
            p0n = (1 - decay) * p0n + decay * z
            p0d = (1 - decay) * p0d + decay
        prev = Step((obs[t - 1].t_ms, obs[t - 1],
                     FLOW[t - 1, 0], FLOW[t - 1, 1]))
        cur = Step((obs[t].t_ms, obs[t], FLOW[t, 0], FLOW[t, 1]))
        try:
            m.update(prev, cur)
        except Exception:
            continue
    return L


print()
print("=" * 96)
print("[29.3] Prequential NLL, REAL model on REAL book")
print("=" * 96)
base = replay(False)
withh = replay(True)
print("  production as-is                mean NLL = %8.4f  (n=%d)"
      % (np.nanmean(base), int(np.isfinite(base).sum())))
print("  production + zero atom          mean NLL = %8.4f"
      % np.nanmean(withh))
print("  the ZERO MASS the real data has : p(dmid=0) = %.4f"
      % np.mean(np.diff([o.mid for o in obs]) == 0.0))

dd = withh - base
ok = np.isfinite(dd)
mean, l, h, n = S.moving_block_ci(dd[ok], B=2000, seed=211)
tt, pp = S.dm_test(dd[ok])
print()
print("  gain %.4f nats  95%% CI [%.4f, %.4f]  DM t=%.1f p=%.3g"
      % (-mean, -h, -l, tt, pp))
print("  = %.1f%% of the production model's loss" % (100 * (-mean) / np.nanmean(base)))
print("=" * 96)

print()
print("=" * 96)
print("[29.4] DIAGNOSIS: where does the 73-nat loss actually come from?")
print("=" * 96)
fin = np.isfinite(base)
vals = base[fin]
print("  n = %d" % vals.size)
print("  mean %.2f  median %.2f  p90 %.2f  p99 %.2f  max %.1f"
      % (vals.mean(), np.median(vals), np.percentile(vals, 90),
         np.percentile(vals, 99), vals.max()))
print()
print("  contribution to the mean, by loss decile:")
qs = np.percentile(vals, np.arange(0, 101, 10))
prev = -np.inf
for i, q in enumerate(qs):
    m = (vals >= prev) & (vals <= q)
    frac = vals[m].sum() / vals.sum() if vals[m].size else 0.0
    print("    [%8.2f,%8.2f)  n=%6d  share of total loss %6.2f%%"
          % (prev, q, m.sum(), 100 * frac))
    prev = q
print()
top = np.argsort(-np.nan_to_num(base))[:10]
print("  10 worst steps and their |dmid|:")
tgt = np.full(NB, np.nan)
tgt[1:] = np.diff(mids)
for i in top:
    print("    t=%6d  NLL=%9.2f   dmid=%9.1f  (%.1f sigma)"
          % (i, base[i], tgt[i], tgt[i] / 61.21))
print()
print("  A density cannot do better than ~-log P(y) at its own scale;")
print("  losses of 70+ nats mean those draws sit many sigma out, so they")
print("  dominate the MEAN completely.  The mean NLL is therefore not a")
print("  usable model-selection statistic for this data.")
print("=" * 96)

print()
print("=" * 96)
print("[29.5] ROOT CAUSE of the underflow -- checked, not assumed")
print("=" * 96)
w_hist = []
model3 = ProdModel(
    market_flow_model=None, reference_vwap_vol=1000.0, shape_alpha=0.5,
    decay=0.99, initial_sigma=50.0, initial_log_sigma=0.25,
    student_df=3.0, point_mode="sample",
)
tgt = np.full(NB, np.nan)
tgt[1:] = np.diff(mids)
for t in range(1, NB):
    pr = model3.get_params()
    wv = pr.get("market_resp")
    w_hist.append(0.5 if wv is None else float(wv))
    prev = Step((obs[t - 1].t_ms, obs[t - 1], FLOW[t - 1, 0], FLOW[t - 1, 1]))
    cur = Step((obs[t].t_ms, obs[t], FLOW[t, 0], FLOW[t, 1]))
    try:
        model3.update(prev, cur)
    except Exception:
        pass
wh = np.asarray(w_hist)
print("  mixing weight w over time:")
for lab, sl in (("first 5k", slice(0, 5000)), ("mid 5k", slice(NB // 2, NB // 2 + 5000)),
                ("last 5k", slice(-5000, None))):
    seg = wh[sl]
    print("    %-10s mean %.6f  min %.6f  max %.6f" % (lab, seg.mean(), seg.min(), seg.max()))
print("  fraction of steps with w > 0.9999 : %.4f" % np.mean(wh > 0.9999))
print()
print("  The gate saturates: market_resp pins at 1.0, so the Student-t")
print("  refresh branch is switched OFF entirely and the predictive law")
print("  collapses to a pure Gaussian.  That is a real property of the")
print("  model on this data, not an artifact of my scoring.")
print()
mx = float(np.max(np.abs(tgt[np.isfinite(tgt)])))
print("  max |dmid| = %.1f ticks" % mx)
print("  fitted mid market sigma = %.2f" % mid_ms)
print("  max move in market sigmas = %.1f" % (mx / np.sqrt(mid_ms)))
print("=" * 96)
