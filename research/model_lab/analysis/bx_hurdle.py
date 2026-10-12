"""Hurdle price models on the tick lattice.

    P(Y = y) = pi0 * 1{y = 0}
             + (1 - pi0) * [F(y + h/2) - F(y - h/2)] / (1 - [F(h/2) - F(-h/2)])

h is the OBSERVED tick width, not a hard-coded 1.0.  With mu = 0 the zero cell
is [-h/2, h/2) so its probability under the symmetric body is 2*F(h/2) - 1.

The body scale is estimated ROBUSTLY.  A second-moment (rms) match is not usable
here: on Nobitex BTC_IRT the median |dmid| is 1e5 IRT and the tail is heavy, so
rms is set by the top percent of prints and the predictive scale is inflated by
orders of magnitude.  We quantile-match on the running EWMA median of |dmid|,
which is bounded by construction, and let a multiplicative calibration correct
any residual bias.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import t as _tdist

from research.model_lab.analysis.bx_price_models import t_cdf

_TQ_CACHE: dict[float, float] = {}


def t_q(nu: float, p: float) -> float:
    """Quantile of |t_nu| at level p (cached)."""
    key = (round(float(nu), 4), round(float(p), 4))
    v = _TQ_CACHE.get(key)
    if v is None:
        v = float(_tdist.ppf(0.5 + p / 2.0, df=float(nu)))
        _TQ_CACHE[key] = v
    return v


class HurdleLattice:
    """Zero-mean hurdle, Student-t body, lattice cell width h = tick."""

    name = "hurdle-t (lattice)"

    def __init__(self, *, nu=3.0, lam_p=0.99, lam_m=0.99, p0_0=0.5,
                 med0=20.0, med_p=0.50, cells=None, calibrate=True,
                 lam_cal=0.995, cal_lo=0.4, cal_hi=4.0, s_min_ratio=0.25):
        self.nu = float(nu)
        self.lam_p = float(lam_p)
        self.lam_m = float(lam_m)
        self.p0_0 = float(p0_0)
        self.med0 = float(med0)
        self.med_p = float(med_p)
        self.cells = cells
        self.calibrate = calibrate
        self.lam_cal = float(lam_cal)
        self.cal_lo = float(cal_lo)
        self.cal_hi = float(cal_hi)
        self.s_min_ratio = float(s_min_ratio)
        self.h = 1.0
        self.alpha = 1.0
        self.Z: dict[int, list[float]] = {}
        self.M: dict[int, list[float]] = {}
        self.R: dict[int, list[float]] = {}

    # -- state cells -------------------------------------------------------
    def _cell(self, y, t):
        if self.cells is None:
            return 0
        return int(self.cells(y, t))

    @staticmethod
    def _acc(d, c, lam, val):
        v = d.get(c)
        if v is None:
            v = [0.0, 0.0]
            d[c] = v
        v[0] = v[0] * lam + val
        v[1] = v[1] * lam + 1.0
        return v

    # -- predictive parameters --------------------------------------------
    def _quant(self):
        return t_q(self.nu, self.med_p)

    def _med(self, y, t):
        c = self._cell(y, t)
        v = self.M.get(c)
        if v is not None and v[1] > 1.0:
            return v[0] / v[1]
        g = self.M.get(0)
        if g is not None and g[1] > 1.0:
            return g[0] / g[1]
        return self.med0

    def scale(self, y, t):
        m = self._med(y, t)
        s = m / max(self._quant(), 1e-9)
        s = max(s, self.s_min_ratio * self.h)
        return s * (self.alpha if self.calibrate else 1.0)

    def p0(self, y, t):
        c = self._cell(y, t)
        z = self.Z.get(c)
        if z is not None and z[1] > 1.0:
            return min(max(z[0] / z[1], 1e-4), 1.0 - 1e-4)
        g = self.Z.get(0)
        if g is not None and g[1] > 1.0:
            return min(max(g[0] / g[1], 1e-4), 1.0 - 1e-4)
        return self.p0_0

    def predict(self, y, t):
        return {"p0": self.p0(y, t), "s": self.scale(y, t), "nu": self.nu,
                "alpha": self.alpha, "h": self.h}

    def _zero_cell_mass(self, s):
        s = max(float(s), 1e-12)
        return float(2.0 * t_cdf(0.5 * self.h / s, self.nu) - 1.0)

    def logpmf(self, tgt, p):
        s = max(float(p["s"]), 1e-12)
        h = float(p["h"])
        p0 = min(max(float(p["p0"]), 1e-6), 1.0 - 1e-6)
        zc = self._zero_cell_mass(s)
        denom = max(1.0 - zc, 1e-12)
        if tgt == 0:
            return math.log(p0 + (1.0 - p0) * zc)
        hi = t_cdf((tgt + 0.5 * h) / s, self.nu)
        lo = t_cdf((tgt - 0.5 * h) / s, self.nu)
        body = max(hi - lo, 0.0) / denom
        if body <= 0.0:
            return -700.0
        return math.log1p(-p0) + math.log(body)

    # -- adaptation --------------------------------------------------------
    def bind(self, y):
        self.h = float(y["tick"])

    def update(self, y, t):
        tgt = y["dmid_n1"][t]
        if not np.isfinite(tgt):
            return
        c = self._cell(y, t)
        if tgt != 0.0:
            self._acc(self.M, c, self.lam_m, abs(tgt))
        self._acc(self.Z, c, self.lam_p, 1.0 if tgt == 0.0 else 0.0)
        if self.calibrate and tgt != 0.0:
            # ratio of the realised |x| to the scale the model just predicted.
            saved = self.alpha
            self.alpha = 1.0
            try:
                raw = self.scale(y, t)
            finally:
                self.alpha = saved
            cc = self.R.setdefault(c, [0.0, 0.0])
            cc[0] = cc[0] * self.lam_cal + abs(tgt) / max(raw, 1e-12)
            cc[1] = cc[1] * self.lam_cal + 1.0
            if cc[1] > 1.0 / (1.0 - self.lam_cal):
                q = max(self._quant(), 1e-9)
                self.alpha = float(np.clip(cc[0] / cc[1] * q,
                                           self.cal_lo / q, self.cal_hi / q))


# -- state partitions ------------------------------------------------------
def cell_none(y, t):
    return 0


def cell_spread(y, t):
    sp = y["spread"][t]
    if not np.isfinite(sp):
        return 0
    h = y["tick"]
    return 0 if sp <= 3 * h else (1 if sp <= 8 * h else 2)


def cell_spread_trade(y, t):
    return cell_spread(y, t) * 2 + (1 if y["tot"][t] > 0 else 0)


def cell_spread_trade_stale(y, t):
    base = cell_spread_trade(y, t)
    st = y["stale_ms"][t]
    return base * 2 + (1 if (np.isfinite(st) and st > 2000.0) else 0)


CELL_SETS = {
    "global": None,
    "spread": cell_spread,
    "spread+trade": cell_spread_trade,
    "spread+trade+stale": cell_spread_trade_stale,
}


def cell_move_size(y, t):
    """Partition on the RECENT realised move size (causal, uses rows <= t).

    BTC_IRT alternates between 'book barely moves' and 'large repricing'
    intervals.  A single global scale cannot express both, but conditioning the
    scale on the magnitude of the last few moves does.  We use the EWMA mean
    of |dmid| up to t, bucketed geometrically.
    """
    import numpy as _np
    m = abs(y["dmid"][t]) if _np.isfinite(y["dmid"][t]) else 0.0
    base = 4.0 * y["tick"]
    if m <= base:
        return 0
    r = m / base
    return 1 if r <= 32 else (2 if r <= 1024 else 3)


CELL_SETS["move_size"] = cell_move_size


class HurdleLogNormal(HurdleLattice):
    """Hurdle at zero + LOG-NORMAL body on |dmid|, lattice-integrated.

    Motivated by the empirical shape: on Nobitex BTC_IRT the distribution of
    log|dmid| has excess kurtosis -1.68, i.e. it is BIMODAL (a 'book barely
    moves' mode near 1 IRT and a 'large repricing' mode near 1e6 IRT).  A
    Student-t is unimodal by construction, so it cannot represent that.  A
    log-normal is not bimodal either, but it at least puts the whole mass on a
    multiplicative scale where the observed spread lives (10^-0.3 .. 10^7.5),
    which the t in raw price units cannot do.

    Cell mass uses the same identity as the production flow pmf:
        P(k - h/2 <= X <= k + h/2) = Phi((ln(k+h/2) - mu)/sig)
                                   - Phi((ln(k-h/2) - mu)/sig)
    renormalised over [h/2, inf) so the pmf sums to one.
    """

    name = "hurdle-lognormal (lattice)"

    def __init__(self, *, lam_p=0.99, lam_m=0.99, p0_0=0.5, logm0=1.0,
                 logsig0=1.0, cells=None, s_min_ratio=0.25):
        super().__init__(lam_p=lam_p, lam_m=lam_m, p0_0=p0_0, cells=cells,
                         calibrate=False, s_min_ratio=s_min_ratio)
        self.logm0 = float(logm0)
        self.logsig0 = float(logsig0)
        self.LM: dict[int, list[float]] = {}
        self.L2: dict[int, list[float]] = {}

    def _log_stats(self, y, t):
        c = self._cell(y, t)
        v, v2 = self.LM.get(c), self.L2.get(c)
        if v is not None and v[1] > 1.0:
            m = v[0] / v[1]
            var = max(v2[0] / v2[1] - m * m, 1e-8)
            return m, math.sqrt(var)
        g, g2 = self.LM.get(0), self.L2.get(0)
        if g is not None and g[1] > 1.0:
            m = g[0] / g[1]
            var = max(g2[0] / g2[1] - m * m, 1e-8)
            return m, math.sqrt(var)
        return self.logm0, self.logsig0

    def predict(self, y, t):
        p0 = self.p0(y, t)
        mu, sg = self._log_stats(y, t)
        return {"p0": p0, "mu": mu, "sg": sg, "h": self.h}

    def _log_diff(self, a, b):
        from scipy.special import log_ndtr
        la = float(log_ndtr(a))
        lb = float(log_ndtr(b))
        if la < lb:
            la, lb = lb, la
        d = lb - la
        return la + math.log(max(1.0 - math.exp(d), 1e-300)) if d > -700.0 else -700.0

    def logpmf(self, tgt, p):
        from scipy.special import log_ndtr
        h = float(p["h"])
        mu = float(p["mu"])
        sg = max(float(p["sg"]), 1e-6)
        p0 = min(max(float(p["p0"]), 1e-6), 1.0 - 1e-6)
        lo_edge = 0.5 * h
        # renormaliser: mass of [lo_edge, inf) under LogNormal(mu, sg)
        z0 = (math.log(max(lo_edge, 1e-300)) - mu) / sg
        log_denom = float(log_ndtr(-z0))
        if log_denom >= -1e-8:
            return math.log(p0)
        if tgt == 0:
            # zero atom plus whatever continuous mass falls in [0, h/2)
            z1 = (math.log(max(lo_edge, 1e-300)) - mu) / sg
            return math.log(p0)
        a = (math.log(max(abs(tgt) - 0.5 * h, 1e-300)) - mu) / sg
        b = (math.log(abs(tgt) + 0.5 * h) - mu) / sg
        cell = self._log_diff(b, a)
        if cell <= -700.0:
            return -700.0
        return math.log1p(-p0) + cell - log_denom

    def update(self, y, t):
        tgt = y["dmid_n1"][t]
        if not np.isfinite(tgt):
            return
        c = self._cell(y, t)
        self._acc(self.Z, c, self.lam_p, 1.0 if tgt == 0.0 else 0.0)
        if tgt == 0.0:
            return
        a = abs(float(tgt))
        lx = math.log(max(a, 0.5 * self.h))
        v = self.LM.setdefault(c, [0.0, 0.0])
        v[0] = v[0] * self.lam_m + lx
        v[1] = v[1] * self.lam_m + 1.0
        v2 = self.L2.setdefault(c, [0.0, 0.0])
        v2[0] = v2[0] * self.lam_m + lx * lx
        v2[1] = v2[1] * self.lam_m + 1.0
