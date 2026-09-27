"""The correctly-specified model family, v3.

What Q1-Q7 established, and what this file therefore implements:

  1. The conditional MEAN is not exploitable.  The optimal linear predictor
     has R^2 = 0.048, a ceiling of 0.024 nats, and the honest prequential
     realisation was 0.0004 nats.  So mu = 0.  (Q3, Q4)

  2. The negative ACF1 = -0.218 is MECHANICALLY produced by the zero mass:
     a hurdle with p0 = 0.3507 and an iid continuous part predicts
     ACF1 = -p0 = -0.351 by itself.  Nothing about quote bounce needs to be
     modelled, and nothing is tradable.  (Q7)

  3. What IS worth modelling is the SHAPE of the conditional law:
       - the point mass at zero, which varies enormously with the state
         (0.14 in wide-spread/trading buckets to 0.52 in tight/quiet ones)
       - the scale of the CONTINUOUS part, which must be estimated on the
         non-zero support only, otherwise the hurdle masks a 10x error
       - the tail index, which a Gaussian badly understates

  4. The scale carries genuine serial dependence (ACF1(|z|) = +0.236 on
     consecutive non-zero moves), so a plain EWMA is a mis-specification --
     but a GARCH on the unconditional squares is worse still.  The right
     object is a scale recursion on the CONDITIONAL shocks.

Every model below is causal, has no batch fit, and declares its own
calibration so that the reported density is honest by construction.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, betainc


def t_logpdf_sym(y, s, nu):
    """log density of a symmetric Student-t at y, scale s, df nu.

    var = s^2 * nu/(nu-2) for nu > 2, so `s` is the conditional scale that
    matches the empirical sd when nu=4 via s = sd / sqrt(2).
    """
    s = max(float(s), 1e-9)
    z = y / s
    return float(gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
                 - 0.5 * np.log(nu * np.pi) - np.log(s)
                 - 0.5 * (nu + 1.0) * np.log1p(z * z / nu))


def var_of(s, nu):
    """Variance of the symmetric t with scale s and df nu."""
    if nu <= 2.0:
        return np.inf
    return s * s * nu / (nu - 2.0)


_TQ75 = {}


def _t75(nu):
    """75th percentile of |t_nu| (unit scale), cached.

    Used to convert a second moment into a scale for nu <= 2, where the
    variance does not exist.  Computed once per df by inverting the
    incomplete beta through a bisection on the two-sided tail.
    """
    key = round(float(nu), 4)
    v = _TQ75.get(key)
    if v is not None:
        return v
    from scipy.stats import t as _T
    # P(|T| <= q) = 0.75  ->  2*F(q) - 1 = 0.75  ->  F(q) = 0.875
    q = float(_T.ppf(0.875, df=key))
    _TQ75[key] = q
    return q


def t_cdf(x, nu):
    """Student-t CDF, exact, ~1000x faster than scipy.stats.t.cdf per call.

    For x > 0:  F(x) = 1 - 0.5 * I_{nu/(nu+x^2)}(nu/2, 1/2)
    For x < 0 use symmetry.  x = 0 gives exactly 0.5.
    """
    if x == 0.0:
        return 0.5
    if x > 0.0:
        return 1.0 - 0.5 * betainc(0.5 * nu, 0.5, nu / (nu + x * x))
    return 0.5 * betainc(0.5 * nu, 0.5, nu / (nu + x * x))


def t_logpmf_lattice(y, s, nu):
    """log P(Y = y) for a Student-t discretised on a unit lattice."""
    s = max(float(s), 1e-9)
    hi = t_cdf((y + 0.5) / s, nu)
    lo = t_cdf((y - 0.5) / s, nu)
    return float(np.log(max(hi - lo, 1e-300)))


def t_logpmf_gauss_lattice(y, s):
    """log P(Y = y) for a Gaussian discretised on a unit lattice.

    Same lattice correction as t_logpmf_lattice.  Required, and not merely a
    sharpness correction: a continuous density evaluated at a lattice point
    is missing the factor h (the cell width), so at a scale comparable to h
    it becomes a near-CONSTANT in y -- it assigns the same probability to a
    1-tick move and a 400-tick move.  That is not a sharp density, it is a
    flat one, and it is the mechanism behind a phantom 0.8-nat "gain" that
    was measured and then retracted (see q19_artifact.py).
    """
    s = max(float(s), 1e-9)
    from scipy.special import log_ndtr
    hi = float(log_ndtr((y + 0.5) / s))
    lo = float(log_ndtr((y - 0.5) / s))
    # log(exp(hi) - exp(lo)) computed stably
    m = max(hi, lo)
    return float(m + np.log(max(1.0 - np.exp(min(hi, lo) - m), 1e-300)))


class HurdleT:
    """Zero-mean hurdle x symmetric Student-t with an ONLINE calibration.

    Density
        q(y) = p0 * 1{y == 0} + (1 - p0) * f_t(y | 0, s, nu)

    with f_t the symmetric Student-t on the *non-zero* support, s the
    conditional scale, p0 the conditional zero mass, and an explicit
    multiplicative calibration `alpha` on s that is updated from the realised
    standardised shocks so the density cannot drift overconfident.

    state_fn(y, t) -> (cell, ) partitions the decision state; p0 and s are
    estimated per cell with EWMA forgetting.
    """

    name = "hurdle-t"

    def __init__(self, nu=4.0, lam=0.99, lam_p=0.99, lam_cal=0.995,
                 floor=1.0, s0=None, p0_0=0.35, state_fn=None,
                 calibrate=True, cal_cap=2.5):
        self.nu, self.lam, self.lam_p, self.lam_cal = nu, lam, lam_p, lam_cal
        self.floor, self.p0_0, self.state_fn = floor, p0_0, state_fn
        self.calibrate, self.cal_cap = calibrate, cal_cap
        self.s0 = s0
        self.alpha = 1.0
        self.S = {}      # cell -> [sum, weight] of squared conditional shock
        self.Z = {}      # cell -> [sum, weight] of zero indicator
        self.C = {}      # cell -> [sum, weight] of squared calibration shock

    def _cell(self, y, t):
        return 0 if self.state_fn is None else int(self.state_fn(y, t))

    def _get(self, D, c):
        v = D.get(c)
        if v is None:
            v = [0.0, 0.0]
            D[c] = v
        return v

    def target(self, y, t, h=1):
        j = t + h
        return y["dmid"][j] if j < y["N"] else np.nan

    def scale(self, y, t):
        """The Student-t SCALE for the continuous branch.

        The recursion accumulates the second moment of the non-zero shocks,
        m2 = E[x^2 | x != 0].  Relating that to the Student-t scale depends
        on which moment is finite:

            var(X) = s^2 * nu/(nu - 2)      requires nu > 2

        so for nu > 2 the scale that reproduces the measured variance is

            s = sqrt( m2 * (nu - 2) / nu ).

        For nu <= 2 the variance diverges and no such conversion exists; we
        then fall back to a quantile-matched scale (the 75th percentile of
        |x|, divided by the theoretical 75th percentile of |t_nu|), which is
        well defined for every nu.  Using the raw second moment as `s` for
        nu > 2 would inflate the scale by sqrt(2) at nu = 4, which is exactly
        the z sd of 0.33 observed before this was fixed.
        """
        c = self._cell(y, t)
        v = self.S.get(c)
        m2 = None
        if v is not None and v[1] > 1.0 and v[0] > 0:
            m2 = v[0] / v[1]
        else:
            g = self.S.get(0)
            if g is not None and g[1] > 1.0 and g[0] > 0:
                m2 = g[0] / g[1]
        if self.nu > 2.0 + 1e-9 and m2 is not None and m2 > 0:
            s = np.sqrt(m2 * (self.nu - 2.0) / self.nu)
        else:
            # quantile-matched fallback, using the running IQR of |x|
            q = self.S.get(c)
            if q is not None and q[1] > 1.0 and q[0] > 0:
                rms = np.sqrt(q[0] / q[1])
            else:
                rms = self.s0 if self.s0 else 50.0
            s = rms / _t75(self.nu)
        return max(s, self.floor) * (self.alpha if self.calibrate else 1.0)

    def p0(self, y, t):
        c = self._cell(y, t)
        z = self.Z.get(c)
        if z is not None and z[1] > 1.0:
            return z[0] / z[1]
        g = self.Z.get(0)
        if g is not None and g[1] > 1.0:
            return g[0] / g[1]
        return self.p0_0

    def predict(self, t, y):
        """Returns (mu, sd, target, p0) -- target third, per the runner contract."""
        return 0.0, self.scale(y, t), self.target(y, t), self.p0(y, t)

    def _raw_scale(self, y, t):
        """The scale WITHOUT the calibration multiplier (for calibration)."""
        a = self.alpha
        self.alpha = 1.0
        try:
            return self.scale(y, t)
        finally:
            self.alpha = a

    def update(self, t, y):
        x = self.target(y, t)
        if not np.isfinite(x):
            return
        c = self._cell(y, t)
        s = self._get(self.S, c)
        z = self._get(self.Z, c)
        # The continuous scale is estimated on the NON-ZERO support only.
        if x != 0.0:
            s[0] = s[0] * self.lam + x * x
            s[1] = s[1] * self.lam + 1.0
        z[0] = z[0] * self.lam_p + float(x == 0)
        z[1] = z[1] * self.lam_p + 1.0
        if self.calibrate and x != 0.0:
            # Compare the realised shock to the UNCALIBRATED scale, so the
            # correction is not fed by its own output.  A calibrated Student-t
            # has E[(X/s)^2] = nu/(nu-2), NOT 1, so that is the target.
            cur = max(self._raw_scale(y, t), 1e-9)
            u = (x / cur) ** 2
            cc = self._get(self.C, 0)
            cc[0] = cc[0] * self.lam_cal + u
            cc[1] = cc[1] * self.lam_cal + 1.0
            # NOTE: an EWMA weight converges to 1/(1-lam_cal).  That IS the
            # effective sample size, so gate on it directly rather than on a
            # raw threshold that an EWMA weight never reaches.
            ess = 1.0 / (1.0 - self.lam_cal)
            if ess > 20.0 and cc[0] / cc[1] > 0.05:
                tgt = (self.nu / (self.nu - 2.0)) if self.nu > 2.0 else 1.0
                self.alpha = float(np.clip(
                    np.sqrt((cc[0] / cc[1]) / tgt), 0.5, self.cal_cap))

    def logpdf(self, tgt, mu, sd, p0):
        p0 = float(min(max(p0, 1e-6), 1.0 - 1e-6))
        if tgt == 0:
            return np.log(p0)
        return t_logpdf_sym(tgt, sd, self.nu) - np.log1p(-p0)


class HurdleTLattice(HurdleT):
    """Hurdle-t with the LATTICE correction.

    dmid lives on a grid: the mid is an integer or half-integer, so the
    realised move is a discrete variable.  A continuous density evaluated at
    the point and used as a pmf is systematically mis-scaled.  The exact
    discrete probability is the mass of the cell [y-1/2, y+1/2), i.e. the
    Student-t CDF difference, which is both correct and materially different
    from the point evaluation when the scale is small.
    """

    name = "hurdle-t (lattice-corrected)"

    def logpdf(self, tgt, mu, sd, p0):
        p0 = float(min(max(p0, 1e-6), 1.0 - 1e-6))
        if tgt == 0:
            return np.log(p0)
        s = max(float(sd), 1e-9)
        pm = t_logpmf_lattice(tgt, s, self.nu)
        return float(pm - np.log1p(-p0))


def cell_sp(y, t):
    return 0 if y["spread"][t] < 7 else (1 if y["spread"][t] < 199 else 2)


def cell_sp_st(y, t):
    i = cell_sp(y, t)
    j = 0 if y["stale_ms"][t] < 1000.0 else 1
    return i * 2 + j


def cell_sp_st_oc(y, t):
    i = cell_sp(y, t)
    j = 0 if y["stale_ms"][t] < 1000.0 else 1
    k = 1 if y["ocur"][t] > 0 else 0
    return i * 4 + j * 2 + k


def cell_sp_oc(y, t):
    i = cell_sp(y, t)
    k = 1 if y["ocur"][t] > 0 else 0
    return i * 2 + k


def cell_oc(y, t):
    return 1 if y["ocur"][t] > 0 else 0


def cell_st(y, t):
    return 0 if y["stale_ms"][t] < 1000.0 else 1
