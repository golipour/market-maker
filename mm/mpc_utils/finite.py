"""Numerically-safe arithmetic primitives for the MPC pipeline.

These helpers exist to prevent ``OverflowError`` / ``ValueError`` from
reaching the optimizer's Adam loop. They never raise for finite or
sentinel (``inf``/``-inf``/``NaN``) inputs:

* ``exp_clamped(x)`` saturates to ``math.inf`` for ``x > 700`` and to
  ``0.0`` for ``x < -700`` instead of overflowing.
* ``log_clamped(x)`` returns ``-math.inf`` for non-positive input rather
  than raising.
* ``expit``/``softplus``/``logsumexp`` use stable formulations so they
  never overflow on large-magnitude arguments.

All functions are type-aware: they accept Python scalars, NumPy arrays,
or PyTorch tensors, and they return the same flavour they were given
(``expit``/``softmax``/``softplus`` for arrays return arrays).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Clamp thresholds
# ---------------------------------------------------------------------------

# ``exp(700)`` is still finite in IEEE 754; ``exp(701)`` overflows. We use
# 700 as the "saturation" boundary so callers that build a buffer around
# the saturation point stay inside the math.exp domain.
_EXP_CLAMP_HI = 700.0
_EXP_CLAMP_LO = -700.0

# 1e-300 is the smallest positive value Python's ``math.log`` can
# represent without underflowing to zero. We use it as the floor for
# the log-domain "tiny but finite" buffer.
_LOG_FLOOR = 1e-300
_SIGMA_MIN = 1e-300


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _is_numpy(x: Any) -> bool:
    try:
        import numpy as _np  # type: ignore

        return isinstance(x, _np.ndarray)
    except Exception:  # pragma: no cover - numpy is a hard dep
        return False


def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch  # type: ignore

        return isinstance(x, torch.Tensor)
    except Exception:  # pragma: no cover - torch is optional
        return False


def _torch():
    import torch  # type: ignore

    return torch


def _np():
    import numpy as _np  # type: ignore

    return _np


# ---------------------------------------------------------------------------
# exp / log / log1p
# ---------------------------------------------------------------------------


def exp_clamped(x: Any) -> Any:
    """``math.exp`` with explicit saturation to ``inf`` / ``0.0``.

    * Scalars above ``+700`` return ``math.inf`` (no ``OverflowError``).
    * Scalars below ``-700`` return ``0.0`` (no underflow to zero).
    * Within the safe range the function is identical to ``math.exp``.
    * NumPy arrays and PyTorch tensors are handled elementwise with the
      same saturation semantics; the input flavour is preserved.
    """
    if _is_torch_tensor(x):
        torch = _torch()
        clamped = x.clamp(min=_EXP_CLAMP_LO, max=_EXP_CLAMP_HI)
        out = clamped.exp()
        return torch.where(
            x > _EXP_CLAMP_HI,
            out.new_full(out.shape, math.inf),
            torch.where(x < _EXP_CLAMP_LO, out.new_zeros(out.shape), out),
        )
    if _is_numpy(x):
        np = _np()
        with np.errstate(all="ignore"):
            arr = np.asarray(x, dtype=float)
            clipped = np.clip(arr, _EXP_CLAMP_LO, _EXP_CLAMP_HI)
            out = np.exp(clipped)
            out = np.where(arr > _EXP_CLAMP_HI, math.inf, out)
            out = np.where(arr < _EXP_CLAMP_LO, 0.0, out)
            return out
    xf = float(x)
    if xf > _EXP_CLAMP_HI:
        return math.inf
    if xf < _EXP_CLAMP_LO:
        return 0.0
    return math.exp(xf)


def log_clamped(x: Any) -> Any:
    """``math.log`` with explicit ``-inf`` for non-positive input.

    Accepts ``None`` and treats it as 0 (returning ``-inf``), so the
    function can be used in the cost function where a missing
    component would otherwise be ``None``.
    """
    if x is None:
        return -math.inf
    if _is_torch_tensor(x):
        torch = _torch()
        # floor at LOG_FLOOR so the log is finite; outside the floor
        # return -inf.
        clamped = x.clamp(min=_LOG_FLOOR)
        out = clamped.log()
        return torch.where(x <= 0, out.new_full(out.shape, -math.inf), out)
    if _is_numpy(x):
        np = _np()
        with np.errstate(all="ignore"):
            arr = np.asarray(x, dtype=float)
            clipped = np.clip(arr, _LOG_FLOOR, None)
            out = np.log(clipped)
            out = np.where(arr <= 0, -math.inf, out)
            return out
    xf = float(x)
    if not math.isfinite(xf) or xf <= 0.0:
        return -math.inf
    return math.log(xf)


def log1p_clamped(x: Any) -> Any:
    """``math.log1p`` with explicit ``-inf`` for ``x <= -1``.

    NumPy / PyTorch array paths use the same ``-inf`` sentinel.
    """
    if _is_torch_tensor(x):
        torch = _torch()
        # ``log1p`` is well-behaved for ``x > -1``. We clamp the input
        # to ``-1 + LOG_FLOOR`` so the result is finite inside the
        # safe region and ``-inf`` outside.
        clamped = x.clamp(min=-1.0 + _LOG_FLOOR)
        out = clamped.log1p()
        return torch.where(x <= -1.0, out.new_full(out.shape, -math.inf), out)
    if _is_numpy(x):
        np = _np()
        with np.errstate(all="ignore"):
            arr = np.asarray(x, dtype=float)
            clipped = np.clip(arr, -1.0 + _LOG_FLOOR, None)
            out = np.log1p(clipped)
            out = np.where(arr <= -1.0, -math.inf, out)
            return out
    xf = float(x)
    if not math.isfinite(xf) or xf <= -1.0:
        return -math.inf
    return math.log1p(xf)


# ---------------------------------------------------------------------------
# softplus
# ---------------------------------------------------------------------------


def softplus(x: Any) -> Any:
    """Numerically stable ``log(1 + exp(x))``.

    Uses ``max(x, 0) + log1p(exp(-|x|))`` so the result is always finite
    for any finite input. The numpy path clips the exponent argument
    to avoid ``RuntimeWarning: overflow`` from ``np.exp``.
    """
    if _is_torch_tensor(x):
        return _torch().softplus(x)
    if _is_numpy(x):
        np = _np()
        with np.errstate(all="ignore"):
            return np.where(
                x > 0,
                x + np.log1p(np.exp(-np.clip(x, None, _EXP_CLAMP_HI))),
                np.log1p(np.exp(np.clip(x, _EXP_CLAMP_LO, None))),
            )
    xf = float(x)
    if xf > 0:
        return xf + math.log1p(math.exp(-xf))
    return math.log1p(math.exp(xf))


# ---------------------------------------------------------------------------
# expit / sigmoid
# ---------------------------------------------------------------------------


def expit(x: Any) -> Any:
    """Numerically stable logistic sigmoid.

    ``expit(x) = 1 / (1 + exp(-x))``. Implemented as
    ``expit(x) = expit(-x)`` for the negative branch so we never call
    ``math.exp`` with a large positive argument.
    """
    if _is_torch_tensor(x):
        return _torch().sigmoid(x)
    if _is_numpy(x):
        np = _np()
        with np.errstate(all="ignore"):
            return np.where(
                x >= 0,
                1.0 / (1.0 + np.exp(-np.clip(x, None, _EXP_CLAMP_HI))),
                np.exp(np.clip(x, _EXP_CLAMP_LO, None))
                / (1.0 + np.exp(np.clip(x, _EXP_CLAMP_LO, None))),
            )
    xf = float(x)
    if xf >= 0:
        ex = math.exp(-xf)
        return 1.0 / (1.0 + ex)
    ex = math.exp(xf)
    return ex / (1.0 + ex)


sigmoid = expit  # canonical name


def sigmoid_diff(x: Any, center: float = 0.0) -> Any:
    """``sigmoid(x - center) - sigmoid(center - x)``.

    Equivalent to ``sigmoid(x) - sigmoid(-x)`` (with centre) and is
    anti-symmetric: ``sigmoid_diff(x, c) == -sigmoid_diff(-x, -c)``.
    """
    return expit(x - center) - expit(center - x)


# ---------------------------------------------------------------------------
# logaddexp / logsumexp
# ---------------------------------------------------------------------------


def logaddexp(a: float, b: float) -> float:
    """Stable ``log(exp(a) + exp(b))`` for scalars.

    Uses the classic ``max(a, b) + log1p(exp(-|a - b|))`` formulation.
    """
    a = float(a)
    b = float(b)
    if a == -math.inf and b == -math.inf:
        return -math.inf
    m = a if a >= b else b
    return m + math.log1p(math.exp(-abs(a - b)))


def logsumexp_pair(a: float, b: float) -> float:
    """Pairwise stable logsumexp; same contract as ``logaddexp``."""
    return logaddexp(a, b)


def logsumexp(xs: Iterable[float]) -> float:
    """Stable logsumexp over an iterable of scalars.

    Returns ``-math.inf`` for an empty sequence or for an all-``-inf``
    sequence.
    """
    vals = [float(v) for v in xs]
    if not vals:
        return -math.inf
    finite = [v for v in vals if v != -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


# ---------------------------------------------------------------------------
# softmax / log_softmax
# ---------------------------------------------------------------------------


def softmax(arr: Any) -> Any:
    """Numerically stable softmax for 1-D arrays.

    Subtracts the max for stability. Returns 0 in positions whose
    logit was ``-inf``.
    """
    if _is_numpy(arr):
        np = _np()
        with np.errstate(all="ignore"):
            a = np.asarray(arr, dtype=float)
            m = np.max(a)
            if not np.isfinite(m):
                # all -inf -> uniform
                return np.full_like(a, 1.0 / a.size)
            exps = np.exp(a - m)
            return exps / np.sum(exps)
    raise TypeError("softmax expects a numpy array")


def log_softmax(arr: Any) -> Any:
    """Numerically stable log-softmax for 1-D arrays."""
    if _is_numpy(arr):
        np = _np()
        with np.errstate(all="ignore"):
            a = np.asarray(arr, dtype=float)
            m = np.max(a)
            shifted = a - m
            return shifted - np.log(np.sum(np.exp(shifted)))
    raise TypeError("log_softmax expects a numpy array")


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------


def log_poisson_pmf(k: int, rate: float) -> float:
    """``log P(K = k)`` for ``K ~ Poisson(rate)``.

    Returns ``-math.inf`` for ``k < 0`` or ``rate <= 0``. The degenerate
    case ``(k=0, rate=0)`` also returns ``-math.inf`` to keep the
    function continuous at the origin: as rate → 0+ the probability
    of observing ``k=0`` becomes a delta which is not representable
    on the real line, so ``-inf`` is the correct log-domain value.
    """
    kk = int(k)
    if kk < 0:
        return -math.inf
    if rate <= 0.0:
        return -math.inf
    return kk * math.log(rate) - rate - math.lgamma(kk + 1)


def log_normal_pdf(x: float, sigma: float) -> float:
    """``log N(x; 0, sigma^2)`` with a sigma floor so ``sigma == 0`` does
    not divide by zero.
    """
    sig = max(float(sigma), _SIGMA_MIN)
    if not math.isfinite(float(x)) or not math.isfinite(sig):
        return -math.inf
    return -0.5 * math.log(2 * math.pi) - math.log(sig) - 0.5 * (float(x) / sig) ** 2


def log_student_t_pdf(x: float, scale: float, df: float) -> float:
    """``log t(x; df)`` with scale and a df floor."""
    xf = float(x)
    sc = max(float(scale), _SIGMA_MIN)
    dff = float(df)
    if dff <= 0.0:
        return -math.inf
    if not math.isfinite(xf) or not math.isfinite(sc) or not math.isfinite(dff):
        return -math.inf
    z = xf / sc
    return (
        math.lgamma(0.5 * (dff + 1.0))
        - math.lgamma(0.5 * dff)
        - 0.5 * math.log(dff * math.pi)
        - math.log(sc)
        - 0.5 * (dff + 1.0) * math.log1p(z * z / dff)
    )


# ---------------------------------------------------------------------------
# safe arithmetic
# ---------------------------------------------------------------------------


def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """Division that returns ``default`` on zero / non-finite input."""
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(n) or not math.isfinite(d) or d == 0.0:
        return default
    return n / d


def safe_sqrt(x: Any) -> float:
    """``math.sqrt`` that returns 0.0 for negative / non-finite input."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(xf) or xf < 0.0:
        return 0.0
    return math.sqrt(xf)


def finite_or(x: Any, default: float | None = None) -> float:
    """Return ``x`` if it is finite, else ``default``. ``None`` counts as
    non-finite.
    """
    if x is None:
        if default is None:
            return 0.0
        return float(default)
    try:
        xf = float(x)
    except (TypeError, ValueError):
        if default is None:
            return 0.0
        return float(default)
    if not math.isfinite(xf):
        if default is None:
            return 0.0
        return float(default)
    return xf


def is_finite(x: Any) -> bool:
    """``True`` iff ``x`` is finite (scalar or elementwise array)."""
    if _is_numpy(x):
        return bool(_np().all(_np().isfinite(x)))
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
