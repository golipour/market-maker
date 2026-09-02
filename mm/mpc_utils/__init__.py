"""Centralised numerically-stable arithmetic primitives for the MPC pipeline.

This module is the single source of truth for exp/log-family operations
that previously lived as inline ``try/except OverflowError`` blocks in
multiple files. Centralising them gives us:

* A consistent behaviour across scalar, NumPy array, and PyTorch tensor
  inputs (no surprises when a value crosses into ``exp`` overflow
  territory in only one of those code paths).
* A single set of clamps and sentinels (``math.inf`` / ``0.0`` /
  ``-math.inf``) so that callers can reason about a returned value
  without needing to remember the original numpy error semantics.
* A regression-proof surface area: any future change to the cost model
  in the optimizer or the dual-state update only needs to be audited
  in one place.

The module is intentionally tiny: it contains no business logic, only
the well-behaved arithmetic primitives. Callers are responsible for
combining them into the objective functions, dual-state updates, and
probability scores that live in the rest of the MPC package.
"""

from mm.mpc_utils.finite import (
    exp_clamped,
    expit,
    finite_or,
    is_finite,
    log1p_clamped,
    log_clamped,
    log_normal_pdf,
    log_poisson_pmf,
    log_student_t_pdf,
    log_softmax,
    logaddexp,
    logsumexp,
    logsumexp_pair,
    safe_divide,
    safe_sqrt,
    sigmoid,
    sigmoid_diff,
    softplus,
    softmax,
)

__all__ = [
    "exp_clamped",
    "log_clamped",
    "log1p_clamped",
    "logaddexp",
    "logsumexp_pair",
    "logsumexp",
    "expit",
    "sigmoid",
    "sigmoid_diff",
    "softplus",
    "softmax",
    "log_softmax",
    "log_poisson_pmf",
    "log_normal_pdf",
    "log_student_t_pdf",
    "safe_divide",
    "safe_sqrt",
    "finite_or",
    "is_finite",
]
