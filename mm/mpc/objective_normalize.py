"""Dimensionless scaling helpers for MPC terminal objective components."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch import Tensor

PenaltyShape = Literal["softplus", "squared"]

# Default sharpness for softplus penalties: ~linear for |rel| < ~0.1, softer tail than x^2.
DEFAULT_PENALTY_SHARPNESS = 8.0


def wealth_scale(
    cash: float,
    inventory: float,
    mark_price: float,
    *,
    eps: float = 1e-8,
) -> float:
    """Reference wealth W = |cash| + |inventory * mark| for relative PnL/risk terms."""

    return max(float(eps), abs(float(cash)) + abs(float(inventory) * float(mark_price)))


def inventory_scale(
    target_inventory: float,
    quote_size_target: float | None = None,
    *,
    eps: float = 1e-8,
) -> float:
    """Scale for inventory target penalty: max(|target|, quote size, 1)."""

    parts = [abs(float(target_inventory)), 1.0]
    if quote_size_target is not None and float(quote_size_target) > 0.0:
        parts.append(float(quote_size_target))
    return max(float(eps), max(parts))


def spread_target_scale(quote_spread_target: float | None, spread_max: float, *, eps: float = 1e-8) -> float:
    """Positive spread reference for relative spread penalty."""

    if quote_spread_target is not None and float(quote_spread_target) > 0.0:
        return float(quote_spread_target)
    return max(float(eps), 0.5 * max(0.0, float(spread_max)))


def softplus_unit_penalty_scalar(
    rel_err_nonneg: float,
    *,
    sharpness: float = DEFAULT_PENALTY_SHARPNESS,
) -> float:
    """
    Map non-negative relative error to a dimensionless penalty.

    Uses ``(softplus(k*x) - log(2)) / k`` so penalty(0)=0 and slope at 0 is ~0.5
    (more comparable to linear PnL terms than x^2, which has zero slope at 0).
    """

    x = max(0.0, float(rel_err_nonneg))
    if x <= 0.0:
        return 0.0
    k = max(1e-6, float(sharpness))
    z = k * x
    sp = math.log1p(math.exp(-abs(z))) + max(z, 0.0)
    return max(0.0, (sp - math.log(2.0)) / k)


def softplus_unit_penalty_tensor(
    rel_err_nonneg: "Tensor",
    *,
    sharpness: float = DEFAULT_PENALTY_SHARPNESS,
) -> "Tensor":
    import torch

    x = torch.clamp(rel_err_nonneg, min=0.0)
    k = max(1e-6, float(sharpness))
    sp = torch.nn.functional.softplus(k * x)
    return torch.clamp((sp - math.log(2.0)) / k, min=0.0)


def squared_unit_penalty_tensor(rel_err_nonneg: "Tensor") -> "Tensor":
    import torch

    x = torch.clamp(rel_err_nonneg, min=0.0)
    return x * x


def dimensionless_excess_penalty_tensor(
    value: "Tensor",
    target: float,
    *,
    shape: PenaltyShape = "softplus",
    sharpness: float = DEFAULT_PENALTY_SHARPNESS,
    eps: float = 1e-8,
) -> "Tensor":
    """One-sided: penalize only when ``value > target`` (e.g. spread wider than target)."""

    import torch

    denom = max(float(eps), abs(float(target)))
    tgt = torch.tensor(denom, dtype=value.dtype, device=value.device)
    rel_excess = torch.clamp((value - tgt) / tgt, min=0.0)
    if shape == "squared":
        return squared_unit_penalty_tensor(rel_excess)
    return softplus_unit_penalty_tensor(rel_excess, sharpness=sharpness)


def dimensionless_symmetric_penalty_tensor(
    value: "Tensor",
    target: float,
    *,
    shape: PenaltyShape = "softplus",
    sharpness: float = DEFAULT_PENALTY_SHARPNESS,
    eps: float = 1e-8,
) -> "Tensor":
    """Symmetric relative error around target (e.g. quote size miss)."""

    import torch

    denom = max(float(eps), abs(float(target)))
    tgt = torch.tensor(denom, dtype=value.dtype, device=value.device)
    rel = (value - tgt) / tgt
    rel_abs = torch.abs(rel)
    if shape == "squared":
        return squared_unit_penalty_tensor(rel_abs)
    return softplus_unit_penalty_tensor(rel_abs, sharpness=sharpness)


def relative_squared_error_scalar(actual: float, target: float, *, eps: float = 1e-8) -> float:
    denom = max(float(eps), abs(float(target)))
    rel = abs((float(actual) - float(target)) / denom)
    return float(rel * rel)


def relative_squared_error_tensor(actual: "Tensor", target: float, *, eps: float = 1e-8) -> "Tensor":
    return dimensionless_symmetric_penalty_tensor(actual, target, shape="squared", eps=eps)


def relative_spread_penalty_excess_tensor(
    spread: "Tensor",
    target_spread: float,
    *,
    shape: PenaltyShape = "softplus",
    sharpness: float = DEFAULT_PENALTY_SHARPNESS,
    eps: float = 1e-8,
) -> "Tensor":
    """One-sided relative spread penalty (wider than target only)."""

    return dimensionless_excess_penalty_tensor(
        spread,
        target_spread,
        shape=shape,
        sharpness=sharpness,
        eps=eps,
    )


def normalize_herfindahl(h: "Tensor", n_levels: int, *, eps: float = 1e-8) -> "Tensor":
    """
    Map Herfindahl index H in [1/n, 1] to [0, 1]: 0 = uniform depth, 1 = single bucket.
    """

    import torch

    n = max(1, int(n_levels))
    if n <= 1:
        return torch.zeros_like(h)
    lo = 1.0 / float(n)
    return torch.clamp((h - lo) / max(float(eps), 1.0 - lo), min=0.0, max=1.0)
