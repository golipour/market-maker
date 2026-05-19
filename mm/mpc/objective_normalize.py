"""Dimensionless scaling helpers for MPC terminal objective components."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


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


def relative_absolute_error_tensor(actual: "Tensor", target: float, *, eps: float = 1e-8) -> "Tensor":
    """
    Dimensionless symmetric deviation: ``|actual - target| / |target|``.

    A weight of 1.0 on this term means roughly one unit of 100% relative error.
    Unlike squared error, 50% miss and 100% miss scale linearly (0.5 vs 1.0).
    """

    import torch

    denom = max(float(eps), abs(float(target)))
    t = torch.tensor(denom, dtype=actual.dtype, device=actual.device)
    return torch.abs(actual - t) / t


def relative_spread_penalty_excess_tensor(
    spread: "Tensor",
    target_spread: float,
    *,
    eps: float = 1e-8,
) -> "Tensor":
    """
    One-sided relative spread penalty: only spreads wider than target.

    Returns ``max(0, spread - s_tgt) / s_tgt`` (linear, zero when spread <= target).
    """

    import torch

    denom = max(float(eps), abs(float(target_spread)))
    tgt = torch.tensor(denom, dtype=spread.dtype, device=spread.device)
    excess = torch.clamp(spread - tgt, min=0.0)
    return excess / tgt


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
