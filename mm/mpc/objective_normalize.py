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


def resolve_quote_spread_target_bps(
    quote_spread_target_bps: float | None,
    *,
    default_bps: float = 50.0,
    eps: float = 1e-8,
) -> float:
    """Target quoted width in basis points (ask depth VWAP − bid depth VWAP) / mid × 10⁴."""

    if quote_spread_target_bps is not None and float(quote_spread_target_bps) > 0.0:
        return float(quote_spread_target_bps)
    return max(float(eps), float(default_bps))


def relative_spread_penalty_excess_bps(
    spread_bps: "Tensor",
    target_bps: float,
    *,
    eps: float = 1e-8,
) -> "Tensor":
    """One-sided linear penalty: ``max(0, spread_bps - target_bps) / target_bps``."""

    import torch

    tgt = max(float(eps), float(target_bps))
    t = torch.tensor(tgt, dtype=spread_bps.dtype, device=spread_bps.device)
    return torch.clamp(spread_bps - t, min=0.0) / t


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
