from __future__ import annotations

"""Batched VWAP on a fixed spread grid (Torch), used inside MPC rollouts.

``vwap_prices_batched`` assumes bid/ask **quantities are already aligned** on
``spread_grid`` relative to ``center_mid``. For **native exchange** price levels
use ``mm.pricing.orderbook_ref.vwap_from_raw_orderbook`` — same depth policy
(``min(cap, Σbid, Σask)``) but a different input layout; do not call this batched
kernel from the scalar path without projecting the book onto the grid first.
"""

import torch


def vwap_prices_batched(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    center_mid: torch.Tensor,
    spread_grid: torch.Tensor,
    *,
    target_depth: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched buy-VWAP (ask side) and sell-VWAP (bid side) to a shared cumulative depth.

    - ``ob_bids``, ``ob_asks``: (n_samples, n_levels) nonnegative base qty per spread level.
    - ``center_mid``: (n_samples, 1) mid for absolute price = mid ± spread_grid.
    - ``spread_grid``: (n_levels,) positive spreads from mid.
    - ``target_depth``: cap on cumulative base qty per side (shared across bid+ask).
      May be a scalar or a per-sample tensor shaped ``(n_samples,)`` or
      ``(n_samples, 1)``.

    Returns ``(buy_vwap, sell_vwap)``, each (n_samples, 1).
    """

    eps = 1e-12
    cum_bid = torch.cumsum(ob_bids, dim=1)
    cum_ask = torch.cumsum(ob_asks, dim=1)
    total_bid = cum_bid[:, -1:]
    total_ask = cum_ask[:, -1:]
    if torch.is_tensor(target_depth):
        target = target_depth.to(device=total_bid.device, dtype=total_bid.dtype)
        if target.ndim == 1:
            target = target.view(-1, 1)
        target = torch.clamp(target, min=0.0)
    else:
        target = torch.full_like(total_bid, max(0.0, float(target_depth)))
    depth_used = torch.minimum(torch.minimum(total_bid, total_ask), target)
    depth_used_safe = torch.clamp(depth_used, min=eps)

    cum_bid_prev = torch.cat([torch.zeros_like(cum_bid[:, :1]), cum_bid[:, :-1]], dim=1)
    cum_ask_prev = torch.cat([torch.zeros_like(cum_ask[:, :1]), cum_ask[:, :-1]], dim=1)
    take_bid = torch.clamp(depth_used - cum_bid_prev, min=0.0)
    take_bid = torch.minimum(take_bid, ob_bids)
    take_ask = torch.clamp(depth_used - cum_ask_prev, min=0.0)
    take_ask = torch.minimum(take_ask, ob_asks)

    bid_prices = center_mid - spread_grid.unsqueeze(0)
    ask_prices = center_mid + spread_grid.unsqueeze(0)
    sell_vwap = (take_bid * bid_prices).sum(dim=1, keepdim=True) / depth_used_safe
    buy_vwap = (take_ask * ask_prices).sum(dim=1, keepdim=True) / depth_used_safe

    no_depth = depth_used <= eps
    sell_vwap = torch.where(no_depth, center_mid, sell_vwap)
    buy_vwap = torch.where(no_depth, center_mid, buy_vwap)
    return buy_vwap, sell_vwap
