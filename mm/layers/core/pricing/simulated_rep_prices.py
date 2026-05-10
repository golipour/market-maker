from __future__ import annotations

"""
Simulated (tensor-batched) representative price computations used in MPC rollouts.

This logic is shared across any horizon simulation that evolves an orderbook depth tensor.
"""

import torch


def representative_prices_batched(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    center_mid: torch.Tensor,
    spread_grid: torch.Tensor,
    *,
    target_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    VWAP representative buy/sell prices to shared depth from level volumes.

    - `ob_bids`, `ob_asks`: (n_samples, n_levels) nonnegative volumes per spread level.
    - `center_mid`: (n_samples, 1) mid reference for absolute price reconstruction.
    - `spread_grid`: (n_levels,) spread ticks for each level (typically 1..n_levels).
    - `target_depth`: max depth (in lots) to VWAP into (shared across bid+ask).

    Returns (buy_rep, sell_rep), each (n_samples, 1).
    """
    eps = 1e-12
    cum_bid = torch.cumsum(ob_bids, dim=1)
    cum_ask = torch.cumsum(ob_asks, dim=1)
    total_bid = cum_bid[:, -1:]
    total_ask = cum_ask[:, -1:]
    depth_used = torch.minimum(
        torch.minimum(total_bid, total_ask),
        torch.full_like(total_bid, float(target_depth)),
    )
    depth_used_safe = torch.clamp(depth_used, min=eps)

    cum_bid_prev = torch.cat([torch.zeros_like(cum_bid[:, :1]), cum_bid[:, :-1]], dim=1)
    cum_ask_prev = torch.cat([torch.zeros_like(cum_ask[:, :1]), cum_ask[:, :-1]], dim=1)
    take_bid = torch.clamp(depth_used - cum_bid_prev, min=0.0)
    take_bid = torch.minimum(take_bid, ob_bids)
    take_ask = torch.clamp(depth_used - cum_ask_prev, min=0.0)
    take_ask = torch.minimum(take_ask, ob_asks)

    bid_prices = center_mid - spread_grid.unsqueeze(0)
    ask_prices = center_mid + spread_grid.unsqueeze(0)
    sell_rep = (take_bid * bid_prices).sum(dim=1, keepdim=True) / depth_used_safe
    buy_rep = (take_ask * ask_prices).sum(dim=1, keepdim=True) / depth_used_safe

    no_depth = depth_used <= eps
    sell_rep = torch.where(no_depth, center_mid, sell_rep)
    buy_rep = torch.where(no_depth, center_mid, buy_rep)
    return buy_rep, sell_rep

