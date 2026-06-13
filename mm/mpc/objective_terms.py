"""Normalized soft objective terms for MPC (book quality and inventory zone)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mm.mpc.l_param import LParamConfig, materialize_side_torch
from mm.mpc.state_tensor import (
    depth_bid_derivative_tensor,
    depth_ask_derivative_tensor,
    effective_depth_ask_tensor,
    effective_depth_bid_tensor,
)


def _torch():
    import torch

    return torch


@dataclass(frozen=True, slots=True)
class InventoryZoneParams:
    q_lo: float = -0.15
    q_hi: float = 0.15
    q_min: float = -0.45
    q_max: float = 0.45
    p: float = 2.0

    @classmethod
    def from_mapping(cls, payload: Any) -> InventoryZoneParams | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("inventory_zone must be an object")
        return cls(
            q_lo=float(payload.get("q_lo", -0.15)),
            q_hi=float(payload.get("q_hi", 0.15)),
            q_min=float(payload.get("q_min", -0.45)),
            q_max=float(payload.get("q_max", 0.45)),
            p=max(2.0, float(payload.get("p", 2.0))),
        )


@dataclass(frozen=True, slots=True)
class BookQualityParams:
    spread_target: float = 50.0
    depth_target: float = 1.0
    touch_spread_fraction: float = 0.01
    measure: str = "quoted"
    v_presence: float = 0.25
    v_tightness: float = 0.25
    v_balance: float = 0.25
    v_uniformity: float = 0.25

    @classmethod
    def from_mapping(cls, payload: Any) -> BookQualityParams | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("book_quality must be an object")
        measure = str(payload.get("measure", "quoted")).strip().lower()
        if measure not in ("quoted", "combined"):
            raise ValueError("book_quality.measure must be 'quoted' or 'combined'")
        return cls(
            spread_target=max(1e-9, float(payload.get("spread_target", 50.0))),
            depth_target=max(1e-9, float(payload.get("depth_target", 1.0))),
            touch_spread_fraction=max(
                1e-9,
                float(payload.get("touch_spread_fraction", 0.01)),
            ),
            measure=measure,
            v_presence=float(payload.get("v_presence", 0.25)),
            v_tightness=float(payload.get("v_tightness", 0.25)),
            v_balance=float(payload.get("v_balance", 0.25)),
            v_uniformity=float(payload.get("v_uniformity", 0.25)),
        )


def inventory_value_fraction(
    inventory: Any,
    *,
    mid_price: Any,
    wealth: float,
) -> Any:
    wealth_safe = max(1e-12, float(wealth))
    return inventory * mid_price / wealth_safe


def inventory_zone_penalty(
    q: Any,
    zone: InventoryZoneParams,
) -> Any:
    """Smooth zone-of-indifference penalty in [0, ~1], C¹ at band edges when p >= 2."""

    torch = _torch()
    q_lo = float(zone.q_lo)
    q_hi = float(zone.q_hi)
    q_min = float(zone.q_min)
    q_max = float(zone.q_max)
    p = float(zone.p)
    eps = 1e-12

    penalty = torch.zeros_like(q)
    if q_lo > q_min + eps:
        low_mask = q < q_lo
        low_excess = (q_lo - q) / max(eps, q_lo - q_min)
        penalty = penalty + torch.where(
            low_mask,
            low_excess**p,
            torch.zeros_like(q),
        )
    if q_max > q_hi + eps:
        high_mask = q > q_hi
        high_excess = (q - q_hi) / max(eps, q_max - q_hi)
        penalty = penalty + torch.where(
            high_mask,
            high_excess**p,
            torch.zeros_like(q),
        )
    return penalty


def inventory_zone_path_mean(
    inventory_path: Any,
    *,
    mid_path: Any,
    wealth: float,
    zone: InventoryZoneParams | None,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    if zone is None:
        zero = torch.zeros((), dtype=inventory_path.dtype, device=inventory_path.device)
        return zero, {}
    q = inventory_value_fraction(inventory_path, mid_price=mid_path, wealth=wealth)
    step_penalty = inventory_zone_penalty(q, zone)
    path_mean = step_penalty.mean()
    return path_mean, {"inventory_zone_penalty": path_mean}


def _market_level_from_cumulative(cum: Any) -> Any:
    torch = _torch()
    prev = torch.cat([torch.zeros(1, dtype=cum.dtype, device=cum.device), cum[:-1]])
    return torch.clamp(cum - prev, min=0.0)


def _cum_depth_at_spread(cum: Any, spread_grid: Any, spread: float) -> Any:
    torch = _torch()
    spread_val = max(0.0, float(spread))
    idx = torch.searchsorted(spread_grid, torch.tensor(spread_val, dtype=spread_grid.dtype, device=spread_grid.device))
    idx = int(idx.item())
    if idx <= 0:
        if spread_grid.numel() == 0:
            return torch.zeros((), dtype=cum.dtype, device=cum.device)
        edge = torch.clamp(spread_grid[0], min=1e-12)
        return cum[0] * (spread_val / edge)
    idx = min(idx, int(cum.numel()) - 1)
    return cum[idx]


def _uniformity_from_marginals(marginals: Any) -> Any:
    torch = _torch()
    mean = marginals.mean()
    var = ((marginals - mean) ** 2).mean()
    return 1.0 - var / (mean * mean + var + 1e-9)


def book_quality_score(
    action_flat: Any,
    *,
    l_param: LParamConfig,
    n_components: int,
    spread_grid: Any,
    bid_quote_size_min: Any | None,
    bid_quote_size_max: Any | None,
    ask_quote_size_min: Any | None,
    ask_quote_size_max: Any | None,
    mid_price: float,
    params: BookQualityParams,
    market_cum_bid: Any | None = None,
    market_cum_ask: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Scalar book-quality score in [0, 1] from the h=0 decoded action."""

    torch = _torch()
    use_combined = str(params.measure).strip().lower() == "combined"
    if use_combined and (market_cum_bid is None or market_cum_ask is None):
        raise ValueError("book_quality.measure='combined' requires market cumulative depth tensors")

    bid_L, bid_k, bid_x0 = materialize_side_torch(
        action_flat,
        side_off=0,
        n_components=n_components,
        l_param=l_param,
        quote_size_min=bid_quote_size_min,
        quote_size_max=bid_quote_size_max,
    )
    ask_L, ask_k, ask_x0 = materialize_side_torch(
        action_flat,
        side_off=3 * n_components,
        n_components=n_components,
        l_param=l_param,
        quote_size_min=ask_quote_size_min,
        quote_size_max=ask_quote_size_max,
    )
    spread_max = float(l_param.spread_max)
    touch = max(1e-9, spread_max * float(params.touch_spread_fraction))

    depth_bid = effective_depth_bid_tensor(
        action_flat,
        spread_grid,
        n_components,
        l_param=l_param,
        quote_size_min=bid_quote_size_min,
        quote_size_max=bid_quote_size_max,
    )
    depth_ask = effective_depth_ask_tensor(
        action_flat,
        spread_grid,
        n_components,
        l_param=l_param,
        quote_size_min=ask_quote_size_min,
        quote_size_max=ask_quote_size_max,
    )
    q_bid = depth_bid[-1]
    q_ask = depth_ask[-1]

    depth_touch_bid = effective_depth_bid_tensor(
        action_flat,
        torch.tensor([touch], dtype=spread_grid.dtype, device=spread_grid.device),
        n_components,
        l_param=l_param,
        quote_size_min=bid_quote_size_min,
        quote_size_max=bid_quote_size_max,
    )[0]
    depth_touch_ask = effective_depth_ask_tensor(
        action_flat,
        torch.tensor([touch], dtype=spread_grid.dtype, device=spread_grid.device),
        n_components,
        l_param=l_param,
        quote_size_min=ask_quote_size_min,
        quote_size_max=ask_quote_size_max,
    )[0]

    if use_combined:
        market_touch_bid = _cum_depth_at_spread(market_cum_bid, spread_grid, touch)
        market_touch_ask = _cum_depth_at_spread(market_cum_ask, spread_grid, touch)
        market_q_bid = market_cum_bid[-1]
        market_q_ask = market_cum_ask[-1]
        d_bid_touch = depth_touch_bid + market_touch_bid
        d_ask_touch = depth_touch_ask + market_touch_ask
        q_bid = q_bid + market_q_bid
        q_ask = q_ask + market_q_ask
        bid_marg = depth_bid_derivative_tensor(
            action_flat,
            spread_grid,
            n_components,
            l_param=l_param,
        ) + _market_level_from_cumulative(market_cum_bid)
        ask_marg = depth_ask_derivative_tensor(
            action_flat,
            spread_grid,
            n_components,
            l_param=l_param,
        ) + _market_level_from_cumulative(market_cum_ask)
    else:
        d_bid_touch = depth_touch_bid
        d_ask_touch = depth_touch_ask
        bid_marg = depth_bid_derivative_tensor(
            action_flat,
            spread_grid,
            n_components,
            l_param=l_param,
        )
        ask_marg = depth_ask_derivative_tensor(
            action_flat,
            spread_grid,
            n_components,
            l_param=l_param,
        )

    smooth_min = torch.minimum(d_bid_touch, d_ask_touch)
    q_ref = max(1e-9, float(params.depth_target))
    presence = smooth_min / (q_ref + smooth_min)

    spread_sum = bid_x0[0] + ask_x0[0]
    tightness = torch.exp(-spread_sum / max(1e-9, float(params.spread_target)))

    q_bid_notional = q_bid * max(0.0, float(mid_price))
    q_ask_notional = q_ask * max(0.0, float(mid_price))
    denom = q_bid_notional + q_ask_notional + 1e-9
    balance = 1.0 - torch.abs(q_bid_notional - q_ask_notional) / denom

    uniformity = 0.5 * (
        _uniformity_from_marginals(bid_marg) + _uniformity_from_marginals(ask_marg)
    )

    w_p = float(params.v_presence)
    w_t = float(params.v_tightness)
    w_b = float(params.v_balance)
    w_u = float(params.v_uniformity)
    w_sum = max(1e-9, w_p + w_t + w_b + w_u)
    score = (w_p * presence + w_t * tightness + w_b * balance + w_u * uniformity) / w_sum
    score = torch.clamp(score, 0.0, 1.0)
    components = {
        "book_quality": score,
        "book_presence": presence,
        "book_tightness": tightness,
        "book_balance": balance,
        "book_uniformity": uniformity,
    }
    return score, components


OBJECTIVE_CONTRIBUTION_KEYS: frozenset[str] = frozenset(
    {
        "risk_adjusted_return",
        "book_quality",
        "inventory_zone_penalty",
        "global_book_quality",
        "global_inventory_zone_penalty",
    }
)


def combine_soft_objective_terms(
    base_objective: Any,
    components: dict[str, Any],
    *,
    w_book: float,
    w_inv: float,
    book_score: Any | None = None,
    inventory_zone_mean: Any | None = None,
    book_components: dict[str, Any] | None = None,
    zone_components: dict[str, Any] | None = None,
    book_term: str = "book_quality",
    zone_term: str = "inventory_zone_penalty",
    component_prefix: str = "",
) -> tuple[Any, dict[str, Any]]:
    """Combine CE return with optional book-quality reward and inventory-zone penalty."""

    out = dict(components)
    objective = base_objective
    if book_score is not None and float(w_book) > 0.0:
        weighted_book = float(w_book) * book_score
        objective = objective + weighted_book
        out[book_term] = weighted_book
        for key, value in (book_components or {}).items():
            if key == "book_quality":
                out[f"{book_term}_score"] = value
            else:
                diag_key = key if key.startswith("book_") else f"book_{key}"
                out[f"{component_prefix}{diag_key}"] = value
    if inventory_zone_mean is not None and float(w_inv) > 0.0:
        weighted_zone = -float(w_inv) * inventory_zone_mean
        objective = objective + weighted_zone
        out[zone_term] = weighted_zone
        for key, value in (zone_components or {}).items():
            if key == "inventory_zone_penalty":
                out[f"{zone_term}_raw"] = value
            else:
                out[f"{component_prefix}{key}"] = value
    return objective, out
