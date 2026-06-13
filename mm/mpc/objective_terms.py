"""Normalized soft objective terms for MPC (book quality and inventory zone)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mm.mpc.l_param import (
    LParamConfig,
    component_delta_at_torch,
    component_delta_spread_range_torch,
    materialize_side_torch,
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
        return cls(
            spread_target=max(1e-9, float(payload.get("spread_target", 50.0))),
            depth_target=max(1e-9, float(payload.get("depth_target", 1.0))),
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


def book_quality_score(
    action_flat: Any,
    *,
    l_param: LParamConfig,
    n_components: int,
    bid_quote_size_min: Any | None,
    bid_quote_size_max: Any | None,
    ask_quote_size_min: Any | None,
    ask_quote_size_max: Any | None,
    mid_price: float,
    params: BookQualityParams,
) -> tuple[Any, dict[str, Any]]:
    """Scalar book-quality score in [0, 1] from the h=0 decoded action."""

    torch = _torch()
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
    delta_max_bid = component_delta_spread_range_torch(bid_k, bid_x0, spread_max)
    delta_max_ask = component_delta_spread_range_torch(ask_k, ask_x0, spread_max)
    q_bid = (bid_L * delta_max_bid).sum()
    q_ask = (ask_L * delta_max_ask).sum()

    touch = max(1e-9, float(l_param.spread_max) * 0.01)
    delta_touch_bid = component_delta_at_torch(bid_k, bid_x0, touch)
    delta_touch_ask = component_delta_at_torch(ask_k, ask_x0, touch)
    d_bid_touch = (bid_L * delta_touch_bid).sum()
    d_ask_touch = (ask_L * delta_touch_ask).sum()
    smooth_min = torch.minimum(d_bid_touch, d_ask_touch)
    q_ref = max(1e-9, float(params.depth_target))
    presence = smooth_min / (q_ref + smooth_min)

    spread_sum = bid_x0[0] + ask_x0[0]
    tightness = torch.exp(-spread_sum / max(1e-9, float(params.spread_target)))

    q_bid_notional = q_bid * max(0.0, float(mid_price))
    q_ask_notional = q_ask * max(0.0, float(mid_price))
    denom = q_bid_notional + q_ask_notional + 1e-9
    balance = 1.0 - torch.abs(q_bid_notional - q_ask_notional) / denom

    bid_mean = bid_L.mean()
    bid_var = ((bid_L - bid_mean) ** 2).mean()
    ask_mean = ask_L.mean()
    ask_var = ((ask_L - ask_mean) ** 2).mean()
    cv_bid = bid_var / (bid_mean * bid_mean + bid_var + 1e-9)
    cv_ask = ask_var / (ask_mean * ask_mean + ask_var + 1e-9)
    uniformity = 0.5 * (1.0 / (1.0 + cv_bid) + 1.0 / (1.0 + cv_ask))

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
) -> tuple[Any, dict[str, Any]]:
    """Combine CE return with optional book-quality reward and inventory-zone penalty."""

    out = dict(components)
    objective = base_objective
    if book_score is not None and float(w_book) > 0.0:
        objective = objective + float(w_book) * book_score
        out.update(book_components or {})
    if inventory_zone_mean is not None and float(w_inv) > 0.0:
        objective = objective - float(w_inv) * inventory_zone_mean
        out.update(zone_components or {})
    return objective, out
