"""Normalized soft objective terms for MPC (book quality and inventory zone)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mm.mpc.l_param import LParamConfig, materialize_side_torch, reservation_r_offset_from_params
from mm.mpc.state_tensor import (
    depth_bid_derivative_tensor,
    depth_ask_derivative_tensor,
    effective_depth_ask_tensor,
    effective_depth_bid_tensor,
)


def _torch():
    import torch

    return torch


_LEGACY_INVENTORY_ZONE_POWER_KEYS = frozenset({"q_min", "q_max", "p"})
_LEGACY_INVENTORY_ZONE_SIGMOID_ALIASES = frozenset(
    {"q_lo", "q_hi", "d_lo", "d_hi", "a_lo", "a_hi"}
)


@dataclass(frozen=True, slots=True)
class InventoryZoneParams:
    """Sigmoid operational zone on wallet imbalance (proposal §4.3 option B)."""

    comfort_low: float = -0.2
    comfort_high: float = 0.2
    low_ramp_width: float = 0.05
    high_ramp_width: float = 0.05
    low_penalty_cap: float = 1.0
    high_penalty_cap: float = 0.6

    @classmethod
    def from_mapping(cls, payload: Any) -> InventoryZoneParams | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("inventory_zone must be an object")
        removed_power = sorted(k for k in _LEGACY_INVENTORY_ZONE_POWER_KEYS if k in payload)
        if removed_power:
            raise ValueError(
                "inventory_zone power-zone keys were removed; use sigmoid zone "
                "{comfort_low, comfort_high, low_ramp_width, high_ramp_width, "
                "low_penalty_cap, high_penalty_cap}: "
                + ", ".join(removed_power)
            )
        removed_aliases = sorted(k for k in _LEGACY_INVENTORY_ZONE_SIGMOID_ALIASES if k in payload)
        if removed_aliases:
            raise ValueError(
                "inventory_zone keys renamed to descriptive sigmoid names "
                "(comfort_low, comfort_high, low_ramp_width, high_ramp_width, "
                "low_penalty_cap, high_penalty_cap); remove: "
                + ", ".join(removed_aliases)
            )
        comfort_low = float(payload.get("comfort_low", -0.2))
        comfort_high = float(payload.get("comfort_high", 0.2))
        if comfort_low >= comfort_high:
            raise ValueError("inventory_zone.comfort_low must be < comfort_high")
        return cls(
            comfort_low=comfort_low,
            comfort_high=comfort_high,
            low_ramp_width=max(1e-9, float(payload.get("low_ramp_width", 0.05))),
            high_ramp_width=max(1e-9, float(payload.get("high_ramp_width", 0.05))),
            low_penalty_cap=max(0.0, float(payload.get("low_penalty_cap", 1.0))),
            high_penalty_cap=max(0.0, float(payload.get("high_penalty_cap", 0.6))),
        )

    def as_mapping(self) -> dict[str, float]:
        return {
            "comfort_low": self.comfort_low,
            "comfort_high": self.comfort_high,
            "low_ramp_width": self.low_ramp_width,
            "high_ramp_width": self.high_ramp_width,
            "low_penalty_cap": self.low_penalty_cap,
            "high_penalty_cap": self.high_penalty_cap,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveNormalizeParams:
    """Optional scaling so CE, book, and inventory terms share comparable magnitude."""

    enabled: bool = False
    ce_ewma_decay: float = 0.95

    @classmethod
    def from_mapping(cls, payload: Any) -> ObjectiveNormalizeParams:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("objective_normalize must be an object")
        unknown = sorted(key for key in payload if key not in {"enabled", "ce_ewma_decay"})
        if unknown:
            raise ValueError(
                "unknown objective_normalize keys: "
                + ", ".join(unknown)
            )
        decay = float(payload.get("ce_ewma_decay", 0.95))
        if not (0.0 <= decay < 1.0):
            raise ValueError("objective_normalize.ce_ewma_decay must be in [0, 1)")
        return cls(
            enabled=bool(payload.get("enabled", False)),
            ce_ewma_decay=decay,
        )

    def as_mapping(self) -> dict[str, bool | float]:
        return {
            "enabled": self.enabled,
            "ce_ewma_decay": self.ce_ewma_decay,
        }


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


def wallet_imbalance(
    inventory: Any,
    *,
    mid_price: Any,
    cash: Any,
    eps: float = 1e-12,
) -> Any:
    """
    Normalized cash/base mix in [-1, 1].

    (Q·mid − cash) / (|Q·mid| + |cash| + eps)
    Negative ⇒ cash-heavy (low base, ask-side capacity risk).
    Positive ⇒ base-heavy (low cash, bid-side capacity risk).
    """

    inv_value = inventory * mid_price
    denom = inv_value.abs() + cash.abs() + float(eps)
    return (inv_value - cash) / denom


def inventory_zone_penalty(
    imbalance: Any,
    zone: InventoryZoneParams,
    *,
    return_components: bool = False,
) -> Any | tuple[Any, dict[str, Any]]:
    """
    Normalized sigmoid zone-of-indifference penalty in [0, 1].

    Z(imb) = (a_lo * sigmoid((comfort_low - imb) / d_lo)
              + a_hi * sigmoid((imb - comfort_high) / d_hi))
             / (a_lo + a_hi)
    """

    torch = _torch()
    cap_sum = float(zone.low_penalty_cap) + float(zone.high_penalty_cap)
    if cap_sum <= 0.0:
        zero = torch.zeros_like(imbalance)
        if return_components:
            return zero, {
                "inventory_zone_low_penalty": zero,
                "inventory_zone_high_penalty": zero,
            }
        return zero
    d_lo = max(1e-9, float(zone.low_ramp_width))
    d_hi = max(1e-9, float(zone.high_ramp_width))
    low_penalty = float(zone.low_penalty_cap) * torch.sigmoid(
        (float(zone.comfort_low) - imbalance) / d_lo
    )
    high_penalty = float(zone.high_penalty_cap) * torch.sigmoid(
        (imbalance - float(zone.comfort_high)) / d_hi
    )
    total = (low_penalty + high_penalty) / cap_sum
    if return_components:
        return total, {
            "inventory_zone_low_penalty": low_penalty / cap_sum,
            "inventory_zone_high_penalty": high_penalty / cap_sum,
        }
    return total


def inventory_zone_path_mean(
    inventory_path: Any,
    *,
    cash_path: Any,
    mid_path: Any,
    zone: InventoryZoneParams | None,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    if zone is None:
        zero = torch.zeros((), dtype=inventory_path.dtype, device=inventory_path.device)
        return zero, {}
    imbalance = wallet_imbalance(inventory_path, mid_price=mid_path, cash=cash_path)
    step_penalty, step_components = inventory_zone_penalty(
        imbalance,
        zone,
        return_components=True,
    )
    path_mean = step_penalty.mean()
    components = {
        "inventory_zone_penalty": path_mean,
        "inventory_zone_imbalance": imbalance.mean(),
        "inventory_zone_low_penalty": step_components["inventory_zone_low_penalty"].mean(),
        "inventory_zone_high_penalty": step_components["inventory_zone_high_penalty"].mean(),
    }
    return path_mean, components


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
    reservation_r_offset = None
    if float(l_param.reservation_r_max) > 0.0:
        reservation_r_offset = reservation_r_offset_from_params(
            action_flat,
            n_components=n_components,
            r_max=float(l_param.reservation_r_max),
        )
    mid_t = torch.tensor(float(mid_price), dtype=spread_grid.dtype, device=spread_grid.device)
    depth_kw = {
        "l_param": l_param,
        "mid_price": mid_t,
        "reservation_r_offset": reservation_r_offset,
    }

    depth_bid = effective_depth_bid_tensor(
        action_flat,
        spread_grid,
        n_components,
        quote_size_min=bid_quote_size_min,
        quote_size_max=bid_quote_size_max,
        **depth_kw,
    )
    depth_ask = effective_depth_ask_tensor(
        action_flat,
        spread_grid,
        n_components,
        quote_size_min=ask_quote_size_min,
        quote_size_max=ask_quote_size_max,
        **depth_kw,
    )
    q_bid = depth_bid[-1]
    q_ask = depth_ask[-1]

    depth_touch_bid = effective_depth_bid_tensor(
        action_flat,
        torch.tensor([touch], dtype=spread_grid.dtype, device=spread_grid.device),
        n_components,
        quote_size_min=bid_quote_size_min,
        quote_size_max=bid_quote_size_max,
        **depth_kw,
    )[0]
    depth_touch_ask = effective_depth_ask_tensor(
        action_flat,
        torch.tensor([touch], dtype=spread_grid.dtype, device=spread_grid.device),
        n_components,
        quote_size_min=ask_quote_size_min,
        quote_size_max=ask_quote_size_max,
        **depth_kw,
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
            **depth_kw,
        ) + _market_level_from_cumulative(market_cum_bid)
        ask_marg = depth_ask_derivative_tensor(
            action_flat,
            spread_grid,
            n_components,
            **depth_kw,
        ) + _market_level_from_cumulative(market_cum_ask)
    else:
        d_bid_touch = depth_touch_bid
        d_ask_touch = depth_touch_ask
        bid_marg = depth_bid_derivative_tensor(
            action_flat,
            spread_grid,
            n_components,
            **depth_kw,
        )
        ask_marg = depth_ask_derivative_tensor(
            action_flat,
            spread_grid,
            n_components,
            **depth_kw,
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
    if reservation_r_offset is not None:
        components["reservation_r_offset"] = reservation_r_offset
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


def _ce_magnitude(value: Any) -> float:
    torch = _torch()
    if torch.is_tensor(value):
        return abs(float(value.detach().item()))
    return abs(float(value))


def apply_ce_objective_normalization(
    objective: Any,
    components: dict[str, Any],
    *,
    normalize: ObjectiveNormalizeParams | None,
    ce_ewma_scale: float,
) -> tuple[Any, dict[str, Any], float]:
    """
    Scale ``risk_adjusted_return`` by the EWMA reference magnitude.

    Uses the scale from the previous decision during optimization; returns the scale
    used so callers can update the EWMA after the frame completes.
    """

    if normalize is None or not normalize.enabled:
        return objective, components, float(ce_ewma_scale)
    ce = components.get("risk_adjusted_return")
    if ce is None:
        return objective, components, float(ce_ewma_scale)
    scale = max(float(ce_ewma_scale), 0.0)
    if scale <= 0.0:
        scale = max(_ce_magnitude(ce), 1e-12)
        return objective, dict(components), scale
    ce_norm = ce / scale
    out = dict(components)
    out["risk_adjusted_return_raw"] = ce
    out["risk_adjusted_return"] = ce_norm
    out["objective_ce_ewma_scale"] = scale
    return objective - ce + ce_norm, out, scale


def update_ce_ewma_scale(
    ce_ewma_scale: float,
    components: dict[str, Any],
    *,
    normalize: ObjectiveNormalizeParams | None,
) -> float:
    """Advance the CE EWMA scale from the raw (unnormalized) closure CE."""

    if normalize is None or not normalize.enabled:
        return float(ce_ewma_scale)
    raw = components.get("risk_adjusted_return_raw", components.get("risk_adjusted_return"))
    if raw is None:
        return float(ce_ewma_scale)
    magnitude = _ce_magnitude(raw)
    if magnitude <= 0.0:
        return float(ce_ewma_scale)
    decay = float(normalize.ce_ewma_decay)
    if ce_ewma_scale <= 0.0:
        return max(magnitude, 1e-12)
    return decay * float(ce_ewma_scale) + (1.0 - decay) * magnitude


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
