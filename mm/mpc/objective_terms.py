"""Normalized soft objective terms for MPC (book validity, viability availability)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mm.mpc.l_param import LParamConfig, materialize_side_torch, reservation_r_offset_from_params
from mm.mpc.optimizer_config_keys import (
    MAX_ABS_SKEW,
    PRESENCE_REFERENCE_QTY,
    REMOVED_BOOK_QUALITY_ALIASES,
    RESERVATION_SKEW,
    SOFT_SCORING,
    TIGHTNESS_SPREAD_SCALE,
    reject_removed_book_quality_keys,
    reject_removed_keys,
    reject_removed_reservation_skew_keys,
)
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
class ViabilityParams:
    """Availability shortfall on total wallet balances (docs/viability_term.md)."""

    min_viable_quote: float = 50_000.0
    margin_buffer: float = 5_000.0
    availability_target: float = 0.99
    shortfall_tolerance: float = 0.05
    lambda_init: float = 0.0
    lambda_learning_rate: float = 0.02

    @classmethod
    def from_mapping(cls, payload: Any) -> ViabilityParams | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("viability must be an object")
        removed = sorted(
            key
            for key in payload
            if key
            in {
                "comfort_low",
                "comfort_high",
                "low_ramp_width",
                "high_ramp_width",
                "low_penalty_cap",
                "high_penalty_cap",
                "min_quote_qty",
                "min_free_quote",
                "margin_sharpness",
                "lambda_leakage",
                "lambda_reset_on_target",
                "lambda_log_update",
            }
        )
        if removed:
            raise ValueError(
                "removed viability keys; use "
                "{min_viable_quote, margin_buffer, availability_target, "
                "shortfall_tolerance, lambda_init, lambda_learning_rate}: "
                + ", ".join(removed)
            )
        availability_target = float(payload.get("availability_target", 0.99))
        if not (0.0 < availability_target <= 1.0):
            raise ValueError("viability.availability_target must be in (0, 1]")
        shortfall_tolerance = float(payload.get("shortfall_tolerance", 0.05))
        if shortfall_tolerance < 0.0:
            raise ValueError("viability.shortfall_tolerance must be >= 0")
        lambda_learning_rate = float(payload.get("lambda_learning_rate", 0.02))
        if lambda_learning_rate <= 0.0:
            raise ValueError("viability.lambda_learning_rate must be > 0")
        return cls(
            min_viable_quote=max(1e-12, float(payload.get("min_viable_quote", 50_000.0))),
            margin_buffer=max(1e-9, float(payload.get("margin_buffer", 5_000.0))),
            availability_target=availability_target,
            shortfall_tolerance=shortfall_tolerance,
            lambda_init=max(0.0, float(payload.get("lambda_init", 0.0))),
            lambda_learning_rate=lambda_learning_rate,
        )

    def as_mapping(self) -> dict[str, float]:
        return {
            "min_viable_quote": self.min_viable_quote,
            "margin_buffer": self.margin_buffer,
            "availability_target": self.availability_target,
            "shortfall_tolerance": self.shortfall_tolerance,
            "lambda_init": self.lambda_init,
            "lambda_learning_rate": self.lambda_learning_rate,
        }


@dataclass(frozen=True, slots=True)
class ValidityParams:
    """Book-shape validity gate with fixed penalty scale (docs/validity_term.md)."""

    min_score: float = 0.3
    penalty_scale: float = 100.0

    @classmethod
    def from_mapping(cls, payload: Any) -> ValidityParams | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("validity must be an object")
        if "w_book" in payload:
            raise ValueError(
                "validity must not include w_book; use "
                "{min_score, penalty_scale} and book_quality.soft_scoring"
            )
        min_score = float(payload.get("min_score", 0.3))
        if not (0.0 <= min_score < 1.0):
            raise ValueError("validity.min_score must be in [0, 1)")
        penalty_scale = float(payload.get("penalty_scale", 100.0))
        if penalty_scale < 0.0:
            raise ValueError("validity.penalty_scale must be >= 0")
        return cls(
            min_score=min_score,
            penalty_scale=penalty_scale,
        )

    def as_mapping(self) -> dict[str, float]:
        return {
            "min_score": self.min_score,
            "penalty_scale": self.penalty_scale,
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


_BOOK_QUALITY_SOFT_KEYS = frozenset(
    {
        TIGHTNESS_SPREAD_SCALE,
        PRESENCE_REFERENCE_QTY,
        "touch_spread_fraction",
        "measure",
        "v_presence",
        "v_tightness",
        "v_balance",
        "v_uniformity",
    }
)
_LEGACY_SPREAD_STRUCTURE_KEYS = frozenset(
    {"center_sum_max", "center_min_depth", "center_k_epsilon"}
)


@dataclass(frozen=True, slots=True)
class BookQualityHardStructure:
    """Hard decoder geometry for first-touch spread legs."""

    action_max_spread: float
    action_min_depth: float = 0.0
    action_k_epsilon: float = 0.01

    @classmethod
    def from_mapping(cls, payload: Any) -> BookQualityHardStructure | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("book_quality.hard_structure must be an object")
        legacy = sorted(k for k in _LEGACY_SPREAD_STRUCTURE_KEYS if k in payload)
        if legacy:
            raise ValueError(
                "book_quality.hard_structure keys renamed: use "
                "action_max_spread, action_min_depth, action_k_epsilon; remove "
                + ", ".join(legacy)
            )
        action_max_spread = float(payload.get("action_max_spread", 0.0))
        if action_max_spread <= 0.0:
            raise ValueError("book_quality.hard_structure.action_max_spread must be > 0")
        return cls(
            action_max_spread=action_max_spread,
            action_min_depth=max(0.0, float(payload.get("action_min_depth", 0.0))),
            action_k_epsilon=float(payload.get("action_k_epsilon", 0.01)),
        )

    def as_mapping(self) -> dict[str, float]:
        out: dict[str, float] = {"action_max_spread": self.action_max_spread}
        if self.action_min_depth > 0.0:
            out["action_min_depth"] = self.action_min_depth
        if self.action_k_epsilon != 0.01:
            out["action_k_epsilon"] = self.action_k_epsilon
        return out


@dataclass(frozen=True, slots=True)
class BookQualitySoftScoring:
    """Soft book-quality objective scoring references and sub-weights."""

    tightness_spread_scale: float = 50.0
    presence_reference_qty: float = 1.0
    touch_spread_fraction: float = 0.01
    measure: str = "quoted"
    v_presence: float = 0.25
    v_tightness: float = 0.25
    v_balance: float = 0.25
    v_uniformity: float = 0.25

    @classmethod
    def from_mapping(cls, payload: Any) -> BookQualitySoftScoring:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError(f"book_quality.{SOFT_SCORING} must be an object")
        reject_removed_keys(payload, removed=REMOVED_BOOK_QUALITY_ALIASES, scope=f"book_quality.{SOFT_SCORING}")
        measure = str(payload.get("measure", "quoted")).strip().lower()
        if measure not in ("quoted", "combined"):
            raise ValueError(f"book_quality.{SOFT_SCORING}.measure must be 'quoted' or 'combined'")
        return cls(
            tightness_spread_scale=max(1e-9, float(payload.get(TIGHTNESS_SPREAD_SCALE, 50.0))),
            presence_reference_qty=max(1e-9, float(payload.get(PRESENCE_REFERENCE_QTY, 1.0))),
            touch_spread_fraction=max(1e-9, float(payload.get("touch_spread_fraction", 0.01))),
            measure=measure,
            v_presence=float(payload.get("v_presence", 0.25)),
            v_tightness=float(payload.get("v_tightness", 0.25)),
            v_balance=float(payload.get("v_balance", 0.25)),
            v_uniformity=float(payload.get("v_uniformity", 0.25)),
        )

    def as_mapping(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {
            TIGHTNESS_SPREAD_SCALE: self.tightness_spread_scale,
            PRESENCE_REFERENCE_QTY: self.presence_reference_qty,
            "measure": self.measure,
        }
        if self.touch_spread_fraction != 0.01:
            out["touch_spread_fraction"] = self.touch_spread_fraction
        if self.v_presence != 0.25:
            out["v_presence"] = self.v_presence
        if self.v_tightness != 0.25:
            out["v_tightness"] = self.v_tightness
        if self.v_balance != 0.25:
            out["v_balance"] = self.v_balance
        if self.v_uniformity != 0.25:
            out["v_uniformity"] = self.v_uniformity
        return out


@dataclass(frozen=True, slots=True)
class BookQualityParams:
    """Venue book geometry: optional hard decoder structure plus soft scoring."""

    hard_structure: BookQualityHardStructure | None = None
    soft_scoring: BookQualitySoftScoring = BookQualitySoftScoring()

    @property
    def touch_spread_fraction(self) -> float:
        return self.soft_scoring.touch_spread_fraction

    @property
    def measure(self) -> str:
        return self.soft_scoring.measure

    @property
    def v_presence(self) -> float:
        return self.soft_scoring.v_presence

    @property
    def v_tightness(self) -> float:
        return self.soft_scoring.v_tightness

    @property
    def v_balance(self) -> float:
        return self.soft_scoring.v_balance

    @property
    def v_uniformity(self) -> float:
        return self.soft_scoring.v_uniformity

    @property
    def tightness_spread_scale(self) -> float:
        return self.soft_scoring.tightness_spread_scale

    @property
    def presence_reference_qty(self) -> float:
        return self.soft_scoring.presence_reference_qty

    @classmethod
    def from_mapping(cls, payload: Any) -> BookQualityParams | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("book_quality must be an object")
        if "spread_structure" in payload:
            raise ValueError(
                "spread_structure belongs under book_quality.hard_structure "
                "(action_max_spread, action_min_depth, action_k_epsilon)"
            )
        reject_removed_book_quality_keys(payload)
        flat_soft = sorted(k for k in _BOOK_QUALITY_SOFT_KEYS if k in payload)
        if flat_soft:
            raise ValueError(
                f"book_quality soft keys belong under book_quality.{SOFT_SCORING}: "
                + ", ".join(flat_soft)
            )
        hard = BookQualityHardStructure.from_mapping(payload.get("hard_structure"))
        soft = BookQualitySoftScoring.from_mapping(payload.get(SOFT_SCORING))
        return cls(hard_structure=hard, soft_scoring=soft)

    def as_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {SOFT_SCORING: self.soft_scoring.as_mapping()}
        if self.hard_structure is not None:
            out["hard_structure"] = self.hard_structure.as_mapping()
        return out


def soft_availability_margin(
    inventory: Any,
    *,
    cash: Any,
    mid_price: Any,
    params: ViabilityParams,
) -> tuple[Any, dict[str, Any]]:
    """
    Smooth two-sided quoting availability in [0, 1] from total wallet margins.

    ``cash`` is total quote balance q_t (free + locked); ``inventory`` is total
    base balance b_t. ``min_viable_quote`` is q_min. Both margins are in quote
    currency for a shared ``margin_buffer``:

    m_bid = q_t - q_min; m_ask = b_t * p_t - q_min;
    a_side = sigmoid(m_side / margin_buffer); A = a_bid * a_ask.
    """

    torch = _torch()
    q_min = max(1e-12, float(params.min_viable_quote))
    delta = max(1e-9, float(params.margin_buffer))
    mid_safe = torch.clamp(mid_price, min=1e-12)
    quote_total = torch.clamp(cash, min=0.0)
    base_total = torch.clamp(inventory, min=0.0)
    base_notional = base_total * mid_safe
    m_bid = quote_total - q_min
    m_ask = base_notional - q_min
    a_bid = torch.sigmoid(m_bid / delta)
    a_ask = torch.sigmoid(m_ask / delta)
    availability = a_bid * a_ask
    return availability, {
        "viability_bid_margin": m_bid,
        "viability_ask_margin": m_ask,
        "viability_base_notional": base_notional,
        "viability_step_availability": availability,
    }


def viability_path_shortfall(
    inventory_path: Any,
    *,
    cash_path: Any,
    mid_path: Any,
    params: ViabilityParams | None,
) -> tuple[Any, dict[str, Any]]:
    """
    Expected availability shortfall over Monte Carlo paths.

    ``inventory_path``, ``cash_path``, and ``mid_path`` must be aligned snapshots
    of simulated **total** wallet balances (free + locked) along each rollout path.

    Per path: A_min = min_{h>=1} A_h (exclude pre-action snapshot at h=0);
    S = max(0, A* - A_min); return mean(S).
    """

    torch = _torch()
    if params is None:
        zero = torch.zeros((), dtype=inventory_path.dtype, device=inventory_path.device)
        return zero, {}
    step_avail, step_parts = soft_availability_margin(
        inventory_path,
        cash=cash_path,
        mid_price=mid_path,
        params=params,
    )
    if step_avail.shape[1] <= 1:
        zero = torch.zeros((), dtype=inventory_path.dtype, device=inventory_path.device)
        return zero, {}
    path_min_avail = step_avail[:, 1:].min(dim=1).values
    shortfall_per_path = torch.clamp(
        float(params.availability_target) - path_min_avail,
        min=0.0,
    )
    shortfall_mean = shortfall_per_path.mean()
    components = {
        "viability_shortfall": shortfall_mean,
        "viability_min_availability": path_min_avail.mean(),
        "viability_step_availability": step_parts["viability_step_availability"].mean(),
        "viability_bid_margin": step_parts["viability_bid_margin"].mean(),
        "viability_ask_margin": step_parts["viability_ask_margin"].mean(),
    }
    return shortfall_mean, components


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
    q_ref = max(1e-9, float(params.presence_reference_qty))
    presence = smooth_min / (q_ref + smooth_min)

    spread_sum = bid_x0[0] + ask_x0[0]
    tightness = torch.exp(-spread_sum / max(1e-9, float(params.tightness_spread_scale)))

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
        "validity_penalty",
        "viability_penalty",
        "global_validity_penalty",
        "global_viability_penalty",
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


_LAMBDA_INIT_FLOOR = 1e-300


def viability_theta_from_lambda_init(lambda_init: float) -> float:
    """Map configured ``lambda_init`` to log-space dual state ``θ`` with ``λ = exp(θ)``."""

    return math.log(max(0.0, float(lambda_init), _LAMBDA_INIT_FLOOR))


def viability_lambda_from_theta(viability_theta: float) -> float:
    """Positive Lagrange multiplier from log-space dual state."""

    theta = float(viability_theta)
    if not math.isfinite(theta):
        return 0.0
    if theta <= math.log(_LAMBDA_INIT_FLOOR):
        return 0.0
    lam = math.exp(theta)
    return lam if math.isfinite(lam) and lam > 0.0 else 0.0


def stamp_learned_viability_dual(
    components: dict[str, Any],
    *,
    local_theta: float,
    global_theta: float | None = None,
) -> None:
    """Record post-update dual state (λ for the next MPC decision) in diagnostics."""

    components["viability_lambda_learned"] = viability_lambda_from_theta(local_theta)
    components["viability_lambda_theta_learned"] = float(local_theta)
    if global_theta is not None:
        components["global_viability_lambda_learned"] = viability_lambda_from_theta(global_theta)
        components["global_viability_lambda_theta_learned"] = float(global_theta)


def _viability_shortfall_from_components(
    components: dict[str, Any],
    *,
    shortfall_key: str = "viability_shortfall",
    shortfall_raw_key: str = "viability_penalty_raw",
) -> Any | None:
    raw = components.get(shortfall_key)
    if raw is None:
        raw = components.get(shortfall_raw_key)
    return raw


def _float_component(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw.detach().item() if hasattr(raw, "detach") else raw)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _signed_availability_gap(
    components: dict[str, Any],
    *,
    params: ViabilityParams,
    min_availability_key: str,
) -> float | None:
    a_min = _float_component(components.get(min_availability_key))
    if a_min is None:
        return None
    return float(params.availability_target) - a_min


def update_viability_theta(
    viability_theta: float,
    components: dict[str, Any],
    *,
    params: ViabilityParams | None,
    min_availability_key: str = "viability_min_availability",
) -> float:
    """
    Log-space dual ascent: θ <- θ + η (A* - E[A_min]); λ = exp(θ).

    Uses the signed availability gap (not one-sided shortfall) so λ can decrease
    when availability is above target. No reset, leakage, or ε in the dual update.
    """

    if params is None:
        return float(viability_theta)
    gap = _signed_availability_gap(
        components,
        params=params,
        min_availability_key=min_availability_key,
    )
    if gap is None:
        return float(viability_theta)
    eta = float(params.lambda_learning_rate)
    return float(viability_theta) + eta * gap


def validity_book_shortfall(
    book_score: Any,
    *,
    params: ValidityParams | None,
) -> tuple[Any, dict[str, Any]]:
    """Shortfall below minimum acceptable h=0 book-quality score: max(0, B_min - B)."""

    torch = _torch()
    if params is None or float(params.penalty_scale) <= 0.0:
        if torch.is_tensor(book_score):
            zero = torch.zeros((), dtype=book_score.dtype, device=book_score.device)
        else:
            zero = 0.0
        return zero, {}
    if torch.is_tensor(book_score):
        shortfall = torch.clamp(float(params.min_score) - book_score, min=0.0)
    else:
        shortfall = max(0.0, float(params.min_score) - float(book_score))
    return shortfall, {
        "validity_shortfall": shortfall,
        "validity_book_score": book_score,
    }


def combine_validity_term(
    base_objective: Any,
    components: dict[str, Any],
    *,
    params: ValidityParams | None,
    book_score: Any | None = None,
    book_components: dict[str, Any] | None = None,
    validity_term: str = "validity_penalty",
    component_prefix: str = "",
) -> tuple[Any, dict[str, Any]]:
    """Subtract fixed-scale validity penalty: -mu * max(0, B_min - B)."""

    out = dict(components)
    objective = base_objective
    if book_score is None or params is None or float(params.penalty_scale) <= 0.0:
        return objective, out
    shortfall, validity_components = validity_book_shortfall(book_score, params=params)
    weighted = -float(params.penalty_scale) * shortfall
    objective = objective + weighted
    out[validity_term] = weighted
    for key, value in validity_components.items():
        if key == "validity_shortfall":
            out[f"{validity_term}_raw"] = value
        else:
            out[f"{component_prefix}{key}"] = value
    for key, value in (book_components or {}).items():
        if key == "book_quality":
            out[f"{component_prefix}validity_book_score"] = value
        else:
            diag_key = key if key.startswith("book_") else f"book_{key}"
            out[f"{component_prefix}{diag_key}"] = value
    return objective, out


def combine_viability_term(
    base_objective: Any,
    components: dict[str, Any],
    *,
    viability_theta: float,
    viability_params: ViabilityParams | None = None,
    viability_shortfall: Any | None = None,
    viability_components: dict[str, Any] | None = None,
    viability_term: str = "viability_penalty",
    component_prefix: str = "",
) -> tuple[Any, dict[str, Any]]:
    """Subtract adaptive Lagrange penalty: -exp(theta) * expected availability shortfall."""

    out = dict(components)
    objective = base_objective
    if viability_shortfall is None:
        return objective, out
    lam = viability_lambda_from_theta(viability_theta)
    weighted = -lam * viability_shortfall
    objective = objective + weighted
    out[viability_term] = weighted
    out[f"{component_prefix}viability_lambda"] = lam
    out[f"{component_prefix}viability_lambda_theta"] = float(viability_theta)
    for key, value in (viability_components or {}).items():
        if key == "viability_shortfall":
            out[f"{viability_term}_raw"] = value
        else:
            out[f"{component_prefix}{key}"] = value
    return objective, out
