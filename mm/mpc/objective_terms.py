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
    """Two-sided wallet-reserve constraint on total balances.

    Penalty term is ``-λ · S`` with squared hinge deficits (zero above
    ``min_viable_quote`` or relative wealth ratio ``min_viable_ratio``).
    Dual ``λ`` is a bounded linear multiplier updated from an EWMA of ``S``,
    not ``exp(θ)``.

    When ``recovery_fraction > 0``, an explicit deficit recovery term penalizes
    failure to reduce existing reserve deficits by at least fraction ``recovery_fraction``
    per step.
    """

    min_viable_quote: float = 50_000.0
    shortfall_tolerance: float = 0.05
    lambda_init: float = 0.0
    lambda_learning_rate: float = 0.02
    lambda_max: float = 100.0
    dual_ewma_decay: float = 0.9
    recovery_fraction: float = 0.0
    min_viable_ratio: float = 0.0
    min_viable_bps: float = 0.0

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
                "{min_viable_quote, shortfall_tolerance, lambda_init, "
                "lambda_learning_rate, lambda_max, dual_ewma_decay, "
                "recovery_fraction, min_viable_ratio, min_viable_bps}: "
                + ", ".join(removed)
            )
        min_viable_ratio = float(payload.get("min_viable_ratio", 0.0))
        min_viable_bps = float(payload.get("min_viable_bps", 0.0))
        if min_viable_bps > 0.0 and min_viable_ratio <= 0.0:
            min_viable_ratio = min_viable_bps / 10000.0
        elif min_viable_ratio > 0.0 and min_viable_bps <= 0.0:
            min_viable_bps = min_viable_ratio * 10000.0
        if min_viable_ratio < 0.0 or min_viable_ratio >= 1.0:
            raise ValueError("viability.min_viable_ratio must be in [0, 1)")
        min_viable_quote_val = payload.get("min_viable_quote")
        if min_viable_quote_val is not None:
            min_viable_quote = max(1e-12, float(min_viable_quote_val))
        elif min_viable_ratio > 0.0:
            min_viable_quote = 0.0
        else:
            min_viable_quote = 50_000.0
        shortfall_tolerance = float(payload.get("shortfall_tolerance", 0.05))
        if shortfall_tolerance < 0.0:
            raise ValueError("viability.shortfall_tolerance must be >= 0")
        lambda_learning_rate = float(payload.get("lambda_learning_rate", 0.02))
        if lambda_learning_rate <= 0.0:
            raise ValueError("viability.lambda_learning_rate must be > 0")
        lambda_max = float(payload.get("lambda_max", 100.0))
        if lambda_max <= 0.0:
            raise ValueError("viability.lambda_max must be > 0")
        dual_ewma_decay = float(payload.get("dual_ewma_decay", 0.9))
        if not (0.0 <= dual_ewma_decay < 1.0):
            raise ValueError("viability.dual_ewma_decay must be in [0, 1)")
        recovery_fraction = float(payload.get("recovery_fraction", 0.0))
        if not (0.0 <= recovery_fraction <= 1.0):
            raise ValueError("viability.recovery_fraction must be in [0, 1]")
        lambda_init = max(0.0, float(payload.get("lambda_init", 0.0)))
        return cls(
            min_viable_quote=min_viable_quote,
            shortfall_tolerance=shortfall_tolerance,
            lambda_init=min(lambda_init, lambda_max),
            lambda_learning_rate=lambda_learning_rate,
            lambda_max=lambda_max,
            dual_ewma_decay=dual_ewma_decay,
            recovery_fraction=recovery_fraction,
            min_viable_ratio=min_viable_ratio,
            min_viable_bps=min_viable_bps,
        )

    def as_mapping(self) -> dict[str, float]:
        out = {
            "min_viable_quote": self.min_viable_quote,
            "shortfall_tolerance": self.shortfall_tolerance,
            "lambda_init": self.lambda_init,
            "lambda_learning_rate": self.lambda_learning_rate,
            "lambda_max": self.lambda_max,
            "dual_ewma_decay": self.dual_ewma_decay,
            "recovery_fraction": self.recovery_fraction,
        }
        if self.min_viable_ratio > 0.0:
            out["min_viable_ratio"] = self.min_viable_ratio
            out["min_viable_bps"] = self.min_viable_bps
        return out


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
    action_max_spread_bps: float | None = None

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
        action_max_spread_bps = payload.get("action_max_spread_bps")
        if action_max_spread_bps is not None:
            action_max_spread = float(action_max_spread_bps)
            action_max_spread_bps = float(action_max_spread_bps)
        else:
            action_max_spread = float(payload.get("action_max_spread", 0.0))
            action_max_spread_bps = None
        if action_max_spread <= 0.0:
            raise ValueError("book_quality.hard_structure.action_max_spread must be > 0")
        return cls(
            action_max_spread=action_max_spread,
            action_min_depth=max(0.0, float(payload.get("action_min_depth", 0.0))),
            action_k_epsilon=float(payload.get("action_k_epsilon", 0.01)),
            action_max_spread_bps=action_max_spread_bps,
        )

    def resolved_action_max_spread(self, mid: float | None = None) -> float:
        if self.action_max_spread_bps is not None and mid is not None and mid > 0:
            return float(mid) * float(self.action_max_spread_bps) / 10000.0
        return float(self.action_max_spread)

    def as_mapping(self) -> dict[str, float]:
        out: dict[str, float] = {"action_max_spread": self.action_max_spread}
        if self.action_max_spread_bps is not None:
            out["action_max_spread_bps"] = self.action_max_spread_bps
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
    tightness_spread_scale_bps: float | None = None

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
        tightness_spread_scale_bps = payload.get("tightness_spread_scale_bps")
        if tightness_spread_scale_bps is not None:
            tightness_spread_scale = float(tightness_spread_scale_bps)
            tightness_spread_scale_bps = float(tightness_spread_scale_bps)
        else:
            tightness_spread_scale = max(1e-9, float(payload.get(TIGHTNESS_SPREAD_SCALE, 50.0)))
            tightness_spread_scale_bps = None
        return cls(
            tightness_spread_scale=tightness_spread_scale,
            presence_reference_qty=max(1e-9, float(payload.get(PRESENCE_REFERENCE_QTY, 1.0))),
            touch_spread_fraction=max(1e-9, float(payload.get("touch_spread_fraction", 0.01))),
            measure=measure,
            v_presence=float(payload.get("v_presence", 0.25)),
            v_tightness=float(payload.get("v_tightness", 0.25)),
            v_balance=float(payload.get("v_balance", 0.25)),
            v_uniformity=float(payload.get("v_uniformity", 0.25)),
            tightness_spread_scale_bps=tightness_spread_scale_bps,
        )

    def resolved_tightness_spread_scale(self, mid: float | None = None) -> float:
        if self.tightness_spread_scale_bps is not None and mid is not None and mid > 0:
            return float(mid) * float(self.tightness_spread_scale_bps) / 10000.0
        return float(self.tightness_spread_scale)

    def as_mapping(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {
            TIGHTNESS_SPREAD_SCALE: self.tightness_spread_scale,
            PRESENCE_REFERENCE_QTY: self.presence_reference_qty,
            "measure": self.measure,
        }
        if self.tightness_spread_scale_bps is not None:
            out["tightness_spread_scale_bps"] = self.tightness_spread_scale_bps
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


def wallet_quadratic_deficit(
    inventory: Any,
    *,
    cash: Any,
    mid_price: Any,
    params: ViabilityParams,
) -> tuple[Any, dict[str, Any]]:
    """Squared-hinge two-sided reserve deficit from total wallet balances.

    ``cash`` is total quote ``q_t`` (free + locked); ``inventory`` is total
    base ``b_t``. Both deficits are in quote-currency units of ``q_min``:

        d_bid = [q_min - q_t]_+ / q_min
        d_ask = [q_min - b_t p_t]_+ / q_min
        S     = d_bid² + d_ask²

    ``S`` is zero above the reserve on both sides (no inventory headwind)
    and steepens as a side falls further below ``q_min``.
    """

    torch = _torch()
    mid_safe = torch.clamp(mid_price, min=1e-12)
    quote_total = torch.clamp(cash, min=0.0)
    base_total = torch.clamp(inventory, min=0.0)
    base_notional = base_total * mid_safe
    total_wealth = quote_total + base_notional

    if float(params.min_viable_ratio) > 0.0:
        ratio = float(params.min_viable_ratio)
        q_min = torch.clamp(total_wealth * ratio, min=1e-12)
    else:
        q_min = max(1e-12, float(params.min_viable_quote))

    m_bid = quote_total - q_min
    m_ask = base_notional - q_min
    d_bid = torch.clamp(-m_bid, min=0.0) / q_min
    d_ask = torch.clamp(-m_ask, min=0.0) / q_min
    deficit = d_bid * d_bid + d_ask * d_ask
    health = 1.0 / (1.0 + deficit)
    return deficit, {
        "viability_bid_margin": m_bid,
        "viability_ask_margin": m_ask,
        "viability_base_notional": base_notional,
        "viability_bid_deficit": d_bid,
        "viability_ask_deficit": d_ask,
        "viability_step_deficit": deficit,
        "viability_step_availability": health,
    }


def soft_availability_margin(
    inventory: Any,
    *,
    cash: Any,
    mid_price: Any,
    params: ViabilityParams,
) -> tuple[Any, dict[str, Any]]:
    """Backward-compatible alias: returns diagnostic health ``1 / (1 + S)``."""

    deficit, parts = wallet_quadratic_deficit(
        inventory,
        cash=cash,
        mid_price=mid_price,
        params=params,
    )
    return parts["viability_step_availability"], parts


def viability_path_shortfall(
    inventory_path: Any,
    *,
    cash_path: Any,
    mid_path: Any,
    params: ViabilityParams | None,
) -> tuple[Any, dict[str, Any]]:
    """Expected next-step quadratic reserve deficit over Monte Carlo paths.

    Paths are simulated **total** wallet balances. Index 0 is the pre-action
    wallet and is excluded; shortfall uses column 1 (first post-fill step).
    Terminal-horizon recovery does not mask an immediate post-fill hole.
    """

    torch = _torch()
    if params is None:
        zero = torch.zeros((), dtype=inventory_path.dtype, device=inventory_path.device)
        return zero, {}
    step_deficit, step_parts = wallet_quadratic_deficit(
        inventory_path,
        cash=cash_path,
        mid_price=mid_path,
        params=params,
    )
    if step_deficit.shape[1] <= 1:
        zero = torch.zeros((), dtype=inventory_path.dtype, device=inventory_path.device)
        return zero, {}
    path_next = step_deficit[:, 1]
    path_next_health = step_parts["viability_step_availability"][:, 1]
    shortfall_mean = path_next.mean()
    zero = torch.zeros((), dtype=inventory_path.dtype, device=inventory_path.device)

    rho = max(0.0, min(1.0, float(params.recovery_fraction)))
    if rho > 0.0:
        d_bid_0 = step_parts["viability_bid_deficit"][:, 0]
        d_ask_0 = step_parts["viability_ask_deficit"][:, 0]
        d_bid_1 = step_parts["viability_bid_deficit"][:, 1]
        d_ask_1 = step_parts["viability_ask_deficit"][:, 1]
        target_bid = (1.0 - rho) * d_bid_0
        target_ask = (1.0 - rho) * d_ask_0
        rec_bid = torch.where(
            d_bid_0 > 1e-9,
            torch.clamp(d_bid_1 - target_bid, min=0.0),
            torch.zeros_like(d_bid_1),
        )
        rec_ask = torch.where(
            d_ask_0 > 1e-9,
            torch.clamp(d_ask_1 - target_ask, min=0.0),
            torch.zeros_like(d_ask_1),
        )
        rec_step_deficit = rec_bid * rec_bid + rec_ask * rec_ask
        recovery_shortfall = rec_step_deficit.mean()
        rec_bid_mean = rec_bid.mean()
        rec_ask_mean = rec_ask.mean()
    else:
        recovery_shortfall = zero
        rec_bid_mean = zero
        rec_ask_mean = zero

    components = {
        "viability_shortfall": shortfall_mean,
        "viability_min_availability": path_next_health.mean(),
        "viability_step_availability": path_next_health.mean(),
        "viability_bid_margin": step_parts["viability_bid_margin"][:, 1].mean(),
        "viability_ask_margin": step_parts["viability_ask_margin"][:, 1].mean(),
        "viability_base_notional": step_parts["viability_base_notional"][:, 1].mean(),
        "viability_bid_deficit": step_parts["viability_bid_deficit"][:, 1].mean(),
        "viability_ask_deficit": step_parts["viability_ask_deficit"][:, 1].mean(),
        "viability_step_deficit": path_next.mean(),
        "viability_recovery_fraction": rho,
        "viability_recovery_shortfall": recovery_shortfall,
        "viability_recovery_bid_deficit": rec_bid_mean,
        "viability_recovery_ask_deficit": rec_ask_mean,
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
        "viability_recovery_penalty",
        "global_validity_penalty",
        "global_viability_penalty",
        "global_viability_recovery_penalty",
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


def _clip_lambda(value: float, lambda_max: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 0.0
    cap = max(0.0, float(lambda_max))
    if value > cap:
        return cap
    return float(value)


def viability_lambda_from_init(lambda_init: float, *, lambda_max: float = 100.0) -> float:
    """Configured starting multiplier, clipped to ``[0, lambda_max]``."""

    return _clip_lambda(float(lambda_init), lambda_max)


def _viability_shortfall_raw_from_components(
    components: dict[str, Any],
    *,
    component_prefix: str = "",
) -> float | None:
    for key in (
        f"{component_prefix}viability_penalty_raw",
        f"{component_prefix}viability_shortfall",
    ):
        value = _float_component(components.get(key))
        if value is not None:
            return value
    return None


def viability_lambda_applied_from_components(
    components: dict[str, Any],
    *,
    lambda_applied: float,
    component_prefix: str = "",
) -> float:
    """λ used in this decision's viability penalty.

    Prefer ``|penalty| / shortfall`` when both are present so diagnostics match ``J``.
    """

    shortfall = _viability_shortfall_raw_from_components(
        components,
        component_prefix=component_prefix,
    )
    penalty = _float_component(components.get(f"{component_prefix}viability_penalty"))
    if shortfall is not None and penalty is not None and abs(shortfall) > 1e-15:
        implied = abs(penalty) / abs(shortfall)
        if math.isfinite(implied) and implied >= 0.0:
            return implied
    return max(0.0, float(lambda_applied)) if math.isfinite(float(lambda_applied)) else 0.0


def stamp_viability_dual_diagnostics(
    components: dict[str, Any],
    *,
    local_lambda_applied: float,
    local_lambda_learned: float,
    global_lambda_applied: float | None = None,
    global_lambda_learned: float | None = None,
) -> None:
    """Stamp linear dual state on optimizer objective components.

    ``viability_lambda_theta`` is kept as an alias of the applied λ so older
    dashboards still receive a multiplier (it is no longer log-space θ).
    """

    components["viability_lambda"] = viability_lambda_applied_from_components(
        components,
        lambda_applied=local_lambda_applied,
    )
    components["viability_lambda_theta"] = float(local_lambda_applied)
    components["viability_lambda_learned"] = float(local_lambda_learned)
    components["viability_lambda_theta_learned"] = float(local_lambda_learned)
    if global_lambda_applied is not None and global_lambda_learned is not None:
        components["global_viability_lambda"] = viability_lambda_applied_from_components(
            components,
            lambda_applied=global_lambda_applied,
            component_prefix="global_",
        )
        components["global_viability_lambda_theta"] = float(global_lambda_applied)
        components["global_viability_lambda_learned"] = float(global_lambda_learned)
        components["global_viability_lambda_theta_learned"] = float(global_lambda_learned)


def stamp_learned_viability_dual(
    components: dict[str, Any],
    *,
    local_lambda: float,
    global_lambda: float | None = None,
) -> None:
    """Deprecated: only stamps learned keys. Use ``stamp_viability_dual_diagnostics``."""

    components["viability_lambda_learned"] = float(local_lambda)
    components["viability_lambda_theta_learned"] = float(local_lambda)
    if global_lambda is not None:
        components["global_viability_lambda_learned"] = float(global_lambda)
        components["global_viability_lambda_theta_learned"] = float(global_lambda)


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


def update_viability_lambda(
    viability_lambda: float,
    components: dict[str, Any],
    *,
    params: ViabilityParams | None,
    ewma: float | None = None,
    shortfall_key: str = "viability_shortfall",
    ewma_key: str = "viability_shortfall_ewma",
    gap_key: str = "viability_dual_gap",
) -> tuple[float, float]:
    """Linear dual ascent on EWMA shortfall: λ ← clip(λ + η (S̄ − ε), 0, λ_max).

    ``S`` is the quadratic deficit from the inner solve (fixed during Adam).
    ``ε`` is ``shortfall_tolerance``. The EWMA decouples Monte Carlo noise
    from the trading-frame dual step. Returns ``(new_lambda, new_ewma)``.
    """

    current = float(viability_lambda)
    if not math.isfinite(current) or current < 0.0:
        current = 0.0
    if params is None:
        return current, (float(ewma) if ewma is not None and math.isfinite(float(ewma)) else 0.0)
    shortfall = _float_component(components.get(shortfall_key))
    if shortfall is None:
        shortfall = _float_component(components.get("viability_penalty_raw"))
    if shortfall is None:
        prior = float(ewma) if ewma is not None and math.isfinite(float(ewma)) else 0.0
        return current, prior
    decay = float(params.dual_ewma_decay)
    if ewma is None or not math.isfinite(float(ewma)):
        new_ewma = float(shortfall)
    else:
        new_ewma = decay * float(ewma) + (1.0 - decay) * float(shortfall)
    gap = new_ewma - float(params.shortfall_tolerance)
    cap = float(params.lambda_max)
    new_lambda = _clip_lambda(current + float(params.lambda_learning_rate) * gap, cap)
    components[ewma_key] = new_ewma
    components[gap_key] = gap
    return new_lambda, new_ewma


def reset_viability_lambda(
    *,
    current_lambda: float,
    reset_fraction: float = 0.5,
) -> float:
    """Partial reset of the linear dual state: ``λ * (1 - reset_fraction)``."""

    if not 0.0 <= reset_fraction <= 1.0:
        raise ValueError(
            f"reset_fraction must be in [0, 1], got {reset_fraction!r}"
        )
    value = float(current_lambda) * (1.0 - float(reset_fraction))
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return value


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
    viability_lambda: float,
    viability_params: ViabilityParams | None = None,
    viability_shortfall: Any | None = None,
    viability_components: dict[str, Any] | None = None,
    viability_term: str = "viability_penalty",
    component_prefix: str = "",
) -> tuple[Any, dict[str, Any]]:
    """Subtract adaptive Lagrange penalty: -λ · expected quadratic deficit."""

    out = dict(components)
    objective = base_objective
    if viability_shortfall is None:
        return objective, out
    cap = float(viability_params.lambda_max) if viability_params is not None else 100.0
    lam = _clip_lambda(float(viability_lambda), cap)
    weighted = -lam * viability_shortfall
    objective = objective + weighted
    out[viability_term] = weighted
    out[f"{component_prefix}viability_lambda"] = lam
    out[f"{component_prefix}viability_lambda_theta"] = lam
    for key, value in (viability_components or {}).items():
        if key == "viability_shortfall":
            out[f"{viability_term}_raw"] = value
        else:
            out[f"{component_prefix}{key}"] = value

    # Explicit recovery objective
    rec_shortfall = (viability_components or {}).get("viability_recovery_shortfall")
    recovery_term = (
        "global_viability_recovery_penalty"
        if component_prefix == "global_" or viability_term.startswith("global_")
        else "viability_recovery_penalty"
    )
    if viability_params is not None and float(viability_params.recovery_fraction) > 0.0 and rec_shortfall is not None:
        rec_weighted = -lam * rec_shortfall
        objective = objective + rec_weighted
        out[recovery_term] = rec_weighted
    else:
        out[recovery_term] = 0.0

    return objective, out
