"""
Batched Monte Carlo MPC: batches over n_samples in a single forward/backward for performance.
Single MPC implementation; MPCOptimizer protocol and helpers live here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F

from mm.mpc.l_param import (
    LParamConfig,
    ReservationSkewParams,
    SpreadStructureParams,
    action_param_len,
    reservation_r_offset_from_params,
    sigmoid_action_from_flat_raw,
    validate_l_param_config,
)
from mm.mpc.optimizer_config_keys import (
    CUMULATIVE_QTY_CAP_AT_SPREAD_MAX,
    CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX,
    RESERVATION_SKEW,
    reject_removed_venue_optimizer_keys,
)
from mm.mpc.objective_terms import (
    BookQualityParams,
    ObjectiveNormalizeParams,
    ViabilityParams,
    apply_ce_objective_normalization,
    book_quality_score,
    combine_soft_objective_terms,
    combine_viability_term,
    update_ce_ewma_scale,
    update_viability_lambda,
    viability_path_shortfall,
)
from mm.mpc.mpc_state import State
from mm.core.wallet import portfolio_wealth
from mm.mpc.torch_device import resolve_torch_device
from mm.mpc.state_tensor import (
    effective_depth_bid_tensor,
    effective_depth_ask_tensor,
)
from mm.mpc.reward_tensor import terminal_objective
from mm.mpc.optimizer_diagnostics import (
    EpochObjectiveMetrics,
    OptimizerRunDiagnostics,
    build_action_snapshot,
)
from mm.observability.logs import log_event

from mm.pricing import vwap_prices_batched


# -----------------------------------------------------------------------------
# MPCOptimizer protocol (controller and package surface)
# -----------------------------------------------------------------------------


@runtime_checkable
class MPCOptimizer(Protocol):
    """Given state, return optimal action. Estimators provided at construction."""

    def optimize(self, state: State) -> SigmoidMixtureAction:
        ...


# -----------------------------------------------------------------------------
# Helpers: bounds, quote-size projection, book cumulative
# -----------------------------------------------------------------------------


def _get_torch():
    return torch


@dataclass(frozen=True, slots=True)
class VenueOptimizerParams:
    """Venue-scoped optimizer, fill, and inventory-shaping parameters."""

    n_components: int | None = None
    return_risk_aversion_beta: float = 1.0
    objective_weight: float = 1.0
    cumulative_qty_floor_at_spread_max: float = 0.0
    cumulative_qty_cap_at_spread_max: float | None = None
    reservation_skew: ReservationSkewParams | None = None
    w_book: float = 0.0
    viability: ViabilityParams | None = None
    book_quality: BookQualityParams | None = None
    soft_fill_temp: float = 0.1
    soft_fill_activation: str = "sigmoid"
    soft_fill_queue_model: str = "back_of_queue"

    @classmethod
    def from_mapping(
        cls,
        payload: Any,
        *,
        defaults: "VenueOptimizerParams | None" = None,
    ) -> "VenueOptimizerParams":
        base = defaults or cls()
        data = dict(payload or {})
        reject_removed_venue_optimizer_keys(data, venue="optimizer")
        if data.get("spread_structure") is not None:
            raise ValueError(
                "optimizer spread_structure was removed; use "
                "book_quality.hard_structure.{action_max_spread, action_min_depth, action_k_epsilon}"
            )
        if "w_inv" in data or "inventory_zone" in data:
            raise ValueError(
                "optimizer venue keys w_inv and inventory_zone were removed; "
                "use viability {min_quote_qty, margin_sharpness, availability_target, "
                "shortfall_tolerance, lambda_init, lambda_learning_rate}"
            )
        return cls(
            n_components=(
                max(1, int(data.get("n_components")))
                if data.get("n_components") is not None
                else base.n_components
            ),
            return_risk_aversion_beta=max(
                1e-6,
                float(data.get("return_risk_aversion_beta", base.return_risk_aversion_beta)),
            ),
            objective_weight=max(0.0, float(data.get("objective_weight", base.objective_weight))),
            cumulative_qty_floor_at_spread_max=float(
                data.get(CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX, base.cumulative_qty_floor_at_spread_max)
            ),
            cumulative_qty_cap_at_spread_max=(
                float(data[CUMULATIVE_QTY_CAP_AT_SPREAD_MAX])
                if CUMULATIVE_QTY_CAP_AT_SPREAD_MAX in data
                else base.cumulative_qty_cap_at_spread_max
            ),
            reservation_skew=(
                ReservationSkewParams.from_mapping(data.get(RESERVATION_SKEW))
                if data.get(RESERVATION_SKEW) is not None
                else base.reservation_skew
            ),
            w_book=max(0.0, float(data.get("w_book", base.w_book))),
            viability=(
                ViabilityParams.from_mapping(data.get("viability"))
                if data.get("viability") is not None
                else base.viability
            ),
            book_quality=(
                BookQualityParams.from_mapping(data.get("book_quality"))
                if data.get("book_quality") is not None
                else base.book_quality
            ),
            soft_fill_temp=float(data.get("soft_fill_temp", base.soft_fill_temp)),
            soft_fill_activation=_normalize_soft_fill_activation(
                data.get("soft_fill_activation", base.soft_fill_activation)
            ),
            soft_fill_queue_model=_normalize_soft_fill_queue_model(
                data.get("soft_fill_queue_model", base.soft_fill_queue_model)
            ),
        )

    def validate(self, *, spread_max: float, n_components: int | None, venue: str) -> None:
        if float(self.cumulative_qty_floor_at_spread_max) < 0.0:
            raise ValueError(
                f"optimizer.{venue}.{CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX} must be >= 0"
            )
        l_param = self.l_param_config(
            spread_max=spread_max,
            n_components=n_components or self.n_components or 1,
        )
        validate_l_param_config(l_param)

    @property
    def spread_structure(self) -> SpreadStructureParams | None:
        if self.book_quality is None or self.book_quality.hard_structure is None:
            return None
        return SpreadStructureParams.from_book_quality_hard_structure(
            self.book_quality.hard_structure
        )

    def l_param_config(self, *, spread_max: float, n_components: int) -> LParamConfig:
        spread_fields = (
            self.spread_structure.to_l_param_fields()
            if self.spread_structure is not None
            else {
                "quote_spread_max": None,
                "quote_spread_min_depth": None,
                "quote_spread_k_epsilon": 0.01,
            }
        )
        return LParamConfig(
            quote_size_min=float(self.cumulative_qty_floor_at_spread_max),
            quote_size_max=(
                float(self.cumulative_qty_cap_at_spread_max)
                if self.cumulative_qty_cap_at_spread_max is not None
                else None
            ),
            spread_max=spread_max,
            n_components=n_components,
            reservation_r_max=(
                float(self.reservation_skew.max_abs_skew)
                if self.reservation_skew is not None
                else 0.0
            ),
            **spread_fields,
        )

    def as_config(self, *, spread_max: float, n_components: int) -> dict[str, Any]:
        del spread_max
        out: dict[str, Any] = {
            "n_components": int(self.n_components or n_components),
            "return_risk_aversion_beta": self.return_risk_aversion_beta,
            "objective_weight": self.objective_weight,
            CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX: self.cumulative_qty_floor_at_spread_max,
            CUMULATIVE_QTY_CAP_AT_SPREAD_MAX: self.cumulative_qty_cap_at_spread_max,
            "w_book": self.w_book,
            "soft_fill_temp": self.soft_fill_temp,
            "soft_fill_activation": self.soft_fill_activation,
            "soft_fill_queue_model": self.soft_fill_queue_model,
        }
        if self.reservation_skew is not None:
            out[RESERVATION_SKEW] = self.reservation_skew.as_mapping()
        if self.viability is not None:
            out["viability"] = self.viability.as_mapping()
        if self.book_quality is not None:
            out["book_quality"] = self.book_quality.as_mapping()
        return out


def _bounds_effective_to_raw(bounds: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Convert optimizer bounds from effective parameter space to raw tensor space.

    Mixture ``L`` slots are logits; ``k`` bounds are inverse-softplus mapped; ``x0`` and
    trailing per-side depth logits pass through unchanged.
    """
    out: list[tuple[float, float]] = []
    for i, (lo, hi) in enumerate(bounds):
        if i >= len(bounds) - 3:
            out.append((float(lo), float(hi)))
        elif i % 3 == 0:
            out.append((float(lo), float(hi)))
        elif i % 3 == 1:
            out.append((_inv_softplus_np(float(lo)), _inv_softplus_np(float(hi))))
        else:
            out.append((float(lo), float(hi)))
    return out


def _inv_softplus_np(y: float) -> float:
    """Inverse softplus for positive target y (numpy scalar)."""
    y_f = float(max(1e-12, y))
    if y_f > 50.0:
        return y_f
    return float(np.log(np.expm1(y_f)))


def _finite_action_flat_or_none(flat: Any, expected_len: int) -> np.ndarray | None:
    try:
        arr = np.array(flat, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < expected_len:
        return None
    arr = arr[:expected_len]
    if not np.isfinite(arr).all():
        return None
    return arr


def _torch_optimizer_state_is_finite(optimizer: Any, torch_mod: Any) -> bool:
    if optimizer is None:
        return False
    try:
        states = optimizer.state.values()
    except AttributeError:
        return False
    for state in states:
        if not isinstance(state, dict):
            continue
        for value in state.values():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                if not bool(torch_mod.isfinite(value).all().item()):
                    return False
    return True


def _initial_action_uniform_marginal_qrange(
    n_components: int,
    spread_max: float,
    l_param: LParamConfig,
) -> np.ndarray:
    """
    Build an initial flat action (raw optimizer params) using a simple linear-range
    sigmoid approximation. Depth logits start at zero (midpoint of configured range).
    """
    n = max(1, int(n_components))
    sm = max(1, int(round(float(spread_max))))
    q_ref = l_param.quote_size_midpoint()
    q_level = float(q_ref) / float(sm)
    x0_center = 0.5 * float(sm)
    k_base = max(1e-3, min(0.05, 2.0 / max(1.0, float(sm))))
    k_eff = np.full(n, k_base, dtype=np.float64)
    L_total = max(1e-6, 4.0 * q_level / max(k_base, 1e-12))
    L_eff = np.full(n, max(1e-6, L_total / float(n)), dtype=np.float64)
    L_raw = np.zeros(n, dtype=np.float64)
    k_raw = np.array([_inv_softplus_np(v) for v in k_eff], dtype=np.float64)

    if l_param.spread_structured:
        u_scale = 0.0
        u_init = 0.0
        x0_raw = np.zeros(n, dtype=np.float64)
        if n > 1:
            fracs = np.linspace(0.25, 0.85, n - 1)
            for j, frac in enumerate(fracs):
                frac = min(max(float(frac), 1e-4), 1.0 - 1e-4)
                x0_raw[j + 1] = float(np.log(frac / (1.0 - frac)))
        bid_part: list[float] = []
        ask_part: list[float] = []
        bid_part.extend([u_scale, float(k_raw[0]), u_init])
        for i in range(1, n):
            bid_part.extend([float(L_raw[i]), float(k_raw[i]), float(x0_raw[i])])
        ask_part.extend([0.0, float(k_raw[0]), u_init])
        for i in range(1, n):
            ask_part.extend([float(L_raw[i]), float(k_raw[i]), float(x0_raw[i])])
        core = np.array(bid_part + ask_part, dtype=np.float64)
    else:
        x0_raw = np.zeros(n, dtype=np.float64)
        bid_part = []
        ask_part = []
        for i in range(n):
            bid_part.extend([float(L_raw[i]), float(k_raw[i]), float(x0_raw[i])])
            ask_part.extend([float(L_raw[i]), float(k_raw[i]), float(x0_raw[i])])
        core = np.array(bid_part + ask_part, dtype=np.float64)

    return np.concatenate([core, np.array([0.0, 0.0, 0.0], dtype=np.float64)])


def _format_action_verbose(flat: np.ndarray, n_components: int) -> str:
    """Format first action (flat 6*n) for verbose logging."""
    n = n_components
    if len(flat) < 6 * n:
        return str(flat)
    bid_L = [flat[3 * i] for i in range(n)]
    bid_k = [flat[3 * i + 1] for i in range(n)]
    bid_x0 = [flat[3 * i + 2] for i in range(n)]
    ask_L = [flat[3 * n + 3 * i] for i in range(n)]
    ask_k = [flat[3 * n + 3 * i + 1] for i in range(n)]
    ask_x0 = [flat[3 * n + 3 * i + 2] for i in range(n)]
    return (
        f"bid L={[round(x, 2) for x in bid_L]} k={[round(x, 3) for x in bid_k]} x0={[round(x, 2) for x in bid_x0]} | "
        f"ask L={[round(x, 2) for x in ask_L]} k={[round(x, 3) for x in ask_k]} x0={[round(x, 2) for x in ask_x0]}"
    )


def _grid_levels(spread_max: float, spread_step: float) -> int:
    step = max(1e-12, float(spread_step))
    return max(2, int(math.ceil(float(spread_max) / step)))


def _spread_grid_tensor(
    torch_mod: Any,
    grid_levels: int,
    spread_step: float,
    device: Any,
    dtype: Any,
    *,
    spread_max: float,
) -> Any:
    """Upper bucket edges from spread_step through grid_levels * spread_step (domain [0, spread_max])."""
    grid = torch_mod.arange(1.0, float(grid_levels) + 1.0, dtype=dtype, device=device) * float(spread_step)
    return torch_mod.minimum(grid, torch_mod.tensor(float(spread_max), dtype=dtype, device=device))


def _book_cumulative_from_ob_rel(
    ob_rel: OrderbookRelative,
    grid_levels: int,
    spread_step: float,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    """Build cumulative qty tensors on quote spread buckets through each bucket upper edge."""
    torch_mod = _get_torch()
    bid_cum = torch_mod.zeros(grid_levels, dtype=dtype, device=device)
    ask_cum = torch_mod.zeros(grid_levels, dtype=dtype, device=device)
    step = max(1e-12, float(spread_step))
    b_sum = 0.0
    for (spread, qty) in sorted(ob_rel.bids, key=lambda x: x[0]):
        idx = _spread_bucket_index(float(spread), step, grid_levels)
        if 0 <= idx < grid_levels:
            b_sum += float(qty)
            bid_cum[idx] = b_sum
    for i in range(1, grid_levels):
        if bid_cum[i].item() == 0:
            bid_cum[i] = bid_cum[i - 1]
    a_sum = 0.0
    for (spread, qty) in sorted(ob_rel.asks, key=lambda x: x[0]):
        idx = _spread_bucket_index(float(spread), step, grid_levels)
        if 0 <= idx < grid_levels:
            a_sum += float(qty)
            ask_cum[idx] = a_sum
    for i in range(1, grid_levels):
        if ask_cum[i].item() == 0:
            ask_cum[i] = ask_cum[i - 1]
    return bid_cum, ask_cum


def _spread_bucket_index(spread: float, spread_step: float, grid_levels: int) -> int:
    if spread <= 0.0:
        return 0
    idx = int(math.ceil(spread / spread_step)) - 1
    return min(max(idx, 0), grid_levels - 1)


def _book_cumulative_from_arrays(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched marginal per-bucket qty -> cumulative depth along the spread grid."""
    return torch.cumsum(ob_bids, dim=1), torch.cumsum(ob_asks, dim=1)



def _mid_from_book_best_levels_batched(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    center_mid: torch.Tensor,
    spread_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Mid implied by post-flow best bid/ask from level volumes.
    center_mid sets the absolute price reference; spread_grid is quote-price distance from mid.
    """
    big = float(spread_grid.max().item()) + 1000.0
    mask_b = (ob_bids > 0).to(center_mid.dtype)
    mask_a = (ob_asks > 0).to(center_mid.dtype)
    spread_b = spread_grid.unsqueeze(0) * mask_b + (1.0 - mask_b) * big
    spread_a = spread_grid.unsqueeze(0) * mask_a + (1.0 - mask_a) * big
    best_bid_spread = spread_b.min(dim=1, keepdim=True)[0]
    best_ask_spread = spread_a.min(dim=1, keepdim=True)[0]
    best_bid = center_mid - best_bid_spread
    best_ask = center_mid + best_ask_spread
    # If one side is empty, keep center_mid for that side (same fallback as previous logic).
    no_bid = best_bid_spread >= big
    no_ask = best_ask_spread >= big
    best_bid = torch.where(no_bid, center_mid, best_bid)
    best_ask = torch.where(no_ask, center_mid, best_ask)
    return 0.5 * (best_bid + best_ask)


def _best_spread_from_book_batched(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    spread_grid: torch.Tensor,
) -> torch.Tensor:
    """Best bid/ask spread width from grid-aligned book levels."""
    big = float(spread_grid.max().item()) + 1000.0
    dtype = ob_bids.dtype
    device = ob_bids.device
    mask_b = (ob_bids > 0).to(dtype)
    mask_a = (ob_asks > 0).to(dtype)
    big_t = torch.tensor(big, dtype=dtype, device=device)
    spread_b = spread_grid.unsqueeze(0) * mask_b + (1.0 - mask_b) * big_t
    spread_a = spread_grid.unsqueeze(0) * mask_a + (1.0 - mask_a) * big_t
    best_bid_spread = spread_b.min(dim=1, keepdim=True)[0]
    best_ask_spread = spread_a.min(dim=1, keepdim=True)[0]
    spread = best_bid_spread + best_ask_spread
    return torch.where(spread >= 2.0 * big_t, torch.zeros_like(spread), spread)


def _fixed_depth_shape_features_batched(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    mid: torch.Tensor,
    spread_grid: torch.Tensor,
    *,
    target_depth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return spread and extra fixed-depth buy/sell shape beyond half-spread."""
    spread = _best_spread_from_book_batched(ob_bids, ob_asks, spread_grid)
    buy_ref, sell_ref = vwap_prices_batched(
        ob_bids,
        ob_asks,
        mid,
        spread_grid,
        target_depth=max(0.0, float(target_depth)),
    )
    half_spread = 0.5 * torch.clamp(spread, min=1e-9)
    buy_shape = torch.clamp(buy_ref - mid - half_spread, min=1e-9)
    sell_shape = torch.clamp(mid - sell_ref - half_spread, min=1e-9)
    return torch.clamp(spread, min=1e-9), buy_shape, sell_shape


_SOFT_FILL_ACTIVATIONS = frozenset({"sigmoid", "hardsigmoid"})
_SOFT_FILL_QUEUE_MODELS = frozenset({"pro_rata", "back_of_queue"})


def _normalize_soft_fill_activation(activation: str) -> str:
    kind = str(activation).lower().strip()
    if kind not in _SOFT_FILL_ACTIVATIONS:
        supported = ", ".join(sorted(_SOFT_FILL_ACTIVATIONS))
        raise ValueError(f"unsupported soft_fill_activation {activation!r}; expected one of: {supported}")
    return kind


def _normalize_soft_fill_queue_model(model: str) -> str:
    kind = str(model or "back_of_queue").lower().strip()
    if kind not in _SOFT_FILL_QUEUE_MODELS:
        supported = ", ".join(sorted(_SOFT_FILL_QUEUE_MODELS))
        raise ValueError(f"unsupported soft_fill_queue_model {model!r}; expected one of: {supported}")
    return kind


def _soft_fill_fraction(
    reach: torch.Tensor,
    soft_temp: float,
    activation: str = "sigmoid",
) -> torch.Tensor:
    """
    Map normalized reach to a per-level consumption fraction in [0, 1].

    ``soft_temp`` scales the input before the activation:
    ``scaled_reach = reach / soft_temp``.

    ``hardsigmoid`` is exactly zero when ``scaled_reach <= -3``, eliminating
    asymptotic ghost fills from far-from-mid quote levels.
    """
    scaled = reach / max(float(soft_temp), 1e-12)
    kind = _normalize_soft_fill_activation(activation)
    if kind == "hardsigmoid":
        return F.hardsigmoid(scaled)
    return torch.sigmoid(scaled)


def soft_fill_batched(
    params: torch.Tensor,
    n_components: int,
    mid_price: torch.Tensor,
    book_cum_bid: torch.Tensor,
    book_cum_ask: torch.Tensor,
    sell_vol: torch.Tensor,
    buy_vol: torch.Tensor,
    spread_grid: torch.Tensor,
    maker_fee: float,
    soft_temp: float = 1.0,
    soft_fill_activation: str = "sigmoid",
    soft_fill_queue_model: str = "back_of_queue",
    bid_quote_size_min: torch.Tensor | float | None = None,
    bid_quote_size_max: torch.Tensor | float | None = None,
    ask_quote_size_min: torch.Tensor | float | None = None,
    ask_quote_size_max: torch.Tensor | float | None = None,
    use_softplus: bool = True,
    l_param: LParamConfig | None = None,
    depth_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched soft transition using combined (book + quoted) depth:
    - soft-consume against combined cumulative book
    - split consumed volume between market book and our quotes proportionally per level
    - split queue by pro-rata or back-of-queue model
    - return next market-only orderbook levels plus cash/inventory deltas from our fills

    Inputs:
      book_cum_*: market-book cumulative depth on spread grid (n_samples, grid_levels)
      sell_vol/buy_vol: market order flow per sample (n_samples, 1)
      mid_price: clean-orderbook mid; quote prices are mid_price ± spread

    Returns:
      ob_bids_next, ob_asks_next: next market-only level volumes (n_samples, grid_levels)
      cash_delta, inv_delta: deltas from our fills only (n_samples,)
    """
    eps = 1e-12
    reservation_r_offset = None
    if l_param is not None and float(l_param.reservation_r_max) > 0.0:
        reservation_r_offset = reservation_r_offset_from_params(
            params,
            n_components=n_components,
            r_max=float(l_param.reservation_r_max),
        )
    depth_kwargs = {
        "use_softplus": use_softplus,
        "l_param": l_param,
        "quote_size_min": bid_quote_size_min,
        "quote_size_max": bid_quote_size_max,
        "mid_price": mid_price,
        "reservation_r_offset": reservation_r_offset,
    }
    depth_bid = effective_depth_bid_tensor(
        params,
        spread_grid,
        n_components,
        **depth_kwargs,
    )
    depth_ask = effective_depth_ask_tensor(
        params,
        spread_grid,
        n_components,
        use_softplus=use_softplus,
        l_param=l_param,
        quote_size_min=ask_quote_size_min,
        quote_size_max=ask_quote_size_max,
        mid_price=mid_price,
        reservation_r_offset=reservation_r_offset,
    )

    # Recover market-book level volumes from cumulative market book.
    book_bid_prev = torch.cat([torch.zeros_like(book_cum_bid[:, :1]), book_cum_bid[:, :-1]], dim=1)
    book_ask_prev = torch.cat([torch.zeros_like(book_cum_ask[:, :1]), book_cum_ask[:, :-1]], dim=1)
    book_bid_level = torch.clamp(book_cum_bid - book_bid_prev, min=0.0)
    book_ask_level = torch.clamp(book_cum_ask - book_ask_prev, min=0.0)

    # Convert cumulative quoted depth to marginal level sizes. Dynamic bid/ask
    # maxima are already inside the decoder above.
    depth_bid_sample = (
        depth_bid.unsqueeze(0).expand_as(book_cum_bid)
        if depth_bid.ndim == 1
        else depth_bid
    )
    depth_ask_sample = (
        depth_ask.unsqueeze(0).expand_as(book_cum_ask)
        if depth_ask.ndim == 1
        else depth_ask
    )
    depth_bid_prev = torch.cat(
        [torch.zeros_like(depth_bid_sample[:, :1]), depth_bid_sample[:, :-1]],
        dim=1,
    )
    depth_ask_prev = torch.cat(
        [torch.zeros_like(depth_ask_sample[:, :1]), depth_ask_sample[:, :-1]],
        dim=1,
    )
    marg_bid = torch.clamp(depth_bid_sample - depth_bid_prev, min=0.0)
    marg_ask = torch.clamp(depth_ask_sample - depth_ask_prev, min=0.0)

    # Combine market book and our quotes on each level.
    scale = max(0.0, float(depth_scale))
    our_bid_level = marg_bid * scale
    our_ask_level = marg_ask * scale
    comb_bid_level = book_bid_level + our_bid_level
    comb_ask_level = book_ask_level + our_ask_level

    # Soft consumption against combined cumulative depth.
    comb_cum_bid = torch.cumsum(comb_bid_level, dim=1)
    comb_cum_ask = torch.cumsum(comb_ask_level, dim=1)
    reach_bid_raw = sell_vol - comb_cum_bid
    reach_ask_raw = buy_vol - comb_cum_ask
    bid_pos_scale = torch.clamp(reach_bid_raw.max(dim=1, keepdim=True).values, min=eps)
    bid_neg_scale = torch.clamp((-reach_bid_raw).max(dim=1, keepdim=True).values, min=eps)
    ask_pos_scale = torch.clamp(reach_ask_raw.max(dim=1, keepdim=True).values, min=eps)
    ask_neg_scale = torch.clamp((-reach_ask_raw).max(dim=1, keepdim=True).values, min=eps)
    reach_bid = torch.where(
        reach_bid_raw >= 0.0,
        reach_bid_raw / bid_pos_scale,
        reach_bid_raw / bid_neg_scale,
    )
    reach_ask = torch.where(
        reach_ask_raw >= 0.0,
        reach_ask_raw / ask_pos_scale,
        reach_ask_raw / ask_neg_scale,
    )
    fill_frac_bid = _soft_fill_fraction(reach_bid, soft_temp, soft_fill_activation)
    fill_frac_ask = _soft_fill_fraction(reach_ask, soft_temp, soft_fill_activation)
    consumed_bid_level = comb_bid_level * fill_frac_bid
    consumed_ask_level = comb_ask_level * fill_frac_ask

    # Split consumed level volumes between market book and our quotes.
    # back_of_queue treats the next worse public level as ahead of our order,
    # which is a conservative differentiable queue approximation.
    queue_model = _normalize_soft_fill_queue_model(soft_fill_queue_model)
    if queue_model == "back_of_queue":
        bid_next_public = torch.cat(
            [book_bid_level[:, 1:], torch.zeros_like(book_bid_level[:, :1])],
            dim=1,
        )
        ask_next_public = torch.cat(
            [book_ask_level[:, 1:], torch.zeros_like(book_ask_level[:, :1])],
            dim=1,
        )
    else:
        bid_next_public = torch.zeros_like(book_bid_level)
        ask_next_public = torch.zeros_like(book_ask_level)
    bid_share_den = comb_bid_level + bid_next_public
    ask_share_den = comb_ask_level + ask_next_public
    bid_share_our = torch.where(
        bid_share_den > eps,
        our_bid_level / torch.clamp(bid_share_den, min=eps),
        torch.zeros_like(comb_bid_level),
    )
    ask_share_our = torch.where(
        ask_share_den > eps,
        our_ask_level / torch.clamp(ask_share_den, min=eps),
        torch.zeros_like(comb_ask_level),
    )
    fill_bid_level = consumed_bid_level * bid_share_our
    fill_ask_level = consumed_ask_level * ask_share_our
    fill_bid = fill_bid_level.sum(dim=1)
    fill_ask = fill_ask_level.sum(dim=1)

    # Market-book next state after market-flow consumption (our resting remainder is canceled).
    consumed_bid_market = torch.minimum(
        torch.clamp(consumed_bid_level - fill_bid_level, min=0.0),
        book_bid_level,
    )
    consumed_ask_market = torch.minimum(
        torch.clamp(consumed_ask_level - fill_ask_level, min=0.0),
        book_ask_level,
    )
    ob_bids_next = torch.clamp(book_bid_level - consumed_bid_market, min=0.0)
    ob_asks_next = torch.clamp(book_ask_level - consumed_ask_market, min=0.0)
    mid_exp = mid_price.expand(-1, spread_grid.shape[0])
    cash_bid = (fill_bid_level * (mid_exp - spread_grid.unsqueeze(0))).sum(dim=1)
    cash_ask = (fill_ask_level * (mid_exp + spread_grid.unsqueeze(0))).sum(dim=1)
    one_minus_f = 1.0 - maker_fee
    cash_delta = cash_ask * one_minus_f - cash_bid
    inv_delta = fill_bid * one_minus_f - fill_ask
    return ob_bids_next, ob_asks_next, cash_delta, inv_delta


def _delta_wealth_mtm_batched(
    buy_vwap: torch.Tensor,
    sell_vwap: torch.Tensor,
    inv: torch.Tensor,
    cash: torch.Tensor,
    buy_vwap_next: torch.Tensor,
    sell_vwap_next: torch.Tensor,
    inv_next: torch.Tensor,
    cash_next: torch.Tensor,
) -> torch.Tensor:
    """Per-sample ΔW with long inventory at bid VWAP and short inventory at ask VWAP."""
    mark = torch.where(inv < 0.0, buy_vwap, sell_vwap)
    mark_next = torch.where(inv_next < 0.0, buy_vwap_next, sell_vwap_next)
    W = cash + inv * mark
    W_next = cash_next + inv_next * mark_next
    return (W_next - W).squeeze(1)


def _quote_capacity_bounds_batched(
    *,
    cash: torch.Tensor,
    inventory: torch.Tensor,
    mid_price: torch.Tensor,
    optimizer: VenueOptimizerParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard side caps from wallet totals; side mins stay at zero (presence is soft)."""
    hard_cap_value = (
        float(optimizer.cumulative_qty_cap_at_spread_max)
        if optimizer.cumulative_qty_cap_at_spread_max is not None
        else float("inf")
    )
    hard_cap = torch.full_like(cash, max(0.0, hard_cap_value))
    cash_qty = torch.clamp(cash, min=0.0) / torch.clamp(mid_price, min=1e-12)
    inventory_qty = torch.clamp(inventory, min=0.0)
    bid_max = torch.minimum(hard_cap, cash_qty)
    ask_max = torch.minimum(hard_cap, inventory_qty)
    bid_min = torch.zeros_like(bid_max)
    ask_min = torch.zeros_like(ask_max)
    return bid_min, bid_max, ask_min, ask_max


def _quote_capacity_bounds_scalar(
    *,
    cash: float,
    inventory: float,
    mid_price: float,
    optimizer: VenueOptimizerParams,
) -> tuple[float, float, float, float]:
    hard_cap = (
        max(0.0, float(optimizer.cumulative_qty_cap_at_spread_max))
        if optimizer.cumulative_qty_cap_at_spread_max is not None
        else float("inf")
    )
    mid_safe = max(1e-12, float(mid_price))
    bid_max = min(hard_cap, max(0.0, float(cash)) / mid_safe)
    ask_max = min(hard_cap, max(0.0, float(inventory)))
    return 0.0, bid_max, 0.0, ask_max


class MonteCarloMPCTensorBatched:
    """
    Batched MC MPC: same API as MonteCarloMPCTensor but batches over n_samples
    in a single forward/backward per epoch for better performance.
    """

    def __init__(
        self,
        local_price_model: Any,
        local_market_flow_model: Any,
        execution_model: Any,
        n_components: int = 2,
        n_samples: int = 32,
        horizon: int = 4,
        spread_max: float = 100.0,
        spread_step: float = 1.0,
        soft_fill_temp: float = 0.1,
        soft_fill_activation: str = "sigmoid",
        soft_fill_queue_model: str = "back_of_queue",
        adam_lr: float = 0.05,
        adam_epochs: int = 80,
        adam_grad_clip: float = 10.0,
        device: str | None = None,
        bounds: list[tuple[float, float]] | None = None,
        warm_start: bool = False,
        fixed_mc_samples: bool = False,
        verbose: bool = False,
        return_risk_aversion_beta: float = 1.0,
        liquidity_sourcing_enabled: bool = False,
        global_market_flow_model: Any = None,
        global_price_model: Any = None,
        local_optimizer: dict[str, Any] | VenueOptimizerParams | None = None,
        global_optimizer: dict[str, Any] | VenueOptimizerParams | None = None,
        reference_price_source: str = "global",
        cumulative_qty_floor_at_spread_max: float | int | None = None,
        cumulative_qty_cap_at_spread_max: float | int | None = None,
        objective_normalize: dict[str, Any] | ObjectiveNormalizeParams | None = None,
    ):
        self.local_price_model = local_price_model
        self.local_market_flow_model = local_market_flow_model
        self.execution_model = execution_model
        self.n_components = max(1, n_components)
        self.n_samples = max(1, n_samples)
        self.horizon = max(1, horizon)
        self.spread_max = float(spread_max)
        self.spread_step = max(1e-12, float(spread_step))
        self.grid_levels = _grid_levels(self.spread_max, self.spread_step)
        self.adam_lr = adam_lr
        self.adam_epochs = adam_epochs
        self.adam_grad_clip = adam_grad_clip
        self.device = device
        self.warm_start = warm_start
        self.fixed_mc_samples = fixed_mc_samples
        self.verbose = verbose
        self.liquidity_sourcing_enabled = bool(liquidity_sourcing_enabled)
        self.global_market_flow_model = global_market_flow_model
        self.global_price_model = global_price_model
        self.reference_price_source = str(reference_price_source or "global").strip().lower()
        if self.reference_price_source not in ("global", "local"):
            raise ValueError("reference_price_source must be 'global' or 'local'")
        flat_defaults = VenueOptimizerParams(
            n_components=self.n_components,
            return_risk_aversion_beta=return_risk_aversion_beta,
            objective_weight=1.0,
            cumulative_qty_floor_at_spread_max=(
                float(cumulative_qty_floor_at_spread_max)
                if cumulative_qty_floor_at_spread_max is not None
                else 0.0
            ),
            cumulative_qty_cap_at_spread_max=(
                float(cumulative_qty_cap_at_spread_max)
                if cumulative_qty_cap_at_spread_max is not None
                else None
            ),
            soft_fill_temp=float(soft_fill_temp),
            soft_fill_activation=_normalize_soft_fill_activation(soft_fill_activation),
            soft_fill_queue_model=_normalize_soft_fill_queue_model(soft_fill_queue_model),
        )
        self.local_optimizer = (
            local_optimizer
            if isinstance(local_optimizer, VenueOptimizerParams)
            else VenueOptimizerParams.from_mapping(local_optimizer, defaults=flat_defaults)
        )
        self.global_optimizer = (
            global_optimizer
            if isinstance(global_optimizer, VenueOptimizerParams)
            else VenueOptimizerParams.from_mapping(global_optimizer, defaults=self.local_optimizer)
        )
        self.local_n_components = int(self.local_optimizer.n_components or self.n_components)
        self.global_n_components = int(self.global_optimizer.n_components or self.n_components)
        self.local_optimizer.validate(
            spread_max=self.spread_max,
            n_components=self.local_n_components,
            venue="local",
        )
        self.global_optimizer.validate(
            spread_max=self.spread_max,
            n_components=self.global_n_components,
            venue="global",
        )
        self.return_risk_aversion_beta = self.local_optimizer.return_risk_aversion_beta
        self.cumulative_qty_floor_at_spread_max = (
            self.local_optimizer.cumulative_qty_floor_at_spread_max
        )
        self.cumulative_qty_cap_at_spread_max = self.local_optimizer.cumulative_qty_cap_at_spread_max
        self.soft_fill_temp = self.local_optimizer.soft_fill_temp
        self.soft_fill_activation = self.local_optimizer.soft_fill_activation
        self.soft_fill_queue_model = self.local_optimizer.soft_fill_queue_model
        self._last_action: SigmoidMixtureAction | None = None
        self._last_raw_flat: np.ndarray | None = None
        self._last_J_components: dict | None = None
        self._last_J_components_initial: dict | None = None
        self._last_J_mean_raw: float = 0.0
        self._last_E_dW_next: float = 0.0
        self._last_main_reward_mean: float = 0.0
        self._last_global_reward_mean: float = 0.0
        self._last_main_reward_mean_initial: float = 0.0
        self._last_global_reward_mean_initial: float = 0.0
        self._last_sigmoid_action: SigmoidMixtureAction | None = None
        self._last_global_sigmoid_action: SigmoidMixtureAction | None = None
        self._last_joint_action: Any = None
        self._last_diagnostics: OptimizerRunDiagnostics | None = None
        self._x_torch: Any = None
        self._optimizer: Any = None
        if isinstance(objective_normalize, ObjectiveNormalizeParams):
            self.objective_normalize = objective_normalize
        else:
            self.objective_normalize = ObjectiveNormalizeParams.from_mapping(objective_normalize)
        self._ce_ewma_scale = 0.0
        local_viability = self.local_optimizer.viability
        global_viability = self.global_optimizer.viability
        self._viability_lambda_local = (
            float(local_viability.lambda_init) if local_viability is not None else 0.0
        )
        self._viability_lambda_global = (
            float(global_viability.lambda_init) if global_viability is not None else 0.0
        )

        if bounds is None:
            logit_bound = (-30.0, 30.0)
            k_bound = (1e-6, 100.0)
            x0_bound = (-self.spread_max, self.spread_max)
            local_bounds = [logit_bound, k_bound, x0_bound] * (2 * self.local_n_components) + [
                logit_bound,
                logit_bound,
                logit_bound,
            ]
            global_bounds = [logit_bound, k_bound, x0_bound] * (2 * self.global_n_components) + [
                logit_bound,
                logit_bound,
                logit_bound,
            ]
        else:
            local_bounds = list(bounds)
            global_bounds = list(bounds)
        self.local_bounds = local_bounds
        self.global_bounds = global_bounds
        self.bounds = self.local_bounds

    @property
    def last_diagnostics(self) -> OptimizerRunDiagnostics | None:
        return self._last_diagnostics

    def normalize_terminal_objective(
        self,
        objective: Any,
        components: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Apply configured CE EWMA normalization to the terminal return term."""

        normalized, out, _ = apply_ce_objective_normalization(
            objective,
            components,
            normalize=self.objective_normalize,
            ce_ewma_scale=self._ce_ewma_scale,
        )
        return normalized, out

    def _venue_optimizer(self, venue: str = "local") -> VenueOptimizerParams:
        return self.global_optimizer if str(venue).lower() == "global" else self.local_optimizer

    def _l_param_config(self, venue: str = "local") -> LParamConfig:
        n_components = (
            self.global_n_components if str(venue).lower() == "global" else self.local_n_components
        )
        return self._venue_optimizer(venue).l_param_config(
            spread_max=self.spread_max,
            n_components=n_components,
        )

    def _depth_call_kwargs(self, venue: str = "local") -> dict[str, Any]:
        return {"use_softplus": True, "l_param": self._l_param_config(venue)}

    def _flat_to_effective_action(
        self,
        flat: np.ndarray,
        *,
        mid_price: float | None = None,
        bid_quote_size_min: float | None = None,
        bid_quote_size_max: float | None = None,
        ask_quote_size_min: float | None = None,
        ask_quote_size_max: float | None = None,
    ) -> SigmoidMixtureAction:
        return sigmoid_action_from_flat_raw(
            flat,
            self.local_n_components,
            l_param=self._l_param_config(),
            mid_price=mid_price,
            bid_quote_size_min=bid_quote_size_min,
            bid_quote_size_max=bid_quote_size_max,
            ask_quote_size_min=ask_quote_size_min,
            ask_quote_size_max=ask_quote_size_max,
        )

    def optimize(self, state: State) -> SigmoidMixtureAction:
        action, _ = self.optimize_with_diagnostics(state)
        return action

    def optimize_with_diagnostics(self, state: State) -> tuple[SigmoidMixtureAction, OptimizerRunDiagnostics]:
        has_global_reference = (
            self.reference_price_source == "global"
            and state.global_sell_vwap is not None
            and float(state.global_sell_vwap) > 0.0
        )
        if self.liquidity_sourcing_enabled or has_global_reference:
            from mm.mpc.joint_optimizer import optimize_joint_with_diagnostics

            return optimize_joint_with_diagnostics(self, state)
        torch_mod = _get_torch()
        if isinstance(self.device, str) and self.device.lower() == "cpu":
            _device = torch_mod.device("cpu")
        elif self.device is None:
            _device = resolve_torch_device(None)
        else:
            _device = resolve_torch_device(str(self.device).strip() or None)
        dtype = torch_mod.float64

        book_cum_bid, book_cum_ask = _book_cumulative_from_ob_rel(
            state.ob_rel, self.grid_levels, self.spread_step, _device, dtype
        )
        spread_grid = _spread_grid_tensor(
            torch_mod,
            self.grid_levels,
            self.spread_step,
            _device,
            dtype,
            spread_max=self.spread_max,
        )
        mid_0 = float(state.mid)
        inv_0 = float(state.inventory)
        cash_0 = float(state.cash)
        buy_vwap_0 = float(state.buy_vwap)
        sell_vwap_0 = float(state.sell_vwap)
        mark_0 = buy_vwap_0 if inv_0 < 0.0 else sell_vwap_0 if inv_0 > 0.0 else mid_0
        wealth_0 = portfolio_wealth(cash_0, inv_0, mark_0)
        (
            bid_quote_size_min_0,
            bid_quote_size_max_0,
            ask_quote_size_min_0,
            ask_quote_size_max_0,
        ) = _quote_capacity_bounds_scalar(
            cash=cash_0,
            inventory=inv_0,
            mid_price=mid_0,
            optimizer=self.local_optimizer,
        )

        ob_bids_0 = torch_mod.zeros(self.n_samples, self.grid_levels, dtype=dtype, device=_device)
        ob_asks_0 = torch_mod.zeros(self.n_samples, self.grid_levels, dtype=dtype, device=_device)
        for (s, q) in state.ob_rel.bids:
            i = _spread_bucket_index(float(s), self.spread_step, self.grid_levels)
            if 0 <= i < self.grid_levels:
                ob_bids_0[:, i] += float(q)
        for (s, q) in state.ob_rel.asks:
            i = _spread_bucket_index(float(s), self.spread_step, self.grid_levels)
            if 0 <= i < self.grid_levels:
                ob_asks_0[:, i] += float(q)
        n = self.local_n_components
        single_len = action_param_len(n)
        param_len = self.horizon * single_len

        one_action = None
        if self.warm_start and self._last_raw_flat is not None:
            one_action = _finite_action_flat_or_none(self._last_raw_flat, single_len)
        if one_action is None and self.warm_start and self._last_action is not None:
            one_action = _finite_action_flat_or_none(self._last_action.to_flat(), single_len)
        if one_action is None:
            one_action = _initial_action_uniform_marginal_qrange(
                n_components=n,
                spread_max=self.spread_max,
                l_param=self._l_param_config(),
            )
        x0_arr = np.tile(one_action, self.horizon)

        bounds_raw = _bounds_effective_to_raw(self.bounds)
        bounds_full = bounds_raw * self.horizon
        lower = torch_mod.tensor([b[0] for b in bounds_full], dtype=dtype, device=_device)
        upper = torch_mod.tensor([b[1] for b in bounds_full], dtype=dtype, device=_device)

        if self._x_torch is None or self._x_torch.numel() != param_len:
            self._x_torch = torch_mod.tensor(x0_arr, dtype=dtype, device=_device, requires_grad=True)
            self._optimizer = torch_mod.optim.Adam([self._x_torch], lr=self.adam_lr)
        else:
            with torch_mod.no_grad():
                self._x_torch.data = torch_mod.tensor(x0_arr, dtype=dtype, device=_device)
            # If warm start is disabled, do not carry Adam moments across optimize()
            # calls; otherwise one-step updates can be dominated by stale momentum.
            if not self.warm_start:
                self._optimizer = torch_mod.optim.Adam([self._x_torch], lr=self.adam_lr)
        x_torch = self._x_torch
        optimizer = self._optimizer
        if not _torch_optimizer_state_is_finite(optimizer, torch_mod):
            self._optimizer = torch_mod.optim.Adam([x_torch], lr=self.adam_lr)
            optimizer = self._optimizer
        maker_fee = getattr(
            self.execution_model,
            "local_maker_fee",
            getattr(self.execution_model, "maker_fee", 0.0),
        )
        vwap_depth = max(0.0, float(getattr(state, "vwap_depth", 0.0)))

        # Pre-draw MC samples: (n_samples, horizon) for MOF and price shocks.
        def draw_samples():
            sample_shape = (self.n_samples, self.horizon)
            if hasattr(self.local_market_flow_model, "sample"):
                buy_vol, sell_vol = self.local_market_flow_model.sample(
                    sample_shape, device=_device, dtype=dtype
                )
            else:
                buy_vol = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
                sell_vol = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
                for s in range(self.n_samples):
                    for h in range(self.horizon):
                        bv, sv = self.local_market_flow_model.sample((1,), device=_device, dtype=dtype)
                        buy_vol[s, h] = bv.squeeze()
                        sell_vol[s, h] = sv.squeeze()
            price_innov = None
            if hasattr(self.local_price_model, "sample_innovations"):
                price_innov = self.local_price_model.sample_innovations(
                    sample_shape, device=_device, dtype=dtype
                )
                eps_buy = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
                eps_sell = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
            elif hasattr(self.local_price_model, "sample_residual"):
                eps_buy, eps_sell = self.local_price_model.sample_residual(
                    sample_shape, device=_device, dtype=dtype
                )
            else:
                eps_buy = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
                eps_sell = torch_mod.zeros(sample_shape, dtype=dtype, device=_device)
            return buy_vol, sell_vol, eps_buy, eps_sell, price_innov

        batch_time_sec = 0.0
        if self.fixed_mc_samples:
            t_batch = time.perf_counter()
            (
                fixed_buy_vol,
                fixed_sell_vol,
                fixed_eps_buy,
                fixed_eps_sell,
                fixed_price_innov,
            ) = draw_samples()
            batch_time_sec = time.perf_counter() - t_batch

        spread_levels = tuple(
            round(min(self.spread_max, i * self.spread_step), 12)
            for i in range(self.grid_levels + 1)
        )
        initial_flat = np.array(x_torch.detach().cpu().numpy()[:single_len], dtype=np.float64)
        initial_action = self._flat_to_effective_action(
            initial_flat,
            mid_price=mid_0,
            bid_quote_size_min=bid_quote_size_min_0,
            bid_quote_size_max=bid_quote_size_max_0,
            ask_quote_size_min=ask_quote_size_min_0,
            ask_quote_size_max=ask_quote_size_max_0,
        )
        action_initial_snap = build_action_snapshot(
            initial_action, mid=mid_0, spread_levels=spread_levels
        )

        best_loss = float("inf")
        best_x = x_torch.detach().clone()
        best_J_mean: float | None = None
        best_J_mean_next: float | None = None
        best_main_reward_mean: float | None = None
        best_epoch: int | None = None
        initial_metrics: EpochObjectiveMetrics | None = None
        final_metrics: EpochObjectiveMetrics | None = None
        best_metrics: EpochObjectiveMetrics | None = None
        self._last_J_components_initial = None
        captured_initial_components = False
        nonfinite_loss_count = 0
        nonfinite_grad_count = 0
        last_eval_x = best_x.detach().clone()
        t0 = time.perf_counter()

        for epoch in range(self.adam_epochs):
            if self.verbose:
                cur_flat = x_torch.detach().cpu().numpy()[:single_len]
                print(f"  Epoch {epoch}: action  {_format_action_verbose(cur_flat, n)}")

            def closure():
                nonlocal batch_time_sec, captured_initial_components
                nonlocal initial_metrics, final_metrics
                nonlocal nonfinite_loss_count, nonfinite_grad_count, last_eval_x
                optimizer.zero_grad()
                with torch_mod.no_grad():
                    x_torch.data = torch_mod.clamp(x_torch.data, lower, upper)
                    last_eval_x = x_torch.detach().clone()

                if self.fixed_mc_samples:
                    buy_vol, sell_vol, eps_buy, eps_sell, price_innov = (
                        fixed_buy_vol,
                        fixed_sell_vol,
                        fixed_eps_buy,
                        fixed_eps_sell,
                        fixed_price_innov,
                    )
                else:
                    t_batch = time.perf_counter()
                    buy_vol, sell_vol, eps_buy, eps_sell, price_innov = draw_samples()
                    batch_time_sec += time.perf_counter() - t_batch

                inv_t = torch_mod.full((self.n_samples, 1), inv_0, dtype=dtype, device=_device)
                cash_t = torch_mod.full((self.n_samples, 1), cash_0, dtype=dtype, device=_device)
                ob_bids = ob_bids_0.clone()
                ob_asks = ob_asks_0.clone()
                mid_fallback_t = torch_mod.full((self.n_samples, 1), mid_0, dtype=dtype, device=_device)

                book_cum_bid_0 = book_cum_bid.unsqueeze(0).expand(self.n_samples, -1)
                book_cum_ask_0 = book_cum_ask.unsqueeze(0).expand(self.n_samples, -1)

                total_reward = torch_mod.zeros(self.n_samples, dtype=dtype, device=_device)
                total_main_reward = torch_mod.zeros(self.n_samples, dtype=dtype, device=_device)
                step_reward_0 = torch_mod.zeros(self.n_samples, dtype=dtype, device=_device)
                inv_path: list[Any] = []
                cash_path: list[Any] = []
                mid_path: list[Any] = []
                conditional_price_model = hasattr(self.local_price_model, "sample_dynamic_vwap")
                conditional_buy_vwap_t = None
                conditional_sell_vwap_t = None

                for h in range(self.horizon):
                    if h == 0:
                        book_cum_bid_h = book_cum_bid_0
                        book_cum_ask_h = book_cum_ask_0
                    else:
                        book_cum_bid_h, book_cum_ask_h = _book_cumulative_from_arrays(ob_bids, ob_asks)

                    action_h = x_torch[h * single_len : (h + 1) * single_len]
                    depth_kw = self._depth_call_kwargs()
                    sell_h = sell_vol[:, h : h + 1]
                    buy_h = buy_vol[:, h : h + 1]
                    eps_buy_h = eps_buy[:, h : h + 1]
                    eps_sell_h = eps_sell[:, h : h + 1]
                    if h + 1 < self.horizon:
                        eps_buy_next = eps_buy[:, h + 1 : h + 2]
                        eps_sell_next = eps_sell[:, h + 1 : h + 2]
                    else:
                        eps_buy_next = eps_buy_h
                        eps_sell_next = eps_sell_h

                    mid_t = _mid_from_book_best_levels_batched(
                        ob_bids,
                        ob_asks,
                        mid_fallback_t,
                        spread_grid,
                    )
                    (
                        bid_quote_size_min_t,
                        bid_quote_size_max_t,
                        ask_quote_size_min_t,
                        ask_quote_size_max_t,
                    ) = _quote_capacity_bounds_batched(
                        cash=cash_t,
                        inventory=inv_t,
                        mid_price=mid_t,
                        optimizer=self.local_optimizer,
                    )
                    sell_vwap_depth_t = torch_mod.abs(inv_t)
                    buy_vwap_depth_t = torch_mod.clamp(cash_t / torch_mod.clamp(mid_t, min=1e-9), min=0.0)
                    if h == 0:
                        # Use upstream VWAPs computed from the initial inventory depth,
                        # instead of recomputing from the book arrays.
                        buy_vwap_book_t = torch_mod.full(
                            (self.n_samples, 1),
                            float(state.buy_vwap),
                            dtype=dtype,
                            device=_device,
                        )
                        sell_vwap_book_t = torch_mod.full(
                            (self.n_samples, 1),
                            float(state.sell_vwap),
                            dtype=dtype,
                            device=_device,
                        )
                    else:
                        if conditional_price_model and conditional_buy_vwap_t is not None:
                            buy_vwap_book_t = conditional_buy_vwap_t
                            sell_vwap_book_t = conditional_sell_vwap_t
                        else:
                            buy_vwap_book_t, sell_vwap_book_t = vwap_prices_batched(
                                ob_bids,
                                ob_asks,
                                mid_t,
                                spread_grid,
                                target_depth=sell_vwap_depth_t,
                            )
                    if conditional_price_model:
                        buy_vwap_t = buy_vwap_book_t
                        sell_vwap_t = sell_vwap_book_t
                    else:
                        buy_vwap_t = buy_vwap_book_t + eps_buy_h
                        sell_vwap_t = sell_vwap_book_t + eps_sell_h

                    ob_bids_next, ob_asks_next, cash_delta, inv_delta = soft_fill_batched(
                        action_h,
                        n,
                        mid_t,
                        book_cum_bid_h,
                        book_cum_ask_h,
                        sell_h,
                        buy_h,
                        spread_grid,
                        maker_fee,
                        soft_temp=self.soft_fill_temp,
                        soft_fill_activation=self.soft_fill_activation,
                        soft_fill_queue_model=self.soft_fill_queue_model,
                        bid_quote_size_min=bid_quote_size_min_t,
                        bid_quote_size_max=bid_quote_size_max_t,
                        ask_quote_size_min=ask_quote_size_min_t,
                        ask_quote_size_max=ask_quote_size_max_t,
                        **depth_kw,
                    )
                    inv_next = inv_t + inv_delta.unsqueeze(1)
                    cash_next = cash_t + cash_delta.unsqueeze(1)
                    mid_next = _mid_from_book_best_levels_batched(
                        ob_bids_next,
                        ob_asks_next,
                        mid_t,
                        spread_grid,
                    )
                    if conditional_price_model:
                        ref_depth = float(getattr(self.local_price_model, "reference_vwap_vol", 100.0))
                        prev_spread_ref, prev_buy_shape_ref, prev_sell_shape_ref = (
                            _fixed_depth_shape_features_batched(
                                ob_bids,
                                ob_asks,
                                mid_t,
                                spread_grid,
                                target_depth=ref_depth,
                            )
                        )
                        mech_spread_ref, mech_buy_shape_ref, mech_sell_shape_ref = (
                            _fixed_depth_shape_features_batched(
                                ob_bids_next,
                                ob_asks_next,
                                mid_next,
                                spread_grid,
                                target_depth=ref_depth,
                            )
                        )
                        innov_h = None
                        if price_innov is not None:
                            innov_h = {
                                "u_market": price_innov["u_market"][:, h : h + 1],
                                "z_market": {
                                    name: value[:, h : h + 1]
                                    for name, value in price_innov["z_market"].items()
                                },
                                "t_refresh": {
                                    name: value[:, h : h + 1]
                                    for name, value in price_innov["t_refresh"].items()
                                },
                            }
                        buy_vwap_next, sell_vwap_next = self.local_price_model.sample_dynamic_vwap(
                            prev_mid=mid_t,
                            prev_spread=prev_spread_ref,
                            prev_buy_shape=prev_buy_shape_ref,
                            prev_sell_shape=prev_sell_shape_ref,
                            mech_mid=mid_next,
                            mech_spread=mech_spread_ref,
                            mech_buy_shape=mech_buy_shape_ref,
                            mech_sell_shape=mech_sell_shape_ref,
                            total_flow=buy_h + sell_h,
                            buy_depth=torch_mod.clamp(cash_next / torch_mod.clamp(mid_next, min=1e-9), min=0.0),
                            sell_depth=torch_mod.abs(inv_next),
                            innovations=innov_h,
                        )
                        conditional_buy_vwap_t = buy_vwap_next
                        conditional_sell_vwap_t = sell_vwap_next
                    else:
                        buy_vwap_book_next, sell_vwap_book_next = vwap_prices_batched(
                            ob_bids_next,
                            ob_asks_next,
                            mid_next,
                            spread_grid,
                            target_depth=torch_mod.abs(inv_next),
                        )
                        buy_vwap_next = buy_vwap_book_next + eps_buy_next
                        sell_vwap_next = sell_vwap_book_next + eps_sell_next

                    ob_bids, ob_asks = ob_bids_next, ob_asks_next

                    step_reward_main = _delta_wealth_mtm_batched(
                        buy_vwap_t,
                        sell_vwap_t,
                        inv_t,
                        cash_t,
                        buy_vwap_next,
                        sell_vwap_next,
                        inv_next,
                        cash_next,
                    )
                    step_reward = step_reward_main
                    if h == 0:
                        step_reward_0 = step_reward
                    inv_path.append(inv_t.squeeze(1))
                    cash_path.append(cash_t.squeeze(1))
                    mid_path.append(mid_t.squeeze(1))
                    total_reward = total_reward + step_reward
                    total_main_reward = total_main_reward + step_reward_main

                    mid_fallback_t = mid_next
                    inv_t = inv_next
                    cash_t = cash_next

                returns = total_reward / wealth_0
                J_mean_raw = torch_mod.nan_to_num(
                    total_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                J_return_raw = torch_mod.nan_to_num(
                    returns.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                J_mean_next_raw = torch_mod.nan_to_num(
                    step_reward_0.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                main_reward_mean_raw = torch_mod.nan_to_num(
                    total_main_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30
                )
                global_reward_mean_raw = torch_mod.zeros((), dtype=dtype, device=_device)
                J, components = terminal_objective(
                    returns,
                    return_risk_aversion_beta=self.return_risk_aversion_beta,
                )
                J, components = self.normalize_terminal_objective(J, components)
                venue_opt = self.local_optimizer
                if float(venue_opt.w_book) > 0.0:
                    book_params = venue_opt.book_quality or BookQualityParams()
                    use_combined = book_params.measure == "combined"
                    book_score, book_components = book_quality_score(
                        x_torch[:single_len],
                        l_param=self._l_param_config(),
                        n_components=n,
                        spread_grid=spread_grid,
                        bid_quote_size_min=bid_quote_size_min_0,
                        bid_quote_size_max=bid_quote_size_max_0,
                        ask_quote_size_min=ask_quote_size_min_0,
                        ask_quote_size_max=ask_quote_size_max_0,
                        mid_price=mid_0,
                        params=book_params,
                        market_cum_bid=book_cum_bid if use_combined else None,
                        market_cum_ask=book_cum_ask if use_combined else None,
                    )
                    J, components = combine_soft_objective_terms(
                        J,
                        components,
                        w_book=venue_opt.w_book,
                        book_score=book_score,
                        book_components=book_components,
                    )
                if venue_opt.viability is not None:
                    inv_path_t = torch_mod.stack(inv_path, dim=1)
                    cash_path_t = torch_mod.stack(cash_path, dim=1)
                    mid_path_t = torch_mod.stack(mid_path, dim=1)
                    shortfall, viability_components = viability_path_shortfall(
                        inv_path_t,
                        cash_path=cash_path_t,
                        mid_path=mid_path_t,
                        params=venue_opt.viability,
                    )
                    J, components = combine_viability_term(
                        J,
                        components,
                        viability_lambda=self._viability_lambda_local,
                        viability_shortfall=shortfall,
                        viability_components=viability_components,
                    )

                def _finite_item(t) -> float:
                    try:
                        x = float(t.detach().item())
                        return x if math.isfinite(x) else 0.0
                    except Exception:
                        return 0.0

                obj_value = _finite_item(J)
                epoch_metrics = EpochObjectiveMetrics.from_tensors(
                    objective_value=obj_value,
                    components={k: _finite_item(v) for k, v in components.items()},
                    expected_total_reward=_finite_item(J_mean_raw),
                    expected_total_return=_finite_item(J_return_raw),
                    expected_next_reward=_finite_item(J_mean_next_raw),
                    main_reward_mean=_finite_item(main_reward_mean_raw),
                    global_reward_mean=_finite_item(global_reward_mean_raw),
                )
                self._last_J_components = dict(epoch_metrics.components)
                if not captured_initial_components:
                    self._last_J_components_initial = dict(epoch_metrics.components)
                    self._last_main_reward_mean_initial = epoch_metrics.main_reward_mean
                    self._last_global_reward_mean_initial = epoch_metrics.global_reward_mean
                    initial_metrics = epoch_metrics
                    captured_initial_components = True
                final_metrics = epoch_metrics
                self._last_J_mean_raw = epoch_metrics.expected_total_reward
                self._last_E_dW_next = epoch_metrics.expected_next_reward
                self._last_main_reward_mean = epoch_metrics.main_reward_mean
                self._last_global_reward_mean = epoch_metrics.global_reward_mean
                loss = -J
                if not torch_mod.isfinite(loss):
                    nonfinite_loss_count += 1
                if torch_mod.isfinite(loss):
                    loss.backward()
                    if x_torch.grad is not None:
                        if not torch_mod.isfinite(x_torch.grad).all():
                            nonfinite_grad_count += 1
                            x_torch.grad.detach().zero_()
                        else:
                            torch_mod.nn.utils.clip_grad_norm_([x_torch], self.adam_grad_clip)
                return loss

            loss = optimizer.step(closure)
            if loss is not None and torch_mod.isfinite(loss) and loss.item() < best_loss:
                best_loss = loss.item()
                best_x = last_eval_x.detach().clone()
                best_epoch = epoch
                if final_metrics is not None:
                    best_metrics = final_metrics
                best_J_mean = final_metrics.expected_total_reward if final_metrics else 0.0
                best_J_mean_next = final_metrics.expected_next_reward if final_metrics else 0.0
                best_main_reward_mean = final_metrics.main_reward_mean if final_metrics else 0.0

        loop_elapsed = time.perf_counter() - t0
        if self.fixed_mc_samples:
            train_time_sec = loop_elapsed
            total_sec = batch_time_sec + loop_elapsed
        else:
            train_time_sec = loop_elapsed - batch_time_sec
            total_sec = loop_elapsed
        best_flat = best_x.cpu().numpy()
        best_flat_slice = _finite_action_flat_or_none(best_flat, single_len)
        if best_flat_slice is None:
            best_flat_slice = _finite_action_flat_or_none(x0_arr, single_len)
        if best_flat_slice is None:
            best_flat_slice = _initial_action_uniform_marginal_qrange(
                n_components=n,
                spread_max=self.spread_max,
                l_param=self._l_param_config(),
            )
        best_action = self._flat_to_effective_action(
            best_flat_slice,
            mid_price=mid_0,
            bid_quote_size_min=bid_quote_size_min_0,
            bid_quote_size_max=bid_quote_size_max_0,
            ask_quote_size_min=ask_quote_size_min_0,
            ask_quote_size_max=ask_quote_size_max_0,
        )
        final_flat_slice = _finite_action_flat_or_none(x_torch.detach().cpu().numpy(), single_len)
        if final_flat_slice is None:
            final_flat_slice = best_flat_slice
        final_action = self._flat_to_effective_action(
            final_flat_slice,
            mid_price=mid_0,
            bid_quote_size_min=bid_quote_size_min_0,
            bid_quote_size_max=bid_quote_size_max_0,
            ask_quote_size_min=ask_quote_size_min_0,
            ask_quote_size_max=ask_quote_size_max_0,
        )
        self._last_sigmoid_action = best_action
        if best_J_mean is not None and math.isfinite(float(best_J_mean)):
            self._last_E_dW = float(best_J_mean)
        else:
            bl = -float(best_loss)
            self._last_E_dW = bl if math.isfinite(bl) else 0.0
        if best_J_mean_next is not None and math.isfinite(float(best_J_mean_next)):
            self._last_E_dW_next = float(best_J_mean_next)
        if best_main_reward_mean is not None and math.isfinite(float(best_main_reward_mean)):
            self._last_main_reward_mean = float(best_main_reward_mean)
        self._last_global_reward_mean = 0.0
        if self.warm_start:
            self._last_action = best_action
            self._last_raw_flat = np.array(best_flat_slice, dtype=np.float64)
        if best_metrics is None and final_metrics is not None:
            best_metrics = final_metrics
        if initial_metrics is None and final_metrics is not None:
            initial_metrics = final_metrics
        if best_metrics is not None:
            self._ce_ewma_scale = update_ce_ewma_scale(
                self._ce_ewma_scale,
                best_metrics.components,
                normalize=self.objective_normalize,
            )
            self._viability_lambda_local = update_viability_lambda(
                self._viability_lambda_local,
                best_metrics.components,
                params=self.local_optimizer.viability,
            )
        diagnostics = OptimizerRunDiagnostics(
            config={
                "kind": "monte_carlo_mpc_tensor_batched",
                "n_samples": self.n_samples,
                "horizon": self.horizon,
                "adam_epochs": self.adam_epochs,
                "fixed_mc_samples": self.fixed_mc_samples,
                "spread_max": self.spread_max,
                "spread_step": self.spread_step,
                "vwap_depth": vwap_depth,
                "reference_price_source": self.reference_price_source,
                "objective_normalize": self.objective_normalize.as_mapping(),
                "local_optimizer": self.local_optimizer.as_config(
                    spread_max=self.spread_max,
                    n_components=self.local_n_components,
                ),
                "global_optimizer": self.global_optimizer.as_config(
                    spread_max=self.spread_max,
                    n_components=self.global_n_components,
                ),
            },
            timing={
                "sample_draw_sec": float(batch_time_sec),
                "train_sec": float(train_time_sec),
                "total_sec": float(total_sec),
                "best_epoch": best_epoch,
            },
            initial=initial_metrics,
            final=final_metrics,
            best=best_metrics,
            best_epoch=best_epoch,
            action_initial=action_initial_snap,
            action_final=build_action_snapshot(final_action, mid=mid_0, spread_levels=spread_levels),
            action_best=build_action_snapshot(best_action, mid=mid_0, spread_levels=spread_levels),
            gradients={
                "nonfinite_loss_count": nonfinite_loss_count,
                "nonfinite_grad_count": nonfinite_grad_count,
            },
            rollout={
                "n_samples": self.n_samples,
                "horizon": self.horizon,
                "liquidity_sourcing": False,
                "soft_fill_temp": self.soft_fill_temp,
                "soft_fill_activation": self.soft_fill_activation,
                "soft_fill_queue_model": self.soft_fill_queue_model,
            },
        )
        self._last_diagnostics = diagnostics
        log_event(
            "mpc_optimizer_timing",
            {
                "batch_sec": batch_time_sec,
                "train_sec": train_time_sec,
                "total_sec": total_sec,
                "n_samples": self.n_samples,
                "horizon": self.horizon,
                "best_epoch": best_epoch,
            },
        )
        return best_action, diagnostics

    def reset_warm_start(self) -> None:
        self._last_action = None
        self._x_torch = None
        self._optimizer = None
