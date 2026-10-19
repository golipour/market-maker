"""
Batched Monte Carlo MPC: batches over n_samples in a single forward/backward for performance.
Single MPC implementation; MPCOptimizer protocol and helpers live here.
"""

from __future__ import annotations

import functools
import math
import os
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
    reservation_logit_index,
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
    ValidityParams,
    ViabilityParams,
    apply_ce_objective_normalization,
    book_quality_score,
    combine_validity_term,
    combine_viability_term,
    update_ce_ewma_scale,
    update_viability_lambda,
    stamp_viability_dual_diagnostics,
    viability_path_shortfall,
    viability_lambda_from_init,
)
from mm.mpc.warm_start import (
    WarmStartParams,
    prepare_warm_start_initial_flat,
    warm_action_from_mpc_state,
)
from mm.mpc.mpc_state import SigmoidMixtureAction, State
from mm.core.wallet import portfolio_wealth
from mm.mpc.decide_profiler import maybe_profile_decide
from mm.mpc.torch_device import apply_torch_threads, resolve_torch_device
from mm.mpc.state_tensor import (
    effective_depth_bid_tensor,
    effective_depth_ask_tensor,
    effective_depth_stacked_tensor,
)
from mm.mpc.reward_tensor import terminal_objective
from mm.mpc.optimizer_diagnostics import (
    EpochObjectiveMetrics,
    OptimizerRunDiagnostics,
    build_action_snapshot,
)
from mm.mpc.optimizer_timing import OptimizerStageTimer, stage_timing_payload
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
    validity: ValidityParams | None = None
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
                "use viability {min_viable_quote, shortfall_tolerance, "
                "lambda_init, lambda_learning_rate, lambda_max, "
                "dual_ewma_decay, recovery_fraction}"
            )
        if "w_book" in data:
            raise ValueError(
                "optimizer venue key w_book was removed; use validity "
                "{min_score, penalty_scale} with book_quality.soft_scoring"
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
            validity=(
                ValidityParams.from_mapping(data.get("validity"))
                if data.get("validity") is not None
                else base.validity
            ),
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
            "soft_fill_temp": self.soft_fill_temp,
            "soft_fill_activation": self.soft_fill_activation,
            "soft_fill_queue_model": self.soft_fill_queue_model,
        }
        if self.reservation_skew is not None:
            out[RESERVATION_SKEW] = self.reservation_skew.as_mapping()
        if self.validity is not None:
            out["validity"] = self.validity.as_mapping()
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


SUPPORTED_TENSOR_DTYPES: tuple[str, ...] = ("float64", "float32")


def _resolve_tensor_dtype(torch_mod: Any, name: str | None) -> Any:
    """Map a configured dtype name to a torch dtype.

    ``float64`` remains the default because it is the validated reference
    behavior. ``float32`` roughly halves memory bandwidth for the rollout,
    which is the dominant optimizer cost on CPU, but is only safe for live
    quoting after deterministic replay parity has been demonstrated.

    An unknown value is rejected rather than silently falling back, so a
    typo in configuration can never quietly change numerics.
    """
    key = str(name or "float64").strip().lower()
    if key in ("float64", "double", "f64"):
        return torch_mod.float64
    if key in ("float32", "float", "f32"):
        return torch_mod.float32
    raise ValueError(
        f"unsupported tensor_dtype {name!r}; expected one of {', '.join(SUPPORTED_TENSOR_DTYPES)}"
    )


def _epoch_needs_metrics(epoch: int, last_epoch: int, every: int) -> bool:
    """Return True when full per-epoch diagnostics should be materialized.

    Converting every objective component to a Python float each epoch was a
    measurable share of optimizer wall time (scalar extraction on a CPU tensor
    is a real Python/C boundary crossing). Only the first, last, and
    best-so-far epochs are needed for the published diagnostics, so the rest
    can be skipped.

    ``every <= 1`` restores the previous behavior of recording every epoch.
    """
    if every <= 1:
        return True
    if epoch == 0 or epoch == last_epoch:
        return True
    return epoch % every == 0


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
        if n > 1:
            fracs = np.linspace(0.20, 0.80, n)
            for j, frac in enumerate(fracs):
                frac = min(max(float(frac), 1e-4), 1.0 - 1e-4)
                x0_raw[j] = float(np.log(frac / (1.0 - frac)))
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


@functools.lru_cache(maxsize=8)
def _spread_grid_cached(
    grid_levels: int,
    spread_step: float,
    spread_max: float,
    device_str: str,
    dtype_name: str,
) -> Any:
    """Memoized ``_spread_grid_tensor`` for repeated decides.

    The spread grid is a pure function of optimizer configuration, which never
    changes across decides, yet it was rebuilt on every call. Caching removes
    one ``arange`` + ``tensor`` + ``minimum`` per decide from the setup path.

    The grid is treated as read-only by every consumer (it is only read,
    broadcast and sliced), so handing the same tensor out is safe.
    """
    import torch as _torch

    device = _torch.device(device_str)
    dtype = getattr(_torch, dtype_name)
    return _spread_grid_tensor(
        _torch, grid_levels, spread_step, device, dtype, spread_max=spread_max
    )


def _book_cumulative_from_ob_rel(
    ob_rel: OrderbookRelative,
    grid_levels: int,
    spread_step: float,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    """Build cumulative qty tensors on quote spread buckets through each bucket upper edge."""
    torch_mod = _get_torch()
    step = max(1e-12, float(spread_step))
    bid_cum = torch_mod.zeros(grid_levels, dtype=dtype, device=device)
    ask_cum = torch_mod.zeros(grid_levels, dtype=dtype, device=device)
    b_sum = 0.0
    for (spread, qty) in sorted(ob_rel.bids, key=lambda x: x[0]):
        idx = _spread_bucket_index(float(spread), step, grid_levels)
        if 0 <= idx < grid_levels:
            b_sum += float(qty)
            bid_cum[idx] = b_sum
    a_sum = 0.0
    for (spread, qty) in sorted(ob_rel.asks, key=lambda x: x[0]):
        idx = _spread_bucket_index(float(spread), step, grid_levels)
        if 0 <= idx < grid_levels:
            a_sum += float(qty)
            ask_cum[idx] = a_sum
    # Forward-fill empty buckets. The book is stored as a running total written
    # only at populated buckets, so an empty bucket inherits the previous level.
    # Only *exact* zeros are filled: a populated bucket can legitimately hold a
    # smaller cumulative value than its predecessor, and must not be raised.
    # (A ``cummax`` would clamp those too and is therefore NOT equivalent.)
    # Building the index list in Python and indexing once keeps the loop off
    # the tensor, removing one ``.item()`` host sync per level per side.
    if grid_levels > 1:
        for cum in (bid_cum, ask_cum):
            zero_idx = (cum == 0).nonzero().flatten().tolist()
            for i in zero_idx:
                if i == 0:
                    continue
                cum[i] = cum[i - 1]
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
    big_t = spread_grid[-1] + 1000.0
    # Select the spread (or the "absent" sentinel) directly instead of
    # blending two float tensors via a 0/1 mask. The arithmetic form
    # ``grid * mask + (1 - mask) * big`` spends four dispatches per side
    # (compare, cast, rsub, mul, add) where the equivalent ``where`` needs
    # one, and it is bit-exact: 1*x + 0 is x and 0*x + 1 is big with no
    # rounding. These helpers run a few hundred times per optimizer run, so
    # the saved dispatches are worth taking.
    spread_b = torch.where(ob_bids > 0, spread_grid.unsqueeze(0), big_t)
    spread_a = torch.where(ob_asks > 0, spread_grid.unsqueeze(0), big_t)
    best_bid_spread = spread_b.min(dim=1, keepdim=True)[0]
    best_ask_spread = spread_a.min(dim=1, keepdim=True)[0]
    best_bid = center_mid - best_bid_spread
    best_ask = center_mid + best_ask_spread
    # If one side is empty, keep center_mid for that side (same fallback as previous logic).
    no_bid = best_bid_spread >= big_t
    no_ask = best_ask_spread >= big_t
    best_bid = torch.where(no_bid, center_mid, best_bid)
    best_ask = torch.where(no_ask, center_mid, best_ask)
    return 0.5 * (best_bid + best_ask)


def _best_spread_from_book_batched(
    ob_bids: torch.Tensor,
    ob_asks: torch.Tensor,
    spread_grid: torch.Tensor,
) -> torch.Tensor:
    """Best bid/ask spread width from grid-aligned book levels."""
    big_t = spread_grid[-1] + 1000.0
    # See ``_mid_from_book_best_levels_batched``: a direct ``where`` replaces
    # the 0/1 mask blend bit-exactly at lower dispatch cost.
    spread_b = torch.where(ob_bids > 0, spread_grid.unsqueeze(0), big_t)
    spread_a = torch.where(ob_asks > 0, spread_grid.unsqueeze(0), big_t)
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
    depth_sample = effective_depth_stacked_tensor(
        params,
        spread_grid,
        n_components,
        use_softplus=use_softplus,
        l_param=l_param,
        bid_quote_size_min=bid_quote_size_min,
        bid_quote_size_max=bid_quote_size_max,
        ask_quote_size_min=ask_quote_size_min,
        ask_quote_size_max=ask_quote_size_max,
        mid_price=mid_price,
        reservation_r_offset=reservation_r_offset,
        spread_max=spread_grid[-1],
    )
    if depth_sample.ndim == 2:
        depth_sample = depth_sample.unsqueeze(1).expand(-1, book_cum_bid.shape[0], -1)
    book_cum = torch.stack([book_cum_bid, book_cum_ask], dim=0)
    vol = torch.stack([sell_vol, buy_vol], dim=0)

    # ``F.pad`` shifts and zero-fills in one op; the equivalent
    # ``torch.cat([zeros, x[:, :-1]])`` allocates a zero tensor and runs a
    # two-input concat, and its backward carries the SliceBackward nodes.
    book_prev = F.pad(book_cum[:, :, :-1], (1, 0))
    book_level = torch.relu(book_cum - book_prev)

    # Convert cumulative quoted depth to marginal level sizes. Dynamic bid/ask
    # maxima are already inside the decoder above.
    depth_prev = F.pad(depth_sample[:, :, :-1], (1, 0))
    marg = torch.relu(depth_sample - depth_prev)

    # Combine market book and our quotes on each level.
    scale = max(0.0, float(depth_scale))
    our_level = marg * scale
    comb_level = book_level + our_level

    # Exact flow-conserving prefix consumption.
    comb_cum = torch.cumsum(comb_level, dim=2)
    cons_cum = torch.minimum(vol, comb_cum)
    cons_prev = F.pad(cons_cum[:, :, :-1], (1, 0))
    consumed_level = torch.relu(cons_cum - cons_prev)

    # Split consumed volume between market book and our quotes.
    queue_model = _normalize_soft_fill_queue_model(soft_fill_queue_model)
    if queue_model == "back_of_queue":
        consumed_market = torch.minimum(consumed_level, book_level)
        fill_level = consumed_level - consumed_market
    else:  # pro_rata
        share_den = torch.clamp(comb_level, min=eps)
        share_our = torch.where(
            comb_level > eps,
            our_level / share_den,
            torch.zeros_like(comb_level),
        )
        fill_level = consumed_level * share_our
        consumed_market = consumed_level - fill_level

    fill = fill_level.sum(dim=2)

    # Market-book next state after market-flow consumption (our resting remainder is canceled).
    ob_next = torch.relu(book_level - consumed_market)
    ob_bids_next, ob_asks_next = ob_next[0], ob_next[1]
    fill_bid_level, fill_ask_level = fill_level[0], fill_level[1]
    fill_bid, fill_ask = fill[0], fill[1]
    mid_exp = mid_price.expand(-1, spread_grid.shape[0])
    cash_bid = (fill_bid_level * (mid_exp - spread_grid.unsqueeze(0))).sum(dim=1)
    cash_ask = (fill_ask_level * (mid_exp + spread_grid.unsqueeze(0))).sum(dim=1)
    one_minus_f = 1.0 - maker_fee
    cash_delta = cash_ask * one_minus_f - cash_bid
    inv_delta = fill_bid * one_minus_f - fill_ask
    return ob_bids_next, ob_asks_next, cash_delta, inv_delta


def _mark_batch_dims_dynamic(*tensors: Any) -> None:
    """Tell dynamo the leading batch axis is dynamic. Call *before* ``fn(...)``.

    ``soft_fill_batched`` accumulated one specialization per batch length. The
    live pod logged ``tensor 'sell_vol' size mismatch`` six times and
    ``tensor 'params' size mismatch`` five times against a
    ``recompile_limit`` of 8 -- and this worker is ``mode: "multi"``, running
    ``mpc-usdt-local`` (n_samples 64) and ``bitwana-sample`` (n_samples 16) in
    one process, so those are permanent axes, not warm-up noise.

    ``mark_dynamic`` replaces the concrete-size guards with symbolic ones, so
    one kernel serves every sample count. That is the alternative to enlarging
    ``recompile_limit``, which would not add capacity but delay the same
    permanent eager fallback.

    Two constraints, both learned the hard way:

    * It must be called on the *inputs* before the compiled region is entered.
      Calling it inside the traced function raises
      ``Attempt to trace forbidden callable <function mark_dynamic>`` and takes
      the whole kernel down to eager -- silently, because the warm-up call is
      wrapped in a fallback handler.
    * Only the batch axis varies. The grid and component axes are fixed by the
      preset, and marking them dynamic would block constant folding and cost
      kernel performance for nothing.
    """
    try:
        import torch as _torch

        for tensor in tensors:
            if tensor is None or not _torch.is_tensor(tensor):
                continue
            if tensor.is_floating_point() and tensor.ndim >= 1:
                _torch._dynamo.mark_dynamic(tensor, 0)
    except Exception:
        # Best-effort only: a failure here must never take a decision down.
        return


_COMPILED_SOFT_FILL_KERNEL: Any = None
_COMPILE_SOFT_FILL_FAILED: bool = False
#: Signatures whose compiled kernel has already been executed once, keyed by
#: the shapes *and* the branch-selecting scalar/string arguments. Inductor
#: specializes generated code on tensor shapes and on which Python branches are
#: taken, so ``pro_rata`` vs ``back_of_queue`` and ``hardsigmoid`` vs
#: ``sigmoid`` each need their own warm pass. The local and global venues
#: normally differ in both component count (4 and 2 by default) and fill
#: semantics, so both are warmed.
_SOFT_FILL_WARMED_SIGNATURES: set[tuple[Any, ...]] = set()

#: Signatures a decide asked for that construction never warmed. Logged once per
#: signature so a venue whose shape was never paid for at boot shows up in the
#: logs instead of silently running eager (or, worse, compiling on the decide
#: path and blowing the decision timeout).
_SOFT_FILL_UNWARMED_SEEN: set[tuple[Any, ...]] = set()

#: Maximum warm-up passes per signature. Bounded so a pathological graph
#: cannot stall process boot; the steady-state check normally exits after 3-4.
_SOFT_FILL_WARMUP_MAX_PASSES: int = 6

#: A pass this cheap means the compiled kernel is running, not compiling.
_SOFT_FILL_WARMUP_STABLE_MS: float = 25.0


def _get_compiled_soft_fill(torch_mod: Any = torch) -> Any:
    """Return a compiled version of soft_fill_batched with safe fallback on failure."""
    global _COMPILED_SOFT_FILL_KERNEL, _COMPILE_SOFT_FILL_FAILED
    if _COMPILE_SOFT_FILL_FAILED:
        return soft_fill_batched
    if _COMPILED_SOFT_FILL_KERNEL is not None:
        return _COMPILED_SOFT_FILL_KERNEL
    try:
        _COMPILED_SOFT_FILL_KERNEL = torch_mod.compile(soft_fill_batched, fullgraph=False)
        return _COMPILED_SOFT_FILL_KERNEL
    except Exception as exc:
        _COMPILE_SOFT_FILL_FAILED = True
        log_event(
            "mpc.torch_compile.fallback",
            {"reason": str(exc)},
        )
        return soft_fill_batched


def _soft_fill_signature(
    *,
    n_components: int,
    n_samples: int,
    grid_levels: int,
    soft_fill_activation: Any,
    soft_fill_queue_model: Any,
    depth_scale: Any = 1.0,
    soft_temp: Any = 1.0,
    maker_fee: Any = 0.0,
) -> tuple[Any, ...]:
    """Key identifying one inductor specialization of ``soft_fill_batched``.

    Inductor specializes generated code on tensor shapes *and* on which Python
    branches are taken, so a signature must cover both. Warm-up and the decide
    path must derive the key through this one function; when they computed it
    separately they drifted, and a live decide paid a 162 s lazy compile that
    blew ``decision_timeout_ms`` and wedged every controller.

    ``depth_scale`` is part of the key because it is a Python ``float`` that
    reaches the kernel as ``marg * scale``, and dynamo guards on Python scalars.
    Measured: compiling at ``depth_scale=1.0`` then calling at ``0.0`` added a
    frame and cost a 2.7 s lazy compile on the first masked call. The joint
    path passes ``depth_scale=1.0`` for the local chain and
    ``depth_scale=global_mask`` (0.0 or 1.0) for the global chain, so omitting
    it would let the global chain compile inside a live decide -- precisely the
    failure this signature set exists to prevent.

    ``soft_temp`` and ``maker_fee`` are here for the same reason. Both are
    Python floats that reach the kernel as ordinary arithmetic
    (``reach / soft_temp``, fee-scaled margins) with no tensor indirection, so
    dynamo guards on their exact values. They are constant per venue today,
    which is why the omission never showed up as a live failure -- but a
    per-venue override in the UI changes the float and would recompile inside
    a decide. Keying on them turns that stall into a warm-up miss, which
    ``resolve_compiled_soft_fill`` already handles by staying eager.
    """
    return (
        int(n_components),
        max(int(n_samples), 1),
        max(int(grid_levels), 1),
        str(soft_fill_activation),
        str(soft_fill_queue_model),
        float(depth_scale),
        float(soft_temp),
        float(maker_fee),
    )


def warm_soft_fill_compile(
    *,
    n_components: int,
    n_samples: int,
    grid_levels: int,
    l_param: Any = None,
    depth_kw: dict[str, Any] | None = None,
    maker_fee: float = 0.0,
    soft_temp: float = 1.0,
    soft_fill_activation: str = "sigmoid",
    soft_fill_queue_model: str = "back_of_queue",
    device: Any = None,
    dtype: Any = None,
    depth_scale: float = 1.0,
    torch_mod: Any = torch,
) -> float:
    """Pay the ``torch.compile`` cost for ``soft_fill_batched`` ahead of the first decide.

    Inductor compilation is lazy: the first call in a process triggers codegen
    and C++ compilation, which measured 3.4-4.7 s on the production image. The
    controller enforces ``decision_timeout_ms`` on every decide (10 s in the
    shipped preset), so paying that cost inside the first decide consumes a
    large fraction of the budget for no benefit. Running the kernel once with
    production shapes at construction moves the cost to process boot, where it
    is not subject to the decide watchdog.

    Best-effort: any failure is logged and the caller keeps the eager path,
    which is always correct. Returns the elapsed milliseconds so the caller can
    log the real cost.
    """
    global _COMPILE_SOFT_FILL_FAILED

    if _COMPILE_SOFT_FILL_FAILED:
        return 0.0
    n = int(n_components)
    ns = max(int(n_samples), 1)
    gl = max(int(grid_levels), 1)
    signature = _soft_fill_signature(
        n_components=n,
        n_samples=ns,
        grid_levels=gl,
        soft_fill_activation=soft_fill_activation,
        soft_fill_queue_model=soft_fill_queue_model,
        depth_scale=depth_scale,
        soft_temp=soft_temp,
        maker_fee=maker_fee,
    )
    if signature in _SOFT_FILL_WARMED_SIGNATURES:
        return 0.0
    started = time.perf_counter()
    try:
        torch = torch_mod
        if l_param is None:
            from mm.mpc.l_param import LParamConfig

            l_param = LParamConfig(
                quote_size_min=1.0,
                quote_size_max=100.0,
                spread_max=max(float(gl), 1.0),
                n_components=n,
            )
        dtype = dtype or torch.float64
        depth_kw = depth_kw or {"use_softplus": True}
        dev = device
        params = torch.zeros(3 * 2 * n + 3, dtype=dtype, device=dev, requires_grad=True)
        mid = torch.ones((ns, 1), dtype=dtype, device=dev)
        book = torch.zeros((ns, gl), dtype=dtype, device=dev)
        vols = torch.ones((ns, 1), dtype=dtype, device=dev)
        grid = torch.arange(1, gl + 1, dtype=dtype, device=dev)
        # ``book``/``vols``/``size_*`` carry requires_grad=False here, matching
        # the decide path: the book cumulative is derived from the observed
        # orderbook and the quote-capacity bounds from cash/inventory, neither
        # of which is a function of the optimized parameters. Only ``params``
        # requires grad. Keeping these flags aligned with the live call means
        # dynamo does not discover a requires_grad mismatch on the first decide
        # and pay a lazy compile for it. (Measured: the live pod logged
        # ``requires_grad mismatch`` against ``book_cum_ask``.)
        # The quote-size bounds must be per-sample tensors of shape (n_samples, 1)
        # to match what ``_quote_capacity_bounds_batched`` produces at runtime.
        # Passing floats instead would guard a different branch and trigger a
        # second lazy compile on the first real decide.
        size_min = torch.full((ns, 1), 0.0, dtype=dtype, device=dev)
        size_max = torch.full((ns, 1), 1.0, dtype=dtype, device=dev)

        fn = _get_compiled_soft_fill(torch)
        if fn is soft_fill_batched:
            # Compilation is unavailable on this platform; nothing to warm.
            _SOFT_FILL_WARMED_SIGNATURES.add(signature)
            return 0.0

        # Must precede the traced call; inside it, dynamo refuses to trace
        # ``mark_dynamic`` and silently degrades the kernel to eager.
        _mark_batch_dims_dynamic(mid, book, vols, size_min, size_max)

        # Repeat the call until the measured cost stops falling. Dynamo guards
        # on tensor identity as well as shape, so freshly allocated inputs
        # force a re-trace on the first couple of calls even though the graph
        # is unchanged. A single warm-up call therefore does *not* clear the
        # lazy-compile cost from the first real decide; iterating to steady
        # state does. Measured: 1402 ms -> 1154 ms -> 0.5 ms -> 0.2 ms.
        previous_ms: float | None = None
        for _ in range(_SOFT_FILL_WARMUP_MAX_PASSES):
            pass_started = time.perf_counter()
            ob_b, ob_a, cash, inv = fn(
                params,
                n,
                mid,
                book,
                book,
                vols,
                vols,
                grid,
                float(maker_fee),
                soft_temp=float(soft_temp),
                soft_fill_activation=soft_fill_activation,
                soft_fill_queue_model=soft_fill_queue_model,
                bid_quote_size_min=size_min,
                bid_quote_size_max=size_max,
                ask_quote_size_min=size_min,
                ask_quote_size_max=size_max,
                depth_scale=float(depth_scale),
                **depth_kw,
            )
            # Touch the backward too: the rollout runs backward on the first
            # decide as well, and the backward graph compiles separately.
            total = ob_b.sum() + ob_a.sum() + cash.sum() + inv.sum()
            total.backward()
            params.grad = None
            pass_ms = (time.perf_counter() - pass_started) * 1000.0
            if previous_ms is not None and pass_ms < _SOFT_FILL_WARMUP_STABLE_MS:
                break
            previous_ms = pass_ms
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        _COMPILE_SOFT_FILL_FAILED = True
        log_event(
            "mpc.torch_compile.warmup_failed",
            {"reason": str(exc)},
        )
        return 0.0
    _SOFT_FILL_WARMED_SIGNATURES.add(signature)
    log_event(
        "mpc.torch_compile.warmed",
        {
            "compile_ms": round(elapsed_ms, 1),
            "n_components": n,
            "n_samples": ns,
            "grid_levels": gl,
            "depth_scale": float(depth_scale),
        },
    )
    return elapsed_ms


def resolve_compiled_soft_fill(
    torch_mod: Any,
    *,
    enabled: bool,
    n_components: int,
    n_samples: int,
    grid_levels: int,
    soft_fill_activation: Any,
    soft_fill_queue_model: Any,
    depth_scale: Any = 1.0,
    soft_temp: Any = 1.0,
    maker_fee: Any = 0.0,
) -> Any:
    """Return the compiled ``soft_fill_batched`` only for a warmed signature.

    Shared by the single-venue rollout in this module and the joint rollout in
    ``mm.mpc.joint_optimizer``, so both derive the decision from one place. When
    the two derived it separately they drifted, and a live decide paid a lazy
    compile (measured 162 s once, 2.7 s for the masked global chain) that blew
    ``decision_timeout_ms``.

    An unwarmed signature returns eager ``soft_fill_batched`` rather than
    compiling: eager is always correct, just slower, and compiling inside a
    decide is the failure mode being guarded against. The shortfall is logged
    once per signature so a venue whose shape was never paid for at boot is
    visible in logs instead of silently costing decisions.
    """
    if not enabled:
        return soft_fill_batched
    signature = _soft_fill_signature(
        n_components=n_components,
        n_samples=n_samples,
        grid_levels=grid_levels,
        soft_fill_activation=soft_fill_activation,
        soft_fill_queue_model=soft_fill_queue_model,
        depth_scale=depth_scale,
        soft_temp=soft_temp,
        maker_fee=maker_fee,
    )
    if signature in _SOFT_FILL_WARMED_SIGNATURES:
        return _get_compiled_soft_fill(torch_mod)
    if signature not in _SOFT_FILL_UNWARMED_SEEN:
        _SOFT_FILL_UNWARMED_SEEN.add(signature)
        log_event(
            "mpc.torch_compile.unwarmed_shape",
            {
                "n_components": int(n_components),
                "n_samples": int(n_samples),
                "grid_levels": int(grid_levels),
                "soft_fill_activation": str(soft_fill_activation),
                "soft_fill_queue_model": str(soft_fill_queue_model),
                "depth_scale": float(depth_scale),
                "soft_temp": float(soft_temp),
                "maker_fee": float(maker_fee),
            },
        )
    return soft_fill_batched


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
    cash_qty = torch.relu(cash) / torch.clamp(mid_price, min=1e-12)
    inventory_qty = torch.relu(inventory)
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
        warm_start: bool | dict[str, Any] | WarmStartParams | None = False,
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
        optimizer_diagnostics_every: int = 1,
        tensor_dtype: str = "float64",
        torch_threads: int | str | None = None,
        compile_soft_fill: bool | None = None,
        is_bps: bool = False,
        spread_max_bps: float | None = None,
        spread_step_bps: float | None = None,
        crn_enabled: bool = False,
        crn_seed: int = 42,
        warmup_gate_enabled: bool = False,
        optimizer_mode: str = "adam",
        flow_scenario_mode: str = "weighted_stratified",
        stratified_positive_nodes: int = 3,
        trust_region_iterations: int = 5,
        trust_region_radius_ticks: float = 2.0,
        trust_region_min_radius_ticks: float = 1.0,
        trust_region_size_step: float = 1.0,
        trust_region_r_step: float = 0.25,
    ):
        self.local_price_model = local_price_model
        self.local_market_flow_model = local_market_flow_model
        self.execution_model = execution_model
        self.n_components = max(1, n_components)
        self.n_samples = max(1, n_samples)
        self.horizon = max(1, horizon)
        self.crn_enabled = bool(crn_enabled)
        self.crn_seed = int(crn_seed)
        self.warmup_gate_enabled = bool(warmup_gate_enabled)
        self.optimizer_mode = str(optimizer_mode or "adam").strip().lower()
        self.flow_scenario_mode = str(flow_scenario_mode or "weighted_stratified").strip().lower()
        self.stratified_positive_nodes = max(1, int(stratified_positive_nodes))
        self.trust_region_iterations = max(1, int(trust_region_iterations))
        self.trust_region_radius_ticks = float(trust_region_radius_ticks)
        self.trust_region_min_radius_ticks = float(trust_region_min_radius_ticks)
        self.trust_region_size_step = float(trust_region_size_step)
        self.trust_region_r_step = float(trust_region_r_step)
        self.is_bps = bool(is_bps) or (spread_max_bps is not None)
        self.spread_max_bps = (
            float(spread_max_bps)
            if spread_max_bps is not None
            else (float(spread_max) if self.is_bps else None)
        )
        self.spread_step_bps = (
            float(spread_step_bps)
            if spread_step_bps is not None
            else (float(spread_step) if self.is_bps else None)
        )
        self.spread_max = float(spread_max)
        self.spread_step = max(1e-12, float(spread_step))
        self.grid_levels = _grid_levels(self.spread_max, self.spread_step)
        self.adam_lr = adam_lr
        self.adam_epochs = adam_epochs
        self.adam_grad_clip = adam_grad_clip
        # How often to materialize full per-epoch objective diagnostics.
        # 1 keeps the historical every-epoch behavior; higher values record
        # only the first, last, sampled, and best epochs. The best epoch is
        # always captured because it drives the published ``best`` snapshot.
        self.optimizer_diagnostics_every = max(1, int(optimizer_diagnostics_every))
        # Validated eagerly so a bad value fails at construction rather than
        # on the first live decision.
        self.tensor_dtype = str(tensor_dtype or "float64").strip().lower()
        _resolve_tensor_dtype(torch, self.tensor_dtype)
        self.torch_threads = apply_torch_threads(torch_threads)
        self.device = device
        # ``soft_fill_batched`` is the single hottest kernel in the decide path
        # and is dispatch-bound on CPU, not FLOP-bound. Compiling it fuses the
        # several hundred small aten ops (and their autograd nodes) into a few
        # generated kernels. Measured at the shipped preset
        # (n_samples=64, horizon=2, 20 epochs, n_components=4, float64, one
        # thread, 10 real Nobitex states):
        #
        #   eager    28.74 ms/decide
        #   compiled 19.32 ms/decide   (1.49x)
        #
        # with decisions bit-identical to 8.4e-11 in action space and 7.3e-16
        # in objective, i.e. float64 reassociation noise. The speedup only
        # materialises after the kernel is compiled, so the default is on and
        # the compile cost is paid at startup by ``warm_soft_fill_compile``
        # rather than inside a decide, which is bounded by
        # ``decision_timeout_ms``.
        #
        # ``MM_TORCH_COMPILE=0`` forces eager, as does ``compile_soft_fill=False``.
        # Any compile or runtime failure falls back to eager and is logged.
        if compile_soft_fill is None:
            compile_env = os.environ.get("MM_TORCH_COMPILE", "1").strip().lower()
            self.compile_soft_fill = compile_env in ("1", "true", "yes", "on")
        else:
            self.compile_soft_fill = bool(compile_soft_fill)
        self.soft_fill_compile_warmup_ms = 0.0
        if isinstance(warm_start, WarmStartParams):
            self.warm_start = warm_start
        else:
            self.warm_start = WarmStartParams.from_mapping(warm_start)
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
            viability_lambda_from_init(
                local_viability.lambda_init,
                lambda_max=local_viability.lambda_max,
            )
            if local_viability is not None
            else 0.0
        )
        self._viability_lambda_global = (
            viability_lambda_from_init(
                global_viability.lambda_init,
                lambda_max=global_viability.lambda_max,
            )
            if global_viability is not None
            else 0.0
        )
        self._viability_shortfall_ewma_local: float | None = None
        self._viability_shortfall_ewma_global: float | None = None
        # Cooperative cancellation: ``StrategyRunner`` installs a
        # ``threading.Event`` via ``set_cancel_event``; the Adam loop
        # checks ``is_set()`` at the top of every epoch and returns the
        # best-so-far result so the pool worker is freed on timeout.
        self._cancel_event: Any = None
        self._cancel_requested: bool = False

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

        # Pay the inductor compile cost here, at construction, rather than on
        # the first decide. This needs ``_l_param_config()`` and the venue
        # fill semantics, so it can only run once ``local_n_components`` and
        # the venue optimizer params exist. Both local and global shapes are
        # warmed because the joint path -- the one production actually takes --
        # calls both, and a second shape would otherwise trigger a second lazy
        # compile inside a decide.
        #
        # ``depth_scale`` is warmed per venue as well: it reaches the kernel as
        # a Python float and is part of the inductor specialization (measured:
        # a masked 0.0 call after a 1.0 warm-up added a frame and cost a 2.7 s
        # lazy compile). The joint path passes 1.0 for local and
        # ``global_mask`` for global, and ``global_mask`` is 0.0 whenever the
        # venue is inactive -- which is every production decision while
        # liquidity sourcing is off. So warm both 1.0 and 0.0.
        if self.compile_soft_fill:
            self.soft_fill_compile_warmup_ms = 0.0
            for venue, n_comp, depth_scales in (
                ("local", self.local_n_components, (1.0,)),
                ("global", self.global_n_components, (1.0, 0.0)),
            ):
                venue_params = self._venue_optimizer(venue)
                for depth_scale in depth_scales:
                    self.soft_fill_compile_warmup_ms += warm_soft_fill_compile(
                        n_components=n_comp,
                        n_samples=self.n_samples,
                        grid_levels=self.grid_levels,
                        l_param=self._l_param_config(venue),
                        depth_kw=self._depth_call_kwargs(venue),
                        maker_fee=float(
                            getattr(
                                self.execution_model,
                                f"{venue}_maker_fee",
                                getattr(self.execution_model, "maker_fee", 0.0),
                            )
                        ),
                        soft_temp=float(venue_params.soft_fill_temp),
                        soft_fill_activation=venue_params.soft_fill_activation,
                        soft_fill_queue_model=venue_params.soft_fill_queue_model,
                        device=self.device,
                        dtype=self._resolve_dtype(torch),
                        depth_scale=depth_scale,
                    )

    @property
    def last_diagnostics(self) -> OptimizerRunDiagnostics | None:
        return self._last_diagnostics

    def _resolve_dtype(self, torch_mod: Any) -> Any:
        return _resolve_tensor_dtype(torch_mod, self.tensor_dtype)

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
        s_max = (
            float(self.spread_max_bps)
            if (getattr(self, "is_bps", False) and getattr(self, "spread_max_bps", None) is not None)
            else float(self.spread_max)
        )
        return self._venue_optimizer(venue).l_param_config(
            spread_max=s_max,
            n_components=n_components,
        )

    def _l_param_config_for_mid(self, mid_price: float, venue: str = "local") -> LParamConfig:
        from dataclasses import replace

        n_components = (
            self.global_n_components if str(venue).lower() == "global" else self.local_n_components
        )
        quote_s_max = float(mid_price) * float(self.spread_max_bps) / 10000.0
        l_param = self._venue_optimizer(venue).l_param_config(
            spread_max=quote_s_max,
            n_components=n_components,
        )
        if (
            l_param.spread_structured
            and l_param.quote_spread_max is not None
            and float(self.spread_max_bps) > 0
        ):
            ratio = float(l_param.quote_spread_max) / float(self.spread_max_bps)
            l_param = replace(l_param, quote_spread_max=quote_s_max * ratio)
        return l_param

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
        l_param = (
            self._l_param_config_for_mid(mid_price)
            if (getattr(self, "is_bps", False) and mid_price is not None and mid_price > 0 and getattr(self, "spread_max_bps", None) is not None)
            else self._l_param_config()
        )
        return sigmoid_action_from_flat_raw(
            flat,
            self.local_n_components,
            l_param=l_param,
            mid_price=mid_price,
            bid_quote_size_min=bid_quote_size_min,
            bid_quote_size_max=bid_quote_size_max,
            ask_quote_size_min=ask_quote_size_min,
            ask_quote_size_max=ask_quote_size_max,
        )

    def _evaluate_rollout(
        self,
        x_action: Any,
        *,
        buy_vol: Any,
        sell_vol: Any,
        eps_buy: Any,
        eps_sell: Any,
        price_innov: dict | None,
        weights: Any | None,
        state: State,
        spread_grid: Any,
        book_cum_bid: Any,
        book_cum_ask: Any,
        ob_bids_0: Any,
        ob_asks_0: Any,
        mid_0: float,
        wealth_0: float,
        maker_fee: float,
        bid_quote_size_min_0: float,
        bid_quote_size_max_0: float,
        ask_quote_size_min_0: float,
        ask_quote_size_max_0: float,
        depth_kw: dict[str, Any],
        single_len: int,
        n: int,
        dtype: Any,
        _device: Any,
        want_metrics: bool = True,
    ) -> tuple[Any, dict[str, Any], EpochObjectiveMetrics | None]:
        torch_mod = _get_torch()
        n_samples = buy_vol.shape[0]
        inv_t = torch_mod.full((n_samples, 1), float(state.inventory), dtype=dtype, device=_device)
        cash_t = torch_mod.full((n_samples, 1), float(state.cash), dtype=dtype, device=_device)
        mid_fallback_t = torch_mod.full((n_samples, 1), mid_0, dtype=dtype, device=_device)
        book_cum_bid_0 = book_cum_bid.unsqueeze(0).expand(n_samples, -1)
        book_cum_ask_0 = book_cum_ask.unsqueeze(0).expand(n_samples, -1)
        mid_init = _mid_from_book_best_levels_batched(
            ob_bids_0,
            ob_asks_0,
            mid_fallback_t,
            spread_grid,
        )
        inv_path_0 = inv_t.squeeze(1)
        cash_path_0 = cash_t.squeeze(1)
        mid_path_0 = mid_init.squeeze(1)
        total_reward = torch_mod.zeros(n_samples, dtype=dtype, device=_device)
        total_main_reward = torch_mod.zeros(n_samples, dtype=dtype, device=_device)
        step_reward_0 = torch_mod.zeros(n_samples, dtype=dtype, device=_device)
        inv_path = [inv_path_0]
        cash_path = [cash_path_0]
        mid_path = [mid_path_0]

        (
            bid_quote_size_min_0_t,
            bid_quote_size_max_0_t,
            ask_quote_size_min_0_t,
            ask_quote_size_max_0_t,
        ) = _quote_capacity_bounds_batched(
            cash=cash_t,
            inventory=inv_t,
            mid_price=mid_init,
            optimizer=self.local_optimizer,
        )
        conditional_price_model = hasattr(self.local_price_model, "sample_dynamic_vwap")
        conditional_buy_vwap_t = None
        conditional_sell_vwap_t = None
        buy_vwap_const_t = torch_mod.full(
            (n_samples, 1), float(state.buy_vwap), dtype=dtype, device=_device
        )
        sell_vwap_const_t = torch_mod.full(
            (n_samples, 1), float(state.sell_vwap), dtype=dtype, device=_device
        )

        ob_bids = ob_bids_0
        ob_asks = ob_asks_0

        fill_fn = resolve_compiled_soft_fill(
            torch_mod,
            enabled=self.compile_soft_fill,
            n_components=n,
            n_samples=int(n_samples),
            grid_levels=int(spread_grid.shape[-1]),
            soft_fill_activation=self.soft_fill_activation,
            soft_fill_queue_model=self.soft_fill_queue_model,
            depth_scale=1.0,
            soft_temp=self.soft_fill_temp,
            maker_fee=maker_fee,
        )

        for h in range(self.horizon):
            if h == 0:
                book_cum_bid_h = book_cum_bid_0
                book_cum_ask_h = book_cum_ask_0
                mid_t = mid_init
                (
                    bid_quote_size_min_t,
                    bid_quote_size_max_t,
                    ask_quote_size_min_t,
                    ask_quote_size_max_t,
                ) = (
                    bid_quote_size_min_0_t,
                    bid_quote_size_max_0_t,
                    ask_quote_size_min_0_t,
                    ask_quote_size_max_0_t,
                )
            else:
                book_cum_bid_h, book_cum_ask_h = _book_cumulative_from_arrays(ob_bids, ob_asks)
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

            action_h = x_action[h * single_len : (h + 1) * single_len]
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

            if conditional_price_model:
                buy_vwap_t = (
                    buy_vwap_const_t
                    if conditional_buy_vwap_t is None
                    else conditional_buy_vwap_t
                )
                sell_vwap_t = (
                    sell_vwap_const_t
                    if conditional_sell_vwap_t is None
                    else conditional_sell_vwap_t
                )
            else:
                buy_vwap_book, sell_vwap_book = vwap_prices_batched(
                    ob_bids,
                    ob_asks,
                    mid_t,
                    spread_grid,
                    target_depth=torch_mod.abs(inv_t),
                )
                buy_vwap_t = buy_vwap_book + eps_buy_h
                sell_vwap_t = sell_vwap_book + eps_sell_h

            try:
                ob_bids_next, ob_asks_next, cash_delta, inv_delta = fill_fn(
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
            except Exception:
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
                        ob_bids, ob_asks, mid_t, spread_grid, target_depth=ref_depth
                    )
                )
                mech_spread_ref, mech_buy_shape_ref, mech_sell_shape_ref = (
                    _fixed_depth_shape_features_batched(
                        ob_bids_next, ob_asks_next, mid_next, spread_grid, target_depth=ref_depth
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
                    buy_depth=torch_mod.relu(cash_next / torch_mod.clamp(mid_next, min=1e-9)),
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
            inv_path.append(inv_next.squeeze(1))
            cash_path.append(cash_next.squeeze(1))
            mid_path.append(mid_next.squeeze(1))
            total_reward = total_reward + step_reward
            total_main_reward = total_main_reward + step_reward_main
            mid_fallback_t = mid_next
            inv_t = inv_next
            cash_t = cash_next

        returns = total_reward / wealth_0
        if weights is not None:
            w_norm = weights.view(-1) / max(1e-12, weights.view(-1).sum())
            J_mean_raw = torch_mod.nan_to_num((total_reward.view(-1) * w_norm).sum(), nan=0.0, posinf=1e30, neginf=-1e30)
            J_return_raw = torch_mod.nan_to_num((returns.view(-1) * w_norm).sum(), nan=0.0, posinf=1e30, neginf=-1e30)
            J_mean_next_raw = torch_mod.nan_to_num((step_reward_0.view(-1) * w_norm).sum(), nan=0.0, posinf=1e30, neginf=-1e30)
            main_reward_mean_raw = torch_mod.nan_to_num((total_main_reward.view(-1) * w_norm).sum(), nan=0.0, posinf=1e30, neginf=-1e30)
        else:
            J_mean_raw = torch_mod.nan_to_num(total_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30)
            J_return_raw = torch_mod.nan_to_num(returns.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30)
            J_mean_next_raw = torch_mod.nan_to_num(step_reward_0.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30)
            main_reward_mean_raw = torch_mod.nan_to_num(total_main_reward.view(-1).mean(), nan=0.0, posinf=1e30, neginf=-1e30)

        global_reward_mean_raw = torch_mod.zeros((), dtype=dtype, device=_device)
        if weights is not None:
            J, components = terminal_objective(
                returns,
                return_risk_aversion_beta=self.return_risk_aversion_beta,
                weights=weights,
            )
        else:
            J, components = terminal_objective(
                returns,
                return_risk_aversion_beta=self.return_risk_aversion_beta,
            )
        J, components = self.normalize_terminal_objective(J, components)

        venue_opt = self.local_optimizer
        if venue_opt.validity is not None:
            book_params = venue_opt.book_quality or BookQualityParams()
            use_combined = book_params.measure == "combined"
            book_score, book_components = book_quality_score(
                x_action[:single_len],
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
            J, components = combine_validity_term(
                J,
                components,
                params=venue_opt.validity,
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
                viability_params=venue_opt.viability,
                viability_shortfall=shortfall,
                viability_components=viability_components,
            )

        l_cfg = self._l_param_config()
        if l_cfg is not None and float(l_cfg.reservation_r_max) > 0.0:
            r_off_t = reservation_r_offset_from_params(
                x_action[:single_len],
                n_components=n,
                r_max=float(l_cfg.reservation_r_max),
            )
            r_idx = reservation_logit_index(n)
            raw_r = x_action[r_idx]
            components["reservation_r_offset"] = r_off_t
            components["reservation_shift_quote"] = r_off_t * mid_0
            components["reservation_raw_logit"] = raw_r

        def _finite_item(t) -> float:
            try:
                if hasattr(t, "detach"):
                    x = float(t.detach().item())
                else:
                    x = float(t)
                return x if math.isfinite(x) else 0.0
            except Exception:
                return 0.0

        epoch_metrics = None
        if want_metrics:
            obj_val = _finite_item(J)
            epoch_metrics = EpochObjectiveMetrics.from_tensors(
                objective_value=obj_val,
                components={k: _finite_item(v) for k, v in components.items()},
                expected_total_reward=_finite_item(J_mean_raw),
                expected_total_return=_finite_item(J_return_raw),
                expected_next_reward=_finite_item(J_mean_next_raw),
                main_reward_mean=_finite_item(main_reward_mean_raw),
                global_reward_mean=_finite_item(global_reward_mean_raw),
            )

        return J, components, epoch_metrics

    def _optimize_discrete_trust_region(
        self,
        x_init: Any,
        lower: Any,
        upper: Any,
        single_len: int,
        n_components: int,
        eval_fn: Any,
    ) -> tuple[Any, EpochObjectiveMetrics, EpochObjectiveMetrics, int, str]:
        """
        Derivative-free discrete trust-region / pattern search over physical ticks, lots, and reservation skew.
        Guarantees monotonic objective improvement or local convergence.
        """
        torch_mod = _get_torch()
        best_x = x_init.detach().clone()
        with torch_mod.no_grad():
            J_best_t, _, metrics_init = eval_fn(best_x, want_metrics=True)
            best_obj = float(J_best_t.item())

        initial_metrics = metrics_init
        best_metrics = metrics_init
        best_iter = 0
        r_ticks = float(self.trust_region_radius_ticks)
        min_ticks = float(self.trust_region_min_radius_ticks)
        size_step = float(self.trust_region_size_step)
        r_step = float(self.trust_region_r_step)
        termination_reason = "max_iterations"
        n = n_components

        spread_cap = float(self.spread_max)
        spread_step = float(self.spread_step)
        effective_tick_fraction = max(1e-4, spread_step / max(spread_cap, spread_step))

        for iteration in range(self.trust_region_iterations):
            if self._cancel_event is not None and self._cancel_event.is_set():
                termination_reason = "cancelled"
                break

            improved_this_iter = False
            candidates: list[Any] = []

            # Calibrated logit step that shifts physical quotes by r_ticks on the spread grid
            logit_tick_delta = max(0.2, min(3.0, 8.0 * r_ticks * effective_tick_fraction))

            # 1. Bilateral spread moves (scale logit at index 0)
            for multiplier in (1.0, 2.0):
                c_tight = best_x.clone()
                c_tight[0] -= multiplier * logit_tick_delta
                candidates.append(c_tight)

                c_wide = best_x.clone()
                c_wide[0] += multiplier * logit_tick_delta
                candidates.append(c_wide)

            # 2. Individual side spread shifts and skews (u_bid index 2, u_ask index 3*n + 2)
            for c in range(n):
                idx_b = 3 * c + 2
                idx_a = 3 * n + 3 * c + 2
                for delta in (-logit_tick_delta, logit_tick_delta):
                    c_b = best_x.clone()
                    c_b[idx_b] += delta
                    candidates.append(c_b)

                    c_a = best_x.clone()
                    c_a[idx_a] += delta
                    candidates.append(c_a)

                # Coordinated skew: bid tight / ask wide, and bid wide / ask tight
                c_skew_b_tight = best_x.clone()
                c_skew_b_tight[idx_b] -= logit_tick_delta
                c_skew_b_tight[idx_a] += logit_tick_delta
                candidates.append(c_skew_b_tight)

                c_skew_a_tight = best_x.clone()
                c_skew_a_tight[idx_b] += logit_tick_delta
                c_skew_a_tight[idx_a] -= logit_tick_delta
                candidates.append(c_skew_a_tight)

            # 3. Depth logits / quote size moves
            idx_depth_b = 6 * n
            idx_depth_a = 6 * n + 1
            if idx_depth_a < single_len:
                for s_delta in (-size_step, size_step):
                    c_db = best_x.clone()
                    c_db[idx_depth_b] += s_delta
                    candidates.append(c_db)

                    c_da = best_x.clone()
                    c_da[idx_depth_a] += s_delta
                    candidates.append(c_da)

                    c_both_d = best_x.clone()
                    c_both_d[idx_depth_b] += s_delta
                    c_both_d[idx_depth_a] += s_delta
                    candidates.append(c_both_d)

            # 4. Reservation price offset / skew (index 6*n + 2)
            idx_res = 6 * n + 2
            if idx_res < single_len:
                for r_delta in (-r_step, r_step, -2.0 * r_step, 2.0 * r_step):
                    c_res = best_x.clone()
                    c_res[idx_res] += r_delta
                    candidates.append(c_res)

            # 5. Curve steepness k (index 1 for bid, 3*n + 1 for ask)
            for k_delta in (-0.5, 0.5):
                c_kb = best_x.clone()
                c_kb[1] += k_delta
                candidates.append(c_kb)

                c_ka = best_x.clone()
                c_ka[3 * n + 1] += k_delta
                candidates.append(c_ka)

            # Evaluate candidates
            iter_best_cand = None
            iter_best_obj = best_obj
            iter_best_metrics = None

            with torch_mod.no_grad():
                for cand in candidates:
                    for h in range(1, self.horizon):
                        cand[h * single_len : (h + 1) * single_len] = cand[:single_len]
                    cand = torch_mod.clamp(cand, lower, upper)
                    if torch_mod.allclose(cand, best_x, atol=1e-7):
                        continue

                    J_cand, _, metrics_cand = eval_fn(cand, want_metrics=True)
                    val = float(J_cand.item())
                    if math.isfinite(val) and val > iter_best_obj + 1e-8:
                        iter_best_obj = val
                        iter_best_cand = cand.clone()
                        iter_best_metrics = metrics_cand

            if iter_best_cand is not None:
                best_x = iter_best_cand
                best_obj = iter_best_obj
                best_metrics = iter_best_metrics
                best_iter = iteration + 1
                improved_this_iter = True

            if not improved_this_iter:
                if r_ticks > min_ticks:
                    r_ticks = max(min_ticks, r_ticks / 2.0)
                    size_step = max(0.5, size_step / 2.0)
                else:
                    termination_reason = "local_optimum_reached"
                    break

        return best_x, initial_metrics, best_metrics, best_iter, termination_reason

    def optimize(self, state: State) -> SigmoidMixtureAction:
        action, _ = self.optimize_with_diagnostics(state)
        return action

    def optimize_with_diagnostics(self, state: State) -> tuple[SigmoidMixtureAction, OptimizerRunDiagnostics]:
        if getattr(self, "is_bps", False) and state.mid > 0 and getattr(self, "spread_max_bps", None) is not None and getattr(self, "spread_step_bps", None) is not None:
            self.spread_max = float(state.mid) * float(self.spread_max_bps) / 10000.0
            self.spread_step = max(1e-12, float(state.mid) * float(self.spread_step_bps) / 10000.0)
            self.grid_levels = _grid_levels(self.spread_max, self.spread_step)
        has_global_reference = (
            self.reference_price_source == "global"
            and state.global_sell_vwap is not None
            and float(state.global_sell_vwap) > 0.0
        )
        if self.liquidity_sourcing_enabled or has_global_reference:
            from mm.mpc.joint_optimizer import optimize_joint_with_diagnostics

            return optimize_joint_with_diagnostics(self, state)
        torch_mod = _get_torch()
        stage_timer = OptimizerStageTimer()
        setup_started = time.perf_counter()
        if isinstance(self.device, str) and self.device.lower() == "cpu":
            _device = torch_mod.device("cpu")
        elif self.device is None:
            _device = resolve_torch_device(None)
        else:
            _device = resolve_torch_device(str(self.device).strip() or None)
        dtype = self._resolve_dtype(torch_mod)

        book_cum_bid, book_cum_ask = _book_cumulative_from_ob_rel(
            state.ob_rel, self.grid_levels, self.spread_step, _device, dtype
        )
        spread_grid = _spread_grid_cached(
            int(self.grid_levels),
            float(self.spread_step),
            float(self.spread_max),
            str(_device),
            str(dtype).replace("torch.", ""),
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

        use_stratified = (
            self.flow_scenario_mode == "weighted_stratified"
            and hasattr(self.local_market_flow_model, "stratified_scenarios")
        )
        scenario_weights = None
        p0_buy = 0.0
        p0_sell = 0.0
        fixed_buy_vol = None
        fixed_sell_vol = None
        fixed_eps_buy = None
        fixed_eps_sell = None
        fixed_price_innov = None
        batch_time_sec = 0.0

        if use_stratified:
            t_batch = time.perf_counter()
            strat = self.local_market_flow_model.stratified_scenarios(
                horizon=self.horizon,
                positive_nodes=self.stratified_positive_nodes,
                device=_device,
                dtype=dtype,
            )
            fixed_buy_vol = strat.buy_vol
            fixed_sell_vol = strat.sell_vol
            scenario_weights = strat.weights
            p0_buy = strat.p0_buy
            p0_sell = strat.p0_sell
            n_samples = int(fixed_buy_vol.shape[0])
            if hasattr(self.local_price_model, "sample_innovations"):
                fixed_price_innov = self.local_price_model.sample_innovations(
                    (n_samples, self.horizon), device=_device, dtype=dtype
                )
                fixed_eps_buy = torch_mod.zeros((n_samples, self.horizon), dtype=dtype, device=_device)
                fixed_eps_sell = torch_mod.zeros((n_samples, self.horizon), dtype=dtype, device=_device)
            elif hasattr(self.local_price_model, "sample_residual"):
                fixed_eps_buy, fixed_eps_sell = self.local_price_model.sample_residual(
                    (n_samples, self.horizon), device=_device, dtype=dtype
                )
                fixed_price_innov = None
            else:
                fixed_eps_buy = torch_mod.zeros((n_samples, self.horizon), dtype=dtype, device=_device)
                fixed_eps_sell = torch_mod.zeros((n_samples, self.horizon), dtype=dtype, device=_device)
                fixed_price_innov = None
            batch_time_sec = time.perf_counter() - t_batch
            stage_timer.record("sample", batch_time_sec * 1000.0)
        else:
            n_samples = self.n_samples

        ob_bids_0 = torch_mod.zeros(n_samples, self.grid_levels, dtype=dtype, device=_device)
        ob_asks_0 = torch_mod.zeros(n_samples, self.grid_levels, dtype=dtype, device=_device)
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

        u_default = _initial_action_uniform_marginal_qrange(
            n_components=n,
            spread_max=self.spread_max,
            l_param=self._l_param_config(),
        )
        u_warm = warm_action_from_mpc_state(
            params=self.warm_start,
            last_raw_flat=self._last_raw_flat,
            last_action_flat=(
                self._last_action.to_flat() if self._last_action is not None else None
            ),
            expected_len=single_len,
            finite_flat_or_none=_finite_action_flat_or_none,
        )
        needs_adaptive_probe = (
            self.warm_start.enabled
            and u_warm is not None
            and self.warm_start.alpha_mode == "adaptive"
        )
        if needs_adaptive_probe:
            x0_arr = np.tile(u_default, self.horizon)
            warm_meta: dict[str, float] = {}
        else:
            x0_arr, warm_meta = prepare_warm_start_initial_flat(
                params=self.warm_start,
                u_default=u_default,
                u_warm=u_warm,
                horizon=self.horizon,
            )

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
            if not (self.warm_start.enabled and self.warm_start.preserve_optimizer_state):
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
            if self.crn_enabled:
                torch_mod.manual_seed(self.crn_seed)
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

        stage_timer.record("setup", (time.perf_counter() - setup_started) * 1000.0)
        if fixed_buy_vol is None and (not use_stratified and (self.fixed_mc_samples or self.optimizer_mode == "discrete_trust_region")):
            t_batch = time.perf_counter()
            (
                fixed_buy_vol,
                fixed_sell_vol,
                fixed_eps_buy,
                fixed_eps_sell,
                fixed_price_innov,
            ) = draw_samples()
            batch_time_sec += time.perf_counter() - t_batch
            stage_timer.record("sample", batch_time_sec * 1000.0)

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
        best_epoch_metrics: EpochObjectiveMetrics | None = None
        self._last_J_components_initial = None
        captured_initial_components = False
        nonfinite_loss_count = 0
        nonfinite_grad_count = 0
        last_eval_x = best_x.detach().clone()
        t0 = time.perf_counter()
        # Wall-clock accumulated inside ``closure`` so the Adam stage can be
        # measured as the residual of ``optimizer.step`` rather than double
        # counting the rollout/objective work it already contains.
        closure_wall_ms = 0.0
        # Monte Carlo draws issued inside ``closure`` are already part of the
        # forward region, so they are subtracted to keep ``optimize_ms`` a
        # clean sum of non-overlapping stages.
        sample_inside_forward_ms = 0.0
        adaptive_warm_resolved = not needs_adaptive_probe

        # ---- Epoch-invariant rollout constants -------------------------------
        # Every value below is a pure function of the incoming state and the
        # optimizer config, so it is identical on all ``adam_epochs`` closures.
        # Building them once outside the loop removes redundant recomputation
        # from the hottest path. ``mid_init`` alone is ~38us at production
        # shapes (n_samples=64, grid_levels=10), i.e. ~0.76ms per decide across
        # 20 epochs.
        n_samples = int(fixed_buy_vol.shape[0]) if fixed_buy_vol is not None else self.n_samples
        inv_t_0 = torch_mod.full((n_samples, 1), inv_0, dtype=dtype, device=_device)
        cash_t_0 = torch_mod.full((n_samples, 1), cash_0, dtype=dtype, device=_device)
        mid_fallback_0 = torch_mod.full((n_samples, 1), mid_0, dtype=dtype, device=_device)
        book_cum_bid_0 = book_cum_bid.unsqueeze(0).expand(n_samples, -1)
        book_cum_ask_0 = book_cum_ask.unsqueeze(0).expand(n_samples, -1)
        mid_init = _mid_from_book_best_levels_batched(
            ob_bids_0,
            ob_asks_0,
            mid_fallback_0,
            spread_grid,
        )
        inv_path_0 = inv_t_0.squeeze(1)
        cash_path_0 = cash_t_0.squeeze(1)
        mid_path_0 = mid_init.squeeze(1)
        total_reward_0 = torch_mod.zeros(n_samples, dtype=dtype, device=_device)
        total_main_reward_0 = torch_mod.zeros(n_samples, dtype=dtype, device=_device)
        step_reward_0_init = torch_mod.zeros(n_samples, dtype=dtype, device=_device)

        # More epoch-invariant ``h == 0`` state. At the first horizon step the
        # book, inventory, cash and mid are all still the initial ones, so the
        # capacity bounds, the VWAP target depths and the upstream VWAP
        # constants are the same on every epoch. Deriving them once here keeps
        # the per-epoch path from redoing the identical arithmetic.
        (
            bid_quote_size_min_0_t,
            bid_quote_size_max_0_t,
            ask_quote_size_min_0_t,
            ask_quote_size_max_0_t,
        ) = _quote_capacity_bounds_batched(
            cash=cash_t_0,
            inventory=inv_t_0,
            mid_price=mid_init,
            optimizer=self.local_optimizer,
        )
        sell_vwap_depth_0_t = torch_mod.abs(inv_t_0)
        buy_vwap_depth_0_t = torch_mod.relu(cash_t_0 / torch_mod.clamp(mid_init, min=1e-9))
        buy_vwap_const_t = torch_mod.full(
            (n_samples, 1), float(state.buy_vwap), dtype=dtype, device=_device
        )
        sell_vwap_const_t = torch_mod.full(
            (n_samples, 1), float(state.sell_vwap), dtype=dtype, device=_device
        )

        if self.optimizer_mode == "discrete_trust_region":
            t_tr_start = time.perf_counter()
            depth_kw = self._depth_call_kwargs()

            def tr_eval_fn(cand_x: Any, want_metrics: bool = True):
                return self._evaluate_rollout(
                    cand_x,
                    buy_vol=fixed_buy_vol,
                    sell_vol=fixed_sell_vol,
                    eps_buy=fixed_eps_buy,
                    eps_sell=fixed_eps_sell,
                    price_innov=fixed_price_innov,
                    weights=scenario_weights,
                    state=state,
                    spread_grid=spread_grid,
                    book_cum_bid=book_cum_bid,
                    book_cum_ask=book_cum_ask,
                    ob_bids_0=ob_bids_0,
                    ob_asks_0=ob_asks_0,
                    mid_0=mid_0,
                    wealth_0=wealth_0,
                    maker_fee=maker_fee,
                    bid_quote_size_min_0=bid_quote_size_min_0,
                    bid_quote_size_max_0=bid_quote_size_max_0,
                    ask_quote_size_min_0=ask_quote_size_min_0,
                    ask_quote_size_max_0=ask_quote_size_max_0,
                    depth_kw=depth_kw,
                    single_len=single_len,
                    n=n,
                    dtype=dtype,
                    _device=_device,
                    want_metrics=want_metrics,
                )

            best_x, initial_metrics, best_metrics, best_epoch, termination_reason = (
                self._optimize_discrete_trust_region(
                    x_init=x_torch,
                    lower=lower,
                    upper=upper,
                    single_len=single_len,
                    n_components=n,
                    eval_fn=tr_eval_fn,
                )
            )
            final_metrics = best_metrics
            with torch_mod.no_grad():
                x_torch.data.copy_(best_x)
            if best_metrics is not None:
                best_loss = -float(best_metrics.objective_value)
                best_J_mean = best_metrics.expected_total_reward
                best_J_mean_next = best_metrics.expected_next_reward
                best_main_reward_mean = best_metrics.main_reward_mean
                self._last_J_components = dict(best_metrics.components)
                self._last_J_mean_raw = best_metrics.expected_total_reward
                self._last_E_dW_next = best_metrics.expected_next_reward
                self._last_main_reward_mean = best_metrics.main_reward_mean
                self._last_global_reward_mean = best_metrics.global_reward_mean
            if initial_metrics is not None:
                self._last_J_components_initial = dict(initial_metrics.components)
                self._last_main_reward_mean_initial = initial_metrics.main_reward_mean
                self._last_global_reward_mean_initial = initial_metrics.global_reward_mean
            stage_timer.record("trust_region", (time.perf_counter() - t_tr_start) * 1000.0)
            stage_timer.bump_epochs()
        else:
            for epoch in range(self.adam_epochs):
                # Cooperative cancellation: ``StrategyRunner`` installs a
                # ``threading.Event`` so a timed-out decide can break out of
                # the Adam loop without killing the worker thread. The
                # best-so-far result is returned so the actor can keep its
                # most recent action rather than throwing away work.
                if self._cancel_event is not None and self._cancel_event.is_set():
                    self._cancel_requested = True
                    break
                if self.verbose:
                    cur_flat = x_torch.detach().cpu().numpy()[:single_len]
                    print(f"  Epoch {epoch}: action  {_format_action_verbose(cur_flat, n)}")

                def closure(probe_only: bool = False, want_metrics: bool = True):
                    nonlocal batch_time_sec, captured_initial_components
                    nonlocal initial_metrics, final_metrics
                    nonlocal nonfinite_loss_count, nonfinite_grad_count, last_eval_x
                    nonlocal closure_wall_ms, sample_inside_forward_ms
                    nonlocal best_epoch_metrics
                    closure_started = time.perf_counter()
                    forward_started = time.perf_counter()
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
                        drawn = time.perf_counter() - t_batch
                        batch_time_sec += drawn
                        stage_timer.record("sample", drawn * 1000.0)
                        sample_inside_forward_ms += drawn * 1000.0

                    # ``inv_t``/``cash_t``/``mid_fallback_t`` are read-only here (only
                    # rebound to derived values below), so reuse the hoisted
                    # constants instead of reallocating them per epoch.
                    inv_t = inv_t_0
                    cash_t = cash_t_0
                    # ``ob_bids``/``ob_asks`` are only read or rebound within the
                    # horizon loop (``ob_bids, ob_asks = ob_bids_next, ob_asks_next``);
                    # nothing writes into them in place, so rebinding straight to the
                    # epoch-invariant originals is safe and skips two per-epoch
                    # (n_samples, grid_levels) copies.
                    ob_bids = ob_bids_0
                    ob_asks = ob_asks_0
                    mid_fallback_t = mid_fallback_0

                    total_reward = total_reward_0
                    total_main_reward = total_main_reward_0
                    step_reward_0 = step_reward_0_init
                    inv_path: list[Any] = [inv_path_0]
                    cash_path: list[Any] = [cash_path_0]
                    mid_path: list[Any] = [mid_path_0]
                    conditional_price_model = hasattr(self.local_price_model, "sample_dynamic_vwap")
                    conditional_buy_vwap_t = None
                    conditional_sell_vwap_t = None

                    for h in range(self.horizon):
                        if h == 0:
                            book_cum_bid_h = book_cum_bid_0
                            book_cum_ask_h = book_cum_ask_0
                            # ``ob_bids``/``ob_asks`` are still the epoch-invariant
                            # initial book and ``mid_fallback_t`` is ``mid_fallback_0``,
                            # so this is exactly the ``mid_init`` computed before the
                            # Adam loop. Reuse it instead of re-deriving it on every
                            # epoch's first horizon step.
                            mid_t = mid_init
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

                        if h != 0:
                            mid_t = _mid_from_book_best_levels_batched(
                                ob_bids,
                                ob_asks,
                                mid_fallback_t,
                                spread_grid,
                            )
                        if h == 0:
                            # Everything here is a function of the initial book,
                            # inventory, cash and mid only, so reuse the values
                            # derived once before the Adam loop.
                            (
                                bid_quote_size_min_t,
                                bid_quote_size_max_t,
                                ask_quote_size_min_t,
                                ask_quote_size_max_t,
                            ) = (
                                bid_quote_size_min_0_t,
                                bid_quote_size_max_0_t,
                                ask_quote_size_min_0_t,
                                ask_quote_size_max_0_t,
                            )
                            sell_vwap_depth_t = sell_vwap_depth_0_t
                            buy_vwap_depth_t = buy_vwap_depth_0_t
                            # Use upstream VWAPs computed from the initial inventory
                            # depth, instead of recomputing from the book arrays.
                            buy_vwap_book_t = buy_vwap_const_t
                            sell_vwap_book_t = sell_vwap_const_t
                        else:
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
                            buy_vwap_depth_t = torch_mod.relu(cash_t / torch_mod.clamp(mid_t, min=1e-9))
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

                        # Never trigger a lazy inductor compile on the decide
                        # path. An unwarmed shape used to compile here (measured
                        # 162 s), which blew ``decision_timeout_ms`` and left the
                        # controller wedged on "optimizer decision in progress"
                        # for every subsequent frame. A signature absent from the
                        # warmed set means this venue's shape was never paid for at
                        # construction, so run eager -- eager is always correct,
                        # just slower -- and record it so the shortfall is visible
                        # in logs rather than silently costing decisions.
                        venue_n = n
                        soft_fill_fn = resolve_compiled_soft_fill(
                            torch_mod,
                            enabled=self.compile_soft_fill,
                            n_components=venue_n,
                            n_samples=int(mid_t.shape[0]),
                            grid_levels=int(spread_grid.shape[-1]),
                            soft_fill_activation=self.soft_fill_activation,
                            soft_fill_queue_model=self.soft_fill_queue_model,
                            depth_scale=1.0,
                            soft_temp=self.soft_fill_temp,
                            maker_fee=maker_fee,
                        )
                        _mark_batch_dims_dynamic(
                            mid_t, book_cum_bid_h, book_cum_ask_h, sell_h, buy_h,
                            bid_quote_size_min_t, bid_quote_size_max_t,
                            ask_quote_size_min_t, ask_quote_size_max_t,
                        )
                        try:
                            ob_bids_next, ob_asks_next, cash_delta, inv_delta = soft_fill_fn(
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
                        except Exception as exc:
                            if self.compile_soft_fill:
                                log_event(
                                    "mpc.torch_compile.runtime_fallback",
                                    {"error": str(exc)},
                                )
                                self.compile_soft_fill = False
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
                            else:
                                raise
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
                                buy_depth=torch_mod.relu(cash_next / torch_mod.clamp(mid_next, min=1e-9)),
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
                        inv_path.append(inv_next.squeeze(1))
                        cash_path.append(cash_next.squeeze(1))
                        mid_path.append(mid_next.squeeze(1))
                        total_reward = total_reward + step_reward
                        total_main_reward = total_main_reward + step_reward_main

                        mid_fallback_t = mid_next
                        inv_t = inv_next
                        cash_t = cash_next

                    stage_timer.record(
                        "forward",
                        max(0.0, (time.perf_counter() - forward_started) * 1000.0 - sample_inside_forward_ms),
                    )
                    returns = total_reward / wealth_0
                    objective_started = time.perf_counter()
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
                    if scenario_weights is not None:
                        J, components = terminal_objective(
                            returns,
                            return_risk_aversion_beta=self.return_risk_aversion_beta,
                            weights=scenario_weights,
                        )
                    else:
                        J, components = terminal_objective(
                            returns,
                            return_risk_aversion_beta=self.return_risk_aversion_beta,
                        )
                    J, components = self.normalize_terminal_objective(J, components)
                    venue_opt = self.local_optimizer
                    if venue_opt.validity is not None:
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
                        J, components = combine_validity_term(
                            J,
                            components,
                            params=venue_opt.validity,
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
                            viability_params=venue_opt.viability,
                            viability_shortfall=shortfall,
                            viability_components=viability_components,
                        )

                    l_cfg = self._l_param_config()
                    if l_cfg is not None and float(l_cfg.reservation_r_max) > 0.0:
                        r_off_t = reservation_r_offset_from_params(
                            x_torch[:single_len],
                            n_components=n,
                            r_max=float(l_cfg.reservation_r_max),
                        )
                        r_idx = reservation_logit_index(n)
                        raw_r = x_torch[r_idx]
                        components["reservation_r_offset"] = r_off_t
                        components["reservation_shift_quote"] = r_off_t * mid_0
                        components["reservation_raw_logit"] = raw_r

                    stage_timer.record("objective", (time.perf_counter() - objective_started) * 1000.0)
                    diagnostics_started = time.perf_counter()

                    def _finite_item(t) -> float:
                        try:
                            if hasattr(t, "detach"):
                                x = float(t.detach().item())
                            else:
                                x = float(t)
                            return x if math.isfinite(x) else 0.0
                        except Exception:
                            return 0.0

                    loss = -J
                    obj_value = _finite_item(J)
                    # Per-epoch scalar extraction is a real Python/C boundary
                    # crossing, so full metrics are only materialized for the
                    # sampled epochs plus any epoch that turns out to be a new
                    # best (the published ``best`` snapshot depends on it).
                    # ``best_loss`` tracks the minimum of ``loss = -J``.
                    is_new_best = math.isfinite(obj_value) and (-obj_value) < best_loss
                    if not probe_only and (want_metrics or is_new_best):
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
                        if is_new_best:
                            best_epoch_metrics = epoch_metrics
                    stage_timer.record("diagnostics", (time.perf_counter() - diagnostics_started) * 1000.0)
                    if not torch_mod.isfinite(loss):
                        nonfinite_loss_count += 1
                    if torch_mod.isfinite(loss):
                        backward_started = time.perf_counter()
                        loss.backward()
                        if x_torch.grad is not None:
                            if not torch_mod.isfinite(x_torch.grad).all():
                                nonfinite_grad_count += 1
                                x_torch.grad.detach().zero_()
                            else:
                                torch_mod.nn.utils.clip_grad_norm_([x_torch], self.adam_grad_clip)
                        stage_timer.record("backward", (time.perf_counter() - backward_started) * 1000.0)
                    closure_wall_ms += (time.perf_counter() - closure_started) * 1000.0
                    return loss

                if epoch == 0 and needs_adaptive_probe and not adaptive_warm_resolved:

                    def _grad_norm_probe(one_flat: np.ndarray) -> float:
                        tiled = np.tile(one_flat, self.horizon)
                        with torch_mod.no_grad():
                            x_torch.data.copy_(
                                torch_mod.tensor(tiled, dtype=dtype, device=_device)
                            )
                        optimizer.zero_grad()
                        closure(probe_only=True)
                        if x_torch.grad is None:
                            return 0.0
                        grad_norm = float(x_torch.grad.detach().norm().item())
                        if not math.isfinite(grad_norm):
                            return 0.0
                        return grad_norm

                    x0_arr, warm_meta = prepare_warm_start_initial_flat(
                        params=self.warm_start,
                        u_default=u_default,
                        u_warm=u_warm,
                        horizon=self.horizon,
                        grad_norm_probe=_grad_norm_probe,
                    )
                    with torch_mod.no_grad():
                        x_torch.data.copy_(
                            torch_mod.tensor(x0_arr, dtype=dtype, device=_device)
                        )
                    if not (self.warm_start.enabled and self.warm_start.preserve_optimizer_state):
                        self._optimizer = torch_mod.optim.Adam([x_torch], lr=self.adam_lr)
                        optimizer = self._optimizer
                    adaptive_warm_resolved = True

                want_metrics = _epoch_needs_metrics(
                    epoch, self.adam_epochs - 1, self.optimizer_diagnostics_every
                )
                step_started = time.perf_counter()
                with maybe_profile_decide():
                    loss = optimizer.step(
                        lambda: closure(probe_only=False, want_metrics=want_metrics)
                    )
                # ``optimizer.step`` runs the closure before applying the update, so
                # the Adam arithmetic itself is the residual after the closure cost.
                step_wall_ms = (time.perf_counter() - step_started) * 1000.0
                stage_timer.record("adam", max(0.0, step_wall_ms - closure_wall_ms))
                closure_wall_ms = 0.0
                stage_timer.bump_epochs()
                if loss is not None and torch_mod.isfinite(loss) and loss.item() < best_loss:
                    best_loss = loss.item()
                    best_x = last_eval_x.detach().clone()
                    best_epoch = epoch
                    if best_epoch_metrics is not None:
                        best_metrics = best_epoch_metrics
                        best_J_mean = best_epoch_metrics.expected_total_reward
                        best_J_mean_next = best_epoch_metrics.expected_next_reward
                        best_main_reward_mean = best_epoch_metrics.main_reward_mean

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
        if self.warm_start.enabled:
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
            lambda_applied_local = self._viability_lambda_local
            self._viability_lambda_local, self._viability_shortfall_ewma_local = (
                update_viability_lambda(
                    self._viability_lambda_local,
                    best_metrics.components,
                    params=self.local_optimizer.viability,
                    ewma=self._viability_shortfall_ewma_local,
                )
            )
            for metrics in (initial_metrics, final_metrics, best_metrics):
                if metrics is not None:
                    for key in ("viability_shortfall_ewma", "viability_dual_gap"):
                        if key in best_metrics.components:
                            metrics.components[key] = best_metrics.components[key]
                    stamp_viability_dual_diagnostics(
                        metrics.components,
                        local_lambda_applied=lambda_applied_local,
                        local_lambda_learned=self._viability_lambda_local,
                    )
        snap_final = build_action_snapshot(final_action, mid=mid_0, spread_levels=spread_levels)
        snap_best = build_action_snapshot(best_action, mid=mid_0, spread_levels=spread_levels)

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
                "tensor_dtype": self.tensor_dtype,
                "optimizer_diagnostics_every": self.optimizer_diagnostics_every,
                "objective_normalize": self.objective_normalize.as_mapping(),
                "warm_start": self.warm_start.as_mapping(),
                "warm_start_resolved": warm_meta,
                "local_optimizer": self.local_optimizer.as_config(
                    spread_max=self.spread_max,
                    n_components=self.local_n_components,
                ),
                "global_optimizer": self.global_optimizer.as_config(
                    spread_max=self.spread_max,
                    n_components=self.global_n_components,
                ),
            },
            timing=stage_timing_payload(
                stage_timer,
                legacy={
                    "sample_draw_sec": float(batch_time_sec),
                    "train_sec": float(train_time_sec),
                    "total_sec": float(total_sec),
                    "best_epoch": best_epoch,
                },
            ),
            initial=initial_metrics,
            final=final_metrics,
            best=best_metrics,
            best_epoch=best_epoch,
            action_initial=action_initial_snap,
            action_final=snap_final,
            action_best=snap_best,
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
                "epochs_run": stage_timer.epochs_run,
                "stage_ms": {f"{k}_ms": v for k, v in stage_timer.totals_ms.items()},
                "optimize_ms": stage_timer.total_ms(),
            },
        )
        return best_action, diagnostics

    def reset_warm_start(self) -> None:
        self._last_action = None
        self._x_torch = None
        self._optimizer = None

    def set_cancel_event(self, event: Any) -> None:
        """Install a ``threading.Event`` consulted by the Adam loop.

        ``StrategyRunner._decide_with_timeout`` calls this before
        submitting ``decide`` to its dedicated thread pool, so a
        timed-out decide can break out of the loop without leaking the
        worker thread. The flag is reset on every call so a follow-up
        decide starts with a clean slate.
        """
        self._cancel_event = event
        self._cancel_requested = False

    def was_cancelled(self) -> bool:
        return bool(self._cancel_requested)
