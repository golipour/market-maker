"""
MDP state and action.

State = [mid, orderbook relative to mid, buy_vol, sell_vol, inventory, cash].
Orderbook relative to mid: list of (spread_ticks, qty) per side for depth/fill logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from trade_stats import Orderbook

from .ob_reference_prices import reference_prices_ticks_from_orderbook


@dataclass(frozen=True)
class OrderbookRelative:
    """
    Orderbook in relative terms: (spread_ticks, qty) per level.
    Bids: spread_ticks = mid - price_ticks (positive = below mid).
    Asks: spread_ticks = price_ticks - mid (positive = above mid).
    """
    bids: tuple[tuple[int, int], ...]  # (spread_ticks, qty) each
    asks: tuple[tuple[float, int], ...]


@dataclass(frozen=True)
class MarketState:
    """
    Observable market state x_t: mid, volatility, spread, imbalance, flow stats.
    Kept for backward compatibility; new code can use State.mid and State.ob_rel.
    """
    mid: float


@dataclass(frozen=True)
class Action:
    """
    Order placement relative to mid: deltas in ticks, sizes in (discretized) volume.
    bid price = mid - delta_b_ticks * tick_size, ask price = mid + delta_a_ticks * tick_size.
    """
    delta_b_ticks: float
    delta_a_ticks: float
    q_b: int
    q_a: int


@dataclass(frozen=True)
class DiscretizedOrderAction:
    """
    Discrete ladder of orders per side: list of (spread_ticks, qty) so that
    cumulative volume at each spread approximates a continuous depth curve (e.g. sigmoid mixture).
    Used by ExecutionModel for level-wise fills and cash.

    Semantics: each (spread_ticks, qty) is MARGINAL — qty is the size at that level only.
    Cumulative depth at spread s = sum of qty over all levels with spread_ticks <= s (levels
    are sorted ascending). So the ladder is stored as marginal; cumulative is implied and
    is non-decreasing when built from levels_from_cumulative.
    """
    bids: tuple[tuple[int, int], ...]  # (spread_ticks, qty) ascending spread (best first); qty = marginal
    asks: tuple[tuple[int, int], ...]  # (spread_ticks, qty) ascending spread (best first); qty = marginal


# No quotes on either side; use when controller is disabled (e.g. use_zero_action=True).
ZERO_DISCRETIZED_ACTION = DiscretizedOrderAction(bids=(), asks=())


def _sigmoid_stable(z: float) -> float:
    """Numerically stable sigmoid 1/(1+exp(-z)); avoids overflow for large |z|."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    return math.exp(z) / (1.0 + math.exp(z))


@dataclass(frozen=True)
class SigmoidMixtureAction:
    """
    Action parameterized by n-component mixture-of-sigmoid parameters for bid and ask.

    Depth curve per side: D(spread) = sum_i [ L_i / (1 + exp(-k_i * (spread - x0_i))) ].
    Used with a dedicated MPC optimizer that searches over these parameters instead of
    (delta_b, delta_a, q_b, q_a).

    Attributes:
        bid_L: (L1, ..., Ln) asymptotes for bid side.
        bid_k: (k1, ..., kn) steepness.
        bid_x0: (x0_1, ..., x0_n) inflection points (ticks).
        ask_L, ask_k, ask_x0: same for ask side.
    """

    bid_L: tuple[float, ...]
    bid_k: tuple[float, ...]
    bid_x0: tuple[float, ...]
    ask_L: tuple[float, ...]
    ask_k: tuple[float, ...]
    ask_x0: tuple[float, ...]

    @property
    def n_components(self) -> int:
        return len(self.bid_L)

    def to_flat(self) -> tuple[float, ...]:
        """Return (6 * n_components)-dim vector [bid_L1, bid_k1, bid_x0_1, ...; ask_...] for optimizer."""
        n = self.n_components
        out: list[float] = []
        for i in range(n):
            out.extend([self.bid_L[i], self.bid_k[i], self.bid_x0[i]])
        for i in range(n):
            out.extend([self.ask_L[i], self.ask_k[i], self.ask_x0[i]])
        return tuple(out)

    @classmethod
    def from_flat(cls, x: tuple[float, ...] | list[float], n_components: int | None = None) -> SigmoidMixtureAction:
        """Build from flat vector (same order as to_flat). n_components inferred from len(x)//6 if None."""
        x = list(x)
        if n_components is None:
            n_components = len(x) // 6
        if len(x) < 6 * n_components:
            raise ValueError(f"SigmoidMixtureAction.from_flat requires at least {6 * n_components} elements, got {len(x)}")
        bid_L, bid_k, bid_x0 = [], [], []
        for i in range(n_components):
            bid_L.append(float(x[0 + 3 * i]))
            bid_k.append(float(x[1 + 3 * i]))
            bid_x0.append(float(x[2 + 3 * i]))
        ask_L, ask_k, ask_x0 = [], [], []
        base = 3 * n_components
        for i in range(n_components):
            ask_L.append(float(x[base + 0 + 3 * i]))
            ask_k.append(float(x[base + 1 + 3 * i]))
            ask_x0.append(float(x[base + 2 + 3 * i]))
        return cls(
            bid_L=tuple(bid_L),
            bid_k=tuple(bid_k),
            bid_x0=tuple(bid_x0),
            ask_L=tuple(ask_L),
            ask_k=tuple(ask_k),
            ask_x0=tuple(ask_x0),
        )

    @classmethod
    def from_flat_positive(cls, x: tuple[float, ...] | list[float], n_components: int | None = None) -> SigmoidMixtureAction:
        """Build from flat vector, applying softplus to L and k so cumulative depth is monotonic in spread.
        Use this when the flat vector comes from an optimizer that may have L or k at or near the bound."""
        x = list(x)
        if n_components is None:
            n_components = len(x) // 6
        if len(x) < 6 * n_components:
            raise ValueError(f"SigmoidMixtureAction.from_flat_positive requires at least {6 * n_components} elements, got {len(x)}")
        eps = 1e-8

        def _pos(v: float) -> float:
            # softplus(v) + eps so L, k > 0; avoid overflow for large |v|
            v = float(v)
            if v >= 20:
                return v + eps
            if v <= -20:
                return eps
            return math.log(1.0 + math.exp(v)) + eps

        bid_L, bid_k, bid_x0 = [], [], []
        for i in range(n_components):
            bid_L.append(_pos(float(x[0 + 3 * i])))
            bid_k.append(_pos(float(x[1 + 3 * i])))
            bid_x0.append(float(x[2 + 3 * i]))
        ask_L, ask_k, ask_x0 = [], [], []
        base = 3 * n_components
        for i in range(n_components):
            ask_L.append(_pos(float(x[base + 0 + 3 * i])))
            ask_k.append(_pos(float(x[base + 1 + 3 * i])))
            ask_x0.append(float(x[base + 2 + 3 * i]))
        return cls(
            bid_L=tuple(bid_L),
            bid_k=tuple(bid_k),
            bid_x0=tuple(bid_x0),
            ask_L=tuple(ask_L),
            ask_k=tuple(ask_k),
            ask_x0=tuple(ask_x0),
        )

    def depth_bid(self, spread_ticks: float) -> float:
        """Cumulative bid depth at given spread (mixture of n sigmoids)."""
        out = 0.0
        for i in range(self.n_components):
            z = self.bid_k[i] * (spread_ticks - self.bid_x0[i])
            s = _sigmoid_stable(z)
            out += self.bid_L[i] * s
        return out

    def depth_ask(self, spread_ticks: float) -> float:
        """Cumulative ask depth at given spread (mixture of n sigmoids)."""
        out = 0.0
        for i in range(self.n_components):
            z = self.ask_k[i] * (spread_ticks - self.ask_x0[i])
            s = _sigmoid_stable(z)
            out += self.ask_L[i] * s
        return out

    def depth_bid_derivative(self, spread_ticks: float) -> float:
        """Marginal limit orders at δ: d/dδ V_mm(δ) = sum_i L_i * k_i * s_i * (1 - s_i), s_i = sigmoid(k_i*(δ-x0_i))."""
        out = 0.0
        for i in range(self.n_components):
            z = self.bid_k[i] * (spread_ticks - self.bid_x0[i])
            s = _sigmoid_stable(z)
            out += self.bid_L[i] * self.bid_k[i] * s * (1.0 - s)
        return out

    def depth_ask_derivative(self, spread_ticks: float) -> float:
        """Marginal limit orders at δ: d/dδ V_mm(δ)."""
        out = 0.0
        for i in range(self.n_components):
            z = self.ask_k[i] * (spread_ticks - self.ask_x0[i])
            s = _sigmoid_stable(z)
            out += self.ask_L[i] * self.ask_k[i] * s * (1.0 - s)
        return out


def sigmoid_action_to_action(sigmoid_action: SigmoidMixtureAction) -> Action:
    """
    Convert SigmoidMixtureAction to a single-level Action (legacy).
    Prefer sigmoid_action_to_discretized_action for execution with depth fit.
    """
    delta_b = float(sigmoid_action.bid_x0[0])
    delta_a = float(sigmoid_action.ask_x0[0])
    q_b = int(round(sigmoid_action.bid_L[0]))
    q_a = int(round(sigmoid_action.ask_L[0]))
    return Action(
        delta_b_ticks=max(0.0, delta_b),
        delta_a_ticks=max(0.0, delta_a),
        q_b=max(0, q_b),
        q_a=max(0, q_a),
    )


def _evaluate_cumulative_depth_levels(spreads: list[float], depth_fn) -> list[float]:
    """Step 1: evaluate monotone cumulative depth D(s) on spread grid."""
    cumulative: list[float] = []
    cum_prev = 0.0
    for s in spreads:
        cum = depth_fn(s)
        if not math.isfinite(cum):
            cum = cum_prev
        cum = max(cum, cum_prev)
        cumulative.append(cum)
        cum_prev = cum
    return cumulative


def _marginal_from_cumulative_levels(cumulative: list[float]) -> list[float]:
    """Step 2: convert cumulative levels to marginal levels with D(0)=0 baseline."""
    marginal_raw: list[float] = []
    prev_cum = 0.0
    for cum in cumulative:
        raw_qty = cum - prev_cum
        if not math.isfinite(raw_qty) or raw_qty < 0:
            raw_qty = 0.0
        marginal_raw.append(raw_qty)
        prev_cum = cum
    return marginal_raw


def _quantize_marginal_levels(
    spreads: list[float],
    marginal_raw: list[float],
    lot_size: int,
) -> list[tuple[float, int]]:
    """Step 3: quantize raw marginals to lot_size and keep positive levels."""
    out: list[tuple[float, int]] = []
    for s, raw_qty in zip(spreads, marginal_raw):
        qty = int(round(raw_qty / lot_size)) * lot_size
        if qty > 0:
            out.append((s, qty))
    return out


def sigmoid_action_to_discretized_action(
    sigmoid_action: SigmoidMixtureAction,
    s_b_min: float = 1.0,
    s_b_max: float = 100.0,
    s_a_min: float = 1.0,
    s_a_max: float = 100.0,
    lot_size: int = 1,
) -> DiscretizedOrderAction:
    """
    Convert SigmoidMixtureAction to a discrete ladder of (spread_ticks, qty) per side
    so that cumulative volume at each level approximates the sigmoid mixture curves.
    Bid grid: s_b_min, s_b_min+1, ..., s_b_max (step 1). Ask grid: s_a_min, ..., s_a_max (step 1).
    Quantities discretized to lot_size. Each (spread_ticks, qty) is MARGINAL; cumulative
    at spread s is the sum of qty over levels with spread <= s (monotonic by construction).

    This function is model-space only: exchange-level minimum order constraints
    (min notional, min qty) are enforced at execution time.
    """
    lot_size = max(1, lot_size)
    # Bid spread levels: step size 1 (tick)
    b_lo, b_hi = int(round(s_b_min)), int(round(s_b_max))
    bid_spreads = [float(i) for i in range(b_lo, b_hi + 1)] if b_hi >= b_lo else []
    # Ask spread levels: step size 1 (tick)
    a_lo, a_hi = int(round(s_a_min)), int(round(s_a_max))
    ask_spreads = [float(i) for i in range(a_lo, a_hi + 1)] if a_hi >= a_lo else []

    bid_cumulative = _evaluate_cumulative_depth_levels(bid_spreads, sigmoid_action.depth_bid)
    ask_cumulative = _evaluate_cumulative_depth_levels(ask_spreads, sigmoid_action.depth_ask)
    bid_marginal = _marginal_from_cumulative_levels(bid_cumulative)
    ask_marginal = _marginal_from_cumulative_levels(ask_cumulative)
    bids = tuple(_quantize_marginal_levels(bid_spreads, bid_marginal, lot_size))
    asks = tuple(_quantize_marginal_levels(ask_spreads, ask_marginal, lot_size))
    return DiscretizedOrderAction(bids=bids, asks=asks)


def sigmoid_mixture_action_bounds(
    n_components: int = 2,
    L_max: float = 1e6,
    k_max: float = 100.0,
    x0_max: float = 500.0,
) -> list[tuple[float, float]]:
    """
    Default (low, high) bounds for (6 * n_components)-dim SigmoidMixtureAction vector.
    Order: bid_L1, bid_k1, bid_x0_1, ...; ask_...
    L in (0.01, L_max), k in (0.01, k_max), x0 in (0, x0_max).
    """
    n_components = max(1, n_components)
    one = (1e-6, L_max)
    k_bound = (1e-6, k_max)  # k must be positive for sigmoid steepness
    x0_bound = (-x0_max, x0_max)
    return [one, k_bound, x0_bound] * (2 * n_components)


@dataclass
class State:
    """
    Full state s_t = [mid, ob_rel, buy_vol, sell_vol, inventory, cash].
    - mid: mid price (e.g. in ticks).
    - ob_rel: orderbook relative to mid (for depth/fill computation).
    - buy_vol, sell_vol: realized or sampled market buy/sell volume in the period.
    - inventory: integer (lots).
    - cash: in (tick_size × lot_size) units.
    - buy_ref, sell_ref: representative ask/bid prices in ticks (from current book; no EWMA).
    """
    mid: float
    ob_rel: OrderbookRelative
    buy_vol: float
    sell_vol: float
    inventory: int
    cash: float = 0.0
    buy_ref: float = 0.0
    sell_ref: float = 0.0

    @property
    def x(self) -> MarketState:
        """Market part for backward compatibility (e.g. reward, price_dynamics)."""
        return MarketState(mid=self.mid)

def state_from_ob_and_flow(
    mid: float,
    ob_bids: list[tuple[float, int]],
    ob_asks: list[tuple[float, int]],
    buy_vol: float,
    sell_vol: float,
    inventory: int,
    cash: float = 0.0,
) -> State:
    """Build State from mid, orderbook levels (spread_ticks, qty), flow, and position."""
    ob_rel = OrderbookRelative(
        bids=tuple(tuple(x) for x in ob_bids),
        asks=tuple(tuple(x) for x in ob_asks),
    )
    return State(
        mid=mid,
        ob_rel=ob_rel,
        buy_vol=buy_vol,
        sell_vol=sell_vol,
        inventory=inventory,
        cash=cash,
        buy_ref=mid,
        sell_ref=mid,
    )


def ob_to_ob_rel(ob: Orderbook, spread_max: float | None = None) -> OrderbookRelative:
    """
    Build OrderbookRelative from trade_stats Orderbook.
    If spread_max is provided, only include levels with spread_ticks <= spread_max.
    """
    if spread_max is not None:
        bids = tuple((lev.spread_ticks, lev.qty) for lev in ob.bids if lev.spread_ticks <= spread_max)
        asks = tuple((lev.spread_ticks, lev.qty) for lev in ob.asks if lev.spread_ticks <= spread_max)
    else:
        bids = tuple((lev.spread_ticks, lev.qty) for lev in ob.bids)
        asks = tuple((lev.spread_ticks, lev.qty) for lev in ob.asks)
    return OrderbookRelative(bids=bids, asks=asks)


def state_from_orderbook(
    ob: Orderbook,
    buy_vol: float,
    sell_vol: float,
    inventory: int,
    cash: float = 0.0,
    spread_max: float | None = None,
    reference_depth_lots: float = 1000.0,
) -> State:
    """
    Build State from trade_stats Orderbook and flow/position.
    If spread_max is provided, only include orderbook levels with spread_ticks <= spread_max.
    """
    mid = float(ob.mid)
    ob_rel = ob_to_ob_rel(ob, spread_max=spread_max)
    buy_ref, sell_ref = reference_prices_ticks_from_orderbook(ob, target_depth_lots=reference_depth_lots)
    return State(
        mid=mid,
        ob_rel=ob_rel,
        buy_vol=buy_vol,
        sell_vol=sell_vol,
        inventory=inventory,
        cash=cash,
        buy_ref=buy_ref,
        sell_ref=sell_ref,
    )


def state_from_step(
    step: tuple,
    *,
    spread_max: float | None = None,
    reference_depth_lots: float = 1000.0,
) -> State:
    """
    Build State directly from a Step tuple:
    (t_end_ms, ob, buy_vol, sell_vol, inventory, cash, ...).
    """
    _t_end, ob, buy_vol, sell_vol, inventory, cash, *_ = step
    return state_from_orderbook(
        ob=ob,
        buy_vol=float(buy_vol),
        sell_vol=float(sell_vol),
        inventory=int(inventory),
        cash=float(cash),
        spread_max=spread_max,
        reference_depth_lots=reference_depth_lots,
    )
