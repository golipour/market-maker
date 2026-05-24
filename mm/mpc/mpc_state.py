"""MPC state and sigmoid action types (decoupled from trade_stats / legacy state module)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderbookRelative:
    """
    Orderbook in relative terms: (quote-price spread, base qty) per level.
    Bids: spread = mid - price (positive = below mid).
    Asks: spread = price - mid (positive = above mid).
    """

    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Action:
    """Legacy single-level action kept for compatibility."""

    delta_b_ticks: float
    delta_a_ticks: float
    q_b: int
    q_a: int


def _sigmoid_stable(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    return math.exp(z) / (1.0 + math.exp(z))


@dataclass(frozen=True)
class SigmoidMixtureAction:
    """Raw mixture-of-sigmoids cumulative depth parameterization for batched MPC.

    ``depth_bid`` / ``depth_ask`` return raw model depth ``D(spread)``. The
    executable quote depth is ``D(spread) - D(0)``; use
    ``effective_depth_bid`` / ``effective_depth_ask`` when building orders or
    diagnostics from this raw action.
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
        n = self.n_components
        out: list[float] = []
        for i in range(n):
            out.extend([self.bid_L[i], self.bid_k[i], self.bid_x0[i]])
        for i in range(n):
            out.extend([self.ask_L[i], self.ask_k[i], self.ask_x0[i]])
        return tuple(out)

    @classmethod
    def from_flat(cls, x: tuple[float, ...] | list[float], n_components: int | None = None) -> SigmoidMixtureAction:
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
        x = list(x)
        if n_components is None:
            n_components = len(x) // 6
        if len(x) < 6 * n_components:
            raise ValueError(
                f"SigmoidMixtureAction.from_flat_positive requires at least {6 * n_components} elements, got {len(x)}"
            )
        eps = 1e-8

        def _pos(v: float) -> float:
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

    def depth_bid(self, spread: float) -> float:
        out = 0.0
        for i in range(self.n_components):
            z = self.bid_k[i] * (spread - self.bid_x0[i])
            s = _sigmoid_stable(z)
            out += self.bid_L[i] * s
        return out

    def depth_ask(self, spread: float) -> float:
        out = 0.0
        for i in range(self.n_components):
            z = self.ask_k[i] * (spread - self.ask_x0[i])
            s = _sigmoid_stable(z)
            out += self.ask_L[i] * s
        return out

    def effective_depth_bid(self, spread: float) -> float:
        """Executable cumulative bid depth: ``max(0, D_bid(spread) - D_bid(0))``."""

        return max(0.0, self.depth_bid(spread) - self.depth_bid(0.0))

    def effective_depth_ask(self, spread: float) -> float:
        """Executable cumulative ask depth: ``max(0, D_ask(spread) - D_ask(0))``."""

        return max(0.0, self.depth_ask(spread) - self.depth_ask(0.0))

    def as_effective(self) -> EffectiveSigmoidMixtureAction:
        """Return a view whose ``depth_*`` methods expose executable depth."""

        return EffectiveSigmoidMixtureAction(
            bid_L=self.bid_L,
            bid_k=self.bid_k,
            bid_x0=self.bid_x0,
            ask_L=self.ask_L,
            ask_k=self.ask_k,
            ask_x0=self.ask_x0,
        )

    def depth_bid_derivative(self, spread: float) -> float:
        out = 0.0
        for i in range(self.n_components):
            z = self.bid_k[i] * (spread - self.bid_x0[i])
            s = _sigmoid_stable(z)
            out += self.bid_L[i] * self.bid_k[i] * s * (1.0 - s)
        return out

    def depth_ask_derivative(self, spread: float) -> float:
        out = 0.0
        for i in range(self.n_components):
            z = self.ask_k[i] * (spread - self.ask_x0[i])
            s = _sigmoid_stable(z)
            out += self.ask_L[i] * self.ask_k[i] * s * (1.0 - s)
        return out


@dataclass(frozen=True)
class EffectiveSigmoidMixtureAction(SigmoidMixtureAction):
    """Sigmoid action view where ``depth_*`` returns executable ``D(spread) - D(0)``.

    The flat parameters remain the original raw sigmoid-mixture parameters, so this
    can still be used for warm starts and diagnostics requiring the underlying model.
    """

    def raw_depth_bid(self, spread: float) -> float:
        return super().depth_bid(spread)

    def raw_depth_ask(self, spread: float) -> float:
        return super().depth_ask(spread)

    def depth_bid(self, spread: float) -> float:
        return max(0.0, self.raw_depth_bid(spread) - self.raw_depth_bid(0.0))

    def depth_ask(self, spread: float) -> float:
        return max(0.0, self.raw_depth_ask(spread) - self.raw_depth_ask(0.0))

    def effective_depth_bid(self, spread: float) -> float:
        return self.depth_bid(spread)

    def effective_depth_ask(self, spread: float) -> float:
        return self.depth_ask(spread)

    def as_effective(self) -> EffectiveSigmoidMixtureAction:
        return self


def sigmoid_mixture_action_bounds(
    n_components: int = 2,
    L_max: float = 1e6,
    k_max: float = 100.0,
    x0_max: float = 500.0,
) -> list[tuple[float, float]]:
    n_components = max(1, int(n_components))
    one = (1e-6, L_max)
    k_bound = (1e-6, k_max)
    x0_bound = (-x0_max, x0_max)
    return [one, k_bound, x0_bound] * (2 * n_components)


@dataclass
class State:
    """Full MPC state passed to the tensor batched optimizer."""

    mid: float
    ob_rel: OrderbookRelative
    buy_vol: float
    sell_vol: float
    inventory: float
    cash: float = 0.0
    buy_vwap: float = 0.0
    sell_vwap: float = 0.0
    hedge_buy_vwap: float | None = None
    hedge_sell_vwap: float | None = None
    hedge_inventory: float = 0.0
    hedge_cash: float = 0.0
    global_ob_rel: OrderbookRelative | None = None
    global_buy_vol: float = 0.0
    global_sell_vol: float = 0.0
    global_inventory: float = 0.0
    global_cash: float = 0.0
    global_buy_vwap: float = 0.0
    global_sell_vwap: float = 0.0
    reference_price: float = 0.0
    global_active: bool = False
