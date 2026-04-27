"""
Execution model: depth D(δ), fills f_b/f_a from order flow and orderbook;
(state, action) → next state. Runner uses state_after_step; sets _last_dW, _last_pnl_decomp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from trade_stats import Orderbook

from .flow_implied import Step
from ..state import Action, DiscretizedOrderAction, OrderbookRelative, State, ob_to_ob_rel


# -----------------------------------------------------------------------------
# Cumulative depth vs spread; depth from sigmoid params (for animation)
# -----------------------------------------------------------------------------


def cumulative_depth_curves(
    ob_rel: OrderbookRelative,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Cumulative bid/ask volume vs spread from orderbook relative. Returns ((bid_spreads, bid_cumvol), (ask_spreads, ask_cumvol))."""
    bid_levels = sorted(ob_rel.bids, key=lambda t: t[0])
    if not bid_levels:
        bid_spreads = np.array([0.0])
        bid_cumvol = np.array([0.0])
    else:
        bid_spreads = np.array([float(t[0]) for t in bid_levels], dtype=np.float64)
        bid_cumvol = np.cumsum(np.array([float(t[1]) for t in bid_levels], dtype=np.float64))
    ask_levels = sorted(ob_rel.asks, key=lambda t: t[0])
    if not ask_levels:
        ask_spreads = np.array([0.0])
        ask_cumvol = np.array([0.0])
    else:
        ask_spreads = np.array([float(t[0]) for t in ask_levels], dtype=np.float64)
        ask_cumvol = np.cumsum(np.array([float(t[1]) for t in ask_levels], dtype=np.float64))
    return (bid_spreads, bid_cumvol), (ask_spreads, ask_cumvol)


def depth_from_sigmoid_params(params: dict[str, list[float]], delta: float) -> float:
    """Cumulative depth at spread δ from fitted sigmoid params. params = {"L": [...], "k": [...], "x0": [...]}."""
    L_list = params.get("L", [])
    k_list = params.get("k", [])
    x0_list = params.get("x0", [])
    n = min(len(L_list), len(k_list), len(x0_list))
    total = 0.0
    for i in range(n):
        total += L_list[i] / (1.0 + np.exp(-k_list[i] * (delta - x0_list[i])))
    return total


# -----------------------------------------------------------------------------
# Execution rule: depth and fills (OB or OB-relative)
# -----------------------------------------------------------------------------


@runtime_checkable
class ExecutionRule(Protocol):
    """Compute depth and fills given OB, action, and flow (V_plus, V_minus)."""

    def depth_bid(self, ob: Orderbook, delta_b_ticks: float) -> float:
        ...

    def depth_ask(self, ob: Orderbook, delta_a_ticks: float) -> float:
        ...

    def fills(
        self,
        ob: Orderbook,
        action: Action,
        V_minus: int,
        V_plus: int,
    ) -> tuple[int, int]:
        ...


def depth_bid_rel(ob_rel: OrderbookRelative, delta_b_ticks: float) -> float:
    """Depth on bid side: sum of qty where spread_ticks <= delta_b."""
    return sum(float(q) for (st, q) in ob_rel.bids if st <= delta_b_ticks)


def depth_ask_rel(ob_rel: OrderbookRelative, delta_a_ticks: float) -> float:
    """Depth on ask side: sum of qty where spread_ticks <= delta_a."""
    return sum(float(q) for (st, q) in ob_rel.asks if st <= delta_a_ticks)


def fills_from_ob_rel(
    ob_rel: OrderbookRelative,
    action: Action,
    V_minus: int,
    V_plus: int,
) -> tuple[int, int]:
    """f_b = min(q_b, max(0, V_minus - D_b)), f_a = min(q_a, max(0, V_plus - D_a))."""
    D_b = depth_bid_rel(ob_rel, action.delta_b_ticks)
    D_a = depth_ask_rel(ob_rel, action.delta_a_ticks)
    reach_b = max(0, V_minus - int(round(D_b)))
    reach_a = max(0, V_plus - int(round(D_a)))
    f_b = min(action.q_b, reach_b)
    f_a = min(action.q_a, reach_a)
    return (f_b, f_a)


def fills_from_ob_rel_discretized(
    ob_rel: OrderbookRelative,
    action: DiscretizedOrderAction,
    V_minus: int,
    V_plus: int,
) -> tuple[int, int, list[tuple[float, int]], list[tuple[float, int]]]:
    """Fills for discrete ladder. Returns (f_b_total, f_a_total, fills_bid, fills_ask)."""
    f_b_total = 0
    fills_bid: list[tuple[float, int]] = []
    our_cum_b = 0
    for (delta_b, q_b) in sorted(action.bids, key=lambda x: x[0]):
        D_b = depth_bid_rel(ob_rel, delta_b)
        depth_ahead = D_b + our_cum_b
        reach = max(0, V_minus - int(round(depth_ahead)))
        fill_b = min(q_b, reach)
        fills_bid.append((delta_b, fill_b))
        f_b_total += fill_b
        our_cum_b += q_b

    f_a_total = 0
    fills_ask: list[tuple[float, int]] = []
    our_cum_a = 0
    for (delta_a, q_a) in sorted(action.asks, key=lambda x: x[0]):
        D_a = depth_ask_rel(ob_rel, delta_a)
        depth_ahead = D_a + our_cum_a
        reach = max(0, V_plus - int(round(depth_ahead)))
        fill_a = min(q_a, reach)
        fills_ask.append((delta_a, fill_a))
        f_a_total += fill_a
        our_cum_a += q_a

    return (f_b_total, f_a_total, fills_bid, fills_ask)


class DefaultExecutionRule:
    """D_t^b(δ), D_t^a(δ) from OB; fills f_b, f_a from flow and depth."""

    def depth_bid(self, ob: Orderbook, delta_b_ticks: float) -> float:
        total = 0.0
        for lev in ob.bids:
            if lev.spread_ticks <= delta_b_ticks:
                total += float(lev.qty)
        return total

    def depth_ask(self, ob: Orderbook, delta_a_ticks: float) -> float:
        total = 0.0
        for lev in ob.asks:
            if lev.spread_ticks <= delta_a_ticks:
                total += float(lev.qty)
        return total

    def fills(
        self,
        ob: Orderbook,
        action: Action,
        V_minus: int,
        V_plus: int,
    ) -> tuple[int, int]:
        D_b = self.depth_bid(ob, action.delta_b_ticks)
        D_a = self.depth_ask(ob, action.delta_a_ticks)
        reach_b = max(0, V_minus - int(round(D_b)))
        reach_a = max(0, V_plus - int(round(D_a)))
        f_b = min(action.q_b, reach_b)
        f_a = min(action.q_a, reach_a)
        return (f_b, f_a)


# -----------------------------------------------------------------------------
# Execution model: (state, action) → next state
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DeltaPnL:
    """One-step realized pnl decomposition returned by execution model."""
    spread_pnl: float
    inventory_pnl: float
    cross_pnl: float
    fee_pnl: float
    wealth_pnl: float


@runtime_checkable
class ExecutionModel(Protocol):
    """Protocol for execution: apply action on current step and return state + pnl."""

    def state_after_step(
        self,
        prev_state: State,
        prev_action: DiscretizedOrderAction,
        curr_step: Step,
        spread_max: float | None = None,
    ) -> tuple[State, DeltaPnL]:
        ...


class DiscretizedExecutionModel:
    """
    Execution model for DiscretizedOrderAction (per-level fills).
    Returns (next_state, delta_pnl). Optional maker_fee in [0, 1).
    """

    def __init__(
        self,
        order_flow_model=None,
        price_model=None,
        maker_fee: float = 0.0,
    ):
        self.maker_fee = max(0.0, min(1.0 - 1e-9, float(maker_fee)))

    def state_after_step(
        self,
        prev_state: State,
        prev_action: DiscretizedOrderAction,
        curr_step: Step,
        spread_max: float | None = None,
    ) -> tuple[State, DeltaPnL]:
        """Apply prev_action with curr_step volumes and return next state + one-step pnl."""
        _t_end, ob_curr, buy_vol, sell_vol, _inventory, _cash, *_rest = curr_step
        anchor_buy_ref = None
        anchor_sell_ref = None
        anchor_mu_delta_buy = None
        anchor_sigma_delta_buy = None
        anchor_mu_delta_sell = None
        anchor_sigma_delta_sell = None
        if len(_rest) >= 5:
            anchor_buy_ref = _rest[3]
            anchor_sell_ref = _rest[4]
        if len(_rest) >= 9:
            anchor_mu_delta_buy = _rest[5]
            anchor_sigma_delta_buy = _rest[6]
            anchor_mu_delta_sell = _rest[7]
            anchor_sigma_delta_sell = _rest[8]
        if not isinstance(prev_action, DiscretizedOrderAction):
            raise TypeError(
                "DiscretizedExecutionModel.state_after_step expects DiscretizedOrderAction"
            )
        mid_curr = float(ob_curr.mid)
        V_plus = int(round(buy_vol))
        V_minus = int(round(sell_vol))
        mid_cur = prev_state.mid
        S_t = mid_cur
        S_t1 = mid_curr
        q_t = prev_state.inventory
        q_anchor_t = int(getattr(prev_state, "anchor_inventory", 0))
        c_anchor_t = float(getattr(prev_state, "anchor_cash", 0.0))
        one_minus_f = 1.0 - self.maker_fee

        f_b, f_a, fills_bid, fills_ask = fills_from_ob_rel_discretized(
            prev_state.ob_rel, prev_action, V_minus, V_plus
        )
        inv_new = prev_state.inventory + f_b * one_minus_f - f_a
        ask_notional = sum(fa * (mid_cur + da) for (da, fa) in fills_ask)
        bid_notional = sum(fb * (mid_cur - db) for (db, fb) in fills_bid)
        cash_delta = ask_notional * one_minus_f - bid_notional
        cash_new = prev_state.cash + cash_delta
        spread_pnl = sum(db * fb for (db, fb) in fills_bid) + sum(da * fa for (da, fa) in fills_ask)
        delta_q = f_b * one_minus_f - f_a
        inventory_pnl = q_t * (S_t1 - S_t)
        cross_pnl = delta_q * (S_t1 - S_t)
        # Fee is charged on received assets:
        # - ask fills: quote fee = maker_fee * ask_notional
        # - bid fills: base fee; converted to quote-like units at bid execution prices.
        total_fee_paid = self.maker_fee * (ask_notional + bid_notional)
        fee_pnl = -float(total_fee_paid)

        anchor_ref_prev = (
            0.5
            * float(
                (prev_state.anchor_buy_ref if prev_state.anchor_buy_ref is not None else prev_state.buy_ref)
                + (prev_state.anchor_sell_ref if prev_state.anchor_sell_ref is not None else prev_state.sell_ref)
            )
        )
        anchor_ref_next = (
            0.5
            * float(
                (anchor_buy_ref if anchor_buy_ref is not None else prev_state.buy_ref)
                + (anchor_sell_ref if anchor_sell_ref is not None else prev_state.sell_ref)
            )
        )
        wealth_before = prev_state.cash + prev_state.inventory * prev_state.mid + c_anchor_t + q_anchor_t * anchor_ref_prev
        wealth_after = cash_new + inv_new * mid_curr + c_anchor_t + q_anchor_t * anchor_ref_next
        wealth_pnl = wealth_after - wealth_before
        ob_rel = ob_to_ob_rel(ob_curr, spread_max=spread_max)
        next_state = State(
            mid=mid_curr,
            ob_rel=ob_rel,
            buy_vol=float(buy_vol),
            sell_vol=float(sell_vol),
            inventory=inv_new,
            cash=cash_new,
            anchor_buy_ref=float(anchor_buy_ref) if anchor_buy_ref is not None else None,
            anchor_sell_ref=float(anchor_sell_ref) if anchor_sell_ref is not None else None,
            anchor_mu_delta_buy=float(anchor_mu_delta_buy) if anchor_mu_delta_buy is not None else None,
            anchor_sigma_delta_buy=float(anchor_sigma_delta_buy) if anchor_sigma_delta_buy is not None else None,
            anchor_mu_delta_sell=float(anchor_mu_delta_sell) if anchor_mu_delta_sell is not None else None,
            anchor_sigma_delta_sell=float(anchor_sigma_delta_sell) if anchor_sigma_delta_sell is not None else None,
            anchor_inventory=q_anchor_t,
            anchor_cash=c_anchor_t,
        )
        return next_state, DeltaPnL(
            spread_pnl=float(spread_pnl),
            inventory_pnl=float(inventory_pnl),
            cross_pnl=float(cross_pnl),
            fee_pnl=fee_pnl,
            wealth_pnl=float(wealth_pnl),
        )


class ObservedStepExecutionModel:
    """
    Execution path for live / replay when inventory and cash come from the stepper
    (balances + private fills), not from simulated OB-relative fills. ``prev_action`` is
    ignored for state; PnL decomposition is chosen so components sum to ``wealth_pnl``.
    """

    def state_after_step(
        self,
        prev_state: State,
        prev_action: DiscretizedOrderAction,
        curr_step: Step,
        spread_max: float | None = None,
    ) -> tuple[State, DeltaPnL]:
        _t_end, ob_curr, buy_vol, sell_vol, inv_new, cash_new, private_trade_fee_tick_lot, *_rest = curr_step
        anchor_buy_ref = None
        anchor_sell_ref = None
        anchor_mu_delta_buy = None
        anchor_sigma_delta_buy = None
        anchor_mu_delta_sell = None
        anchor_sigma_delta_sell = None
        anchor_inventory = None
        anchor_cash = None
        if len(_rest) >= 4:
            anchor_buy_ref = _rest[2]
            anchor_sell_ref = _rest[3]
        if len(_rest) >= 8:
            anchor_mu_delta_buy = _rest[4]
            anchor_sigma_delta_buy = _rest[5]
            anchor_mu_delta_sell = _rest[6]
            anchor_sigma_delta_sell = _rest[7]
        if len(_rest) >= 10:
            anchor_inventory = _rest[8]
            anchor_cash = _rest[9]
        mid_curr = float(ob_curr.mid)
        ob_rel = ob_to_ob_rel(ob_curr, spread_max=spread_max)
        cash_f = float(cash_new)
        inv_i = int(inv_new)
        anchor_inv_i = int(anchor_inventory) if anchor_inventory is not None else int(getattr(prev_state, "anchor_inventory", 0))
        anchor_cash_f = float(anchor_cash) if anchor_cash is not None else float(getattr(prev_state, "anchor_cash", 0.0))
        anchor_ref_prev = (
            0.5
            * float(
                (prev_state.anchor_buy_ref if prev_state.anchor_buy_ref is not None else prev_state.buy_ref)
                + (prev_state.anchor_sell_ref if prev_state.anchor_sell_ref is not None else prev_state.sell_ref)
            )
        )
        anchor_ref_next = (
            0.5
            * float(
                (anchor_buy_ref if anchor_buy_ref is not None else prev_state.buy_ref)
                + (anchor_sell_ref if anchor_sell_ref is not None else prev_state.sell_ref)
            )
        )
        main_wealth_before = prev_state.cash + prev_state.inventory * prev_state.mid
        main_wealth_after = cash_f + inv_i * mid_curr
        main_wealth_pnl = main_wealth_after - main_wealth_before
        wealth_before = (
            main_wealth_before
            + float(getattr(prev_state, "anchor_cash", 0.0))
            + int(getattr(prev_state, "anchor_inventory", 0)) * anchor_ref_prev
        )
        wealth_after = main_wealth_after + anchor_cash_f + anchor_inv_i * anchor_ref_next
        wealth_pnl = wealth_after - wealth_before
        inventory_pnl = prev_state.inventory * (mid_curr - prev_state.mid)
        delta_inv = inv_i - prev_state.inventory
        cross_pnl = delta_inv * (mid_curr - prev_state.mid)
        # Exchange-reported trade fees from stepper are in tick*lot units (same as cash).
        fee_pnl = -float(private_trade_fee_tick_lot)
        # Spread decomposition is for main market only (not total portfolio wealth).
        spread_pnl = main_wealth_pnl - inventory_pnl - cross_pnl - fee_pnl
        next_state = State(
            mid=mid_curr,
            ob_rel=ob_rel,
            buy_vol=float(buy_vol),
            sell_vol=float(sell_vol),
            inventory=inv_i,
            cash=cash_f,
            anchor_buy_ref=float(anchor_buy_ref) if anchor_buy_ref is not None else None,
            anchor_sell_ref=float(anchor_sell_ref) if anchor_sell_ref is not None else None,
            anchor_mu_delta_buy=float(anchor_mu_delta_buy) if anchor_mu_delta_buy is not None else None,
            anchor_sigma_delta_buy=float(anchor_sigma_delta_buy) if anchor_sigma_delta_buy is not None else None,
            anchor_mu_delta_sell=float(anchor_mu_delta_sell) if anchor_mu_delta_sell is not None else None,
            anchor_sigma_delta_sell=float(anchor_sigma_delta_sell) if anchor_sigma_delta_sell is not None else None,
            anchor_inventory=anchor_inv_i,
            anchor_cash=anchor_cash_f,
        )
        return next_state, DeltaPnL(
            spread_pnl=float(spread_pnl),
            inventory_pnl=float(inventory_pnl),
            cross_pnl=float(cross_pnl),
            fee_pnl=fee_pnl,
            wealth_pnl=float(wealth_pnl),
        )
