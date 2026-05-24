"""
Execution model: map prior controller ``State`` + new market ``Step`` → next ``State`` and ΔPnL.

Depth/fill helpers above remain for MPC / ``ExecutionRule``; the runtime step uses
``ObservedStepExecutionModel`` only (balances and refs from the step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from mm.integrations.trade_stats import Orderbook

from mm.mpc.step import Step
from mm.core.orderflow_state import Action, OrderbookRelative, State, ob_to_ob_rel


# -----------------------------------------------------------------------------
# Cumulative depth vs spread; depth from sigmoid params (MPC visualization helpers)
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
# Execution model: prior state + step → next state + PnL
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
    """
    Protocol for execution: map previous state + new market step to state + pnl.

    ``prev_action`` is for simulated / OB-fill models. Implementations that read
    inventory and cash from the step (e.g. ``ObservedStepExecutionModel``) may ignore it.
    """

    def state_after_step(
        self,
        prev_state: State,
        prev_action: Action | None,
        curr_step: Step,
        spread_max: float | None = None,
    ) -> tuple[State, DeltaPnL]:
        ...


class ObservedStepExecutionModel:
    """
    Live / replay: inventory, cash, and refs come from the ``Step`` (exchange path).
    PnL decomposition is chosen so components sum to ``wealth_pnl``.

    ``maker_fee`` and ``hedge_fee`` are not applied inside ``state_after_step`` (observed
    fees come from the step). They are read by MPC rollouts (e.g.
    ``MonteCarloMPCTensorBatched``) like ``maker_fee`` for main-venue simulation and
    ``hedge_fee`` for proportional fee on simulated hedge-venue trades.
    """

    def __init__(self, maker_fee: float = 0.0, hedge_fee: float | None = None):
        # Kept for compatibility: MPC reads execution_model.maker_fee / hedge_fee.
        self.maker_fee = max(0.0, min(1.0 - 1e-9, float(maker_fee)))
        if hedge_fee is not None:
            self.global__fee = max(0.0, min(1.0 - 1e-9, float(hedge_fee)))

    def state_after_step(
        self,
        prev_state: State,
        curr_step: Step,
        spread_max: float | None = None,
    ) -> tuple[State, DeltaPnL]:
        (
            _t_end,
            ob_curr,
            buy_vol,
            sell_vol,
            inv_new,
            cash_new,
            private_trade_fee_tick_lot,
            _private_fills_bid,
            _private_fills_ask,
            main_buy_ref,
            main_sell_ref,
            hedge_buy_ref,
            hedge_sell_ref,
            global_inventory,
            hedge_cash,
        ) = curr_step
        mid_curr = float(ob_curr.mid)
        ob_rel = ob_to_ob_rel(ob_curr, spread_max=spread_max)
        cash_f = float(cash_new)
        inv_i = int(inv_new)
        hedge_inv_i = int(global_inventory)
        hedge_cash_f = float(hedge_cash)

        # Option B: liquidation-style mark-to-bid (sell-side representative price).
        main_ref_prev = float(prev_state.sell_ref)
        main_ref_next = float(main_sell_ref)
        hedge_ref_prev = float(
            prev_state.hedge_sell_ref
            if prev_state.hedge_sell_ref is not None
            else prev_state.sell_ref
        )
        hedge_ref_next = float(
            hedge_sell_ref
            if hedge_sell_ref is not None
            else main_ref_next
        )

        main_wealth_before = prev_state.cash + prev_state.inventory * main_ref_prev
        main_wealth_after = cash_f + inv_i * main_ref_next
        main_wealth_pnl = main_wealth_after - main_wealth_before
        wealth_before = (
            main_wealth_before
            + float(getattr(prev_state, "hedge_cash", 0.0))
            + int(getattr(prev_state, "global_inventory", 0)) * hedge_ref_prev
        )
        wealth_after = main_wealth_after + hedge_cash_f + hedge_inv_i * hedge_ref_next
        wealth_pnl = wealth_after - wealth_before
        inventory_pnl = prev_state.inventory * (main_ref_next - main_ref_prev)
        delta_inv = inv_i - prev_state.inventory
        cross_pnl = delta_inv * (main_ref_next - main_ref_prev)
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
            buy_ref=float(main_buy_ref),
            sell_ref=float(main_sell_ref),
            hedge_buy_ref=float(hedge_buy_ref) if hedge_buy_ref is not None else None,
            hedge_sell_ref=float(hedge_sell_ref) if hedge_sell_ref is not None else None,
            global_inventory=hedge_inv_i,
            hedge_cash=hedge_cash_f,
        )
        return next_state, DeltaPnL(
            spread_pnl=float(spread_pnl),
            inventory_pnl=float(inventory_pnl),
            cross_pnl=float(cross_pnl),
            fee_pnl=fee_pnl,
            wealth_pnl=float(wealth_pnl),
        )
