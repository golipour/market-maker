"""Net-delta risk engine: approve or reject target-book transitions against ledger state."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from mm.core.adapters.joint import joint_target_book_to_action
from mm.core.interfaces.risk import AbstractRiskEngine
from mm.core.models.execution import Action, TradingState
from mm.core.models.instrument import SymbolBinding
from mm.core.models.ledger import JointLedgerSnapshot, VenueLedger
from mm.core.models.market import MarketEvent
from mm.core.models.private import OpenOrder
from mm.core.models.strategy import StrategyInputFrame
from mm.core.models.target_book import JointTargetBook, TargetOrder, VenueTargetBook
from mm.core.risk_limits import HedgeVenueRiskLimits, VenueRiskLimits
from mm.core.wallet import order_collateral_from_open_orders
from mm.execution.plan import RiskResult
from mm.observability import counter_inc, gauge
from mm.risk.engine import RiskLimits, _proposed_order_count
from mm.risk.models import TargetBookRiskResult, VenueTransitionExposure
from mm.runtime.config import LiquiditySourcingRuntimeConfig


def _scope_actor_live_orders(
    live_orders: dict[str, OpenOrder],
    *,
    actor_id: str,
    exchange: str,
    symbol: str,
) -> dict[str, OpenOrder]:
    ex = exchange.lower().strip()
    sym = symbol.strip()
    actor = actor_id.strip()
    scoped: dict[str, OpenOrder] = {}
    for order_id, open_order in live_orders.items():
        if open_order.actor_id.strip() != actor:
            continue
        order = open_order.order
        if order.exchange.lower().strip() != ex:
            continue
        if order.symbol.strip() != sym:
            continue
        scoped[order_id] = open_order
    return scoped


def _collateral_from_target_orders(orders: tuple[TargetOrder, ...]) -> tuple[float, float]:
    base_locked = 0.0
    quote_locked = 0.0
    for order in orders:
        qty = max(0.0, float(order.qty))
        if qty <= 0.0:
            continue
        side = str(order.side).lower().strip()
        if side == "sell":
            base_locked += qty
        elif side == "buy":
            quote_locked += max(0.0, float(order.price)) * qty
    return base_locked, quote_locked


def _open_notional(orders: tuple[TargetOrder, ...]) -> float:
    total = 0.0
    for order in orders:
        qty = max(0.0, float(order.qty))
        if qty <= 0.0:
            continue
        total += max(0.0, float(order.price)) * qty
    return total


def _hedge_limits_as_venue_limits(limits: HedgeVenueRiskLimits) -> VenueRiskLimits:
    return VenueRiskLimits(
        min_inventory=limits.min_inventory,
        max_inventory=limits.max_inventory,
        min_cash=limits.min_cash,
        max_open_notional=float("inf"),
        max_order_notional=limits.max_market_order_notional,
        max_market_data_age_ms=limits.max_market_data_age_ms,
        max_private_state_age_ms=limits.max_private_state_age_ms,
    )


def _venue_base_quote_totals(ledger: VenueLedger) -> tuple[float, float]:
    base_total = float(ledger.available_base) + float(ledger.locked_base)
    quote_total = float(ledger.available_quote) + float(ledger.locked_quote)
    return base_total, quote_total


def evaluate_venue_target_transition(
    *,
    ledger: VenueLedger,
    target: VenueTargetBook,
    limits: VenueRiskLimits,
) -> tuple[bool, list[str], VenueTransitionExposure]:
    """Evaluate whether ``target`` resting book is feasible from ``ledger`` state."""

    actor_live = _scope_actor_live_orders(
        ledger.live_orders,
        actor_id=target.actor_id,
        exchange=target.exchange,
        symbol=target.symbol,
    )
    actor_locked_base, actor_locked_quote = order_collateral_from_open_orders(
        actor_live,
        trading_symbol=target.symbol,
        base_asset=ledger.base_asset,
        quote_asset=ledger.quote_asset,
    )
    target_locked_base, target_locked_quote = _collateral_from_target_orders(target.orders)
    open_notional = _open_notional(target.orders)
    base_total, quote_total = _venue_base_quote_totals(ledger)

    exposure = VenueTransitionExposure(
        exchange=target.exchange,
        symbol=target.symbol,
        actor_locked_base=actor_locked_base,
        actor_locked_quote=actor_locked_quote,
        target_locked_base=target_locked_base,
        target_locked_quote=target_locked_quote,
        available_base=float(ledger.available_base),
        available_quote=float(ledger.available_quote),
        open_notional=open_notional,
    )

    reasons: list[str] = []
    target_sell_qty = sum(
        max(0.0, float(order.qty))
        for order in target.orders
        if str(order.side).lower().strip() == "sell"
    )
    target_buy_qty = sum(
        max(0.0, float(order.qty))
        for order in target.orders
        if str(order.side).lower().strip() == "buy"
    )

    base_capacity = float(ledger.available_base) + actor_locked_base
    quote_capacity = float(ledger.available_quote) + actor_locked_quote

    for order in target.orders:
        qty = float(order.qty)
        if qty <= 0.0:
            reasons.append("invalid_qty")
            continue
        notional = max(0.0, float(order.price)) * qty
        if notional > limits.max_order_notional:
            reasons.append("max_order_notional")

    if open_notional > limits.max_open_notional:
        reasons.append("max_open_notional")
    if target_locked_base > base_capacity + 1e-9:
        reasons.append("insufficient_base_for_sells")
    if target_locked_quote > quote_capacity + 1e-9:
        reasons.append("insufficient_quote_for_buys")

    free_quote_after_lock = quote_total - target_locked_quote
    if free_quote_after_lock < limits.min_cash:
        reasons.append("min_cash")

    inventory_after_sells = base_total - target_sell_qty
    if inventory_after_sells < limits.min_inventory:
        reasons.append("min_inventory")

    inventory_after_buys = base_total + target_buy_qty
    if inventory_after_buys > limits.max_inventory:
        reasons.append("max_inventory")

    return not reasons, reasons, exposure


@dataclass(slots=True)
class NetDeltaRiskEngine(AbstractRiskEngine):
    """Approve or reject the full target-book transition against ledger snapshots."""

    limits: RiskLimits
    binding: SymbolBinding
    liquidity_sourcing: LiquiditySourcingRuntimeConfig
    clock_ms: Any = None
    kill_switch: bool = False

    def __post_init__(self) -> None:
        if self.clock_ms is None:
            self.clock_ms = lambda: int(time.time() * 1000)

    def set_kill_switch(self, enabled: bool) -> None:
        self.kill_switch = bool(enabled)
        gauge("mm_kill_switch_state", 1.0 if self.kill_switch else 0.0)

    def validate(self, actions: list[Action], state: TradingState, event: MarketEvent) -> list[Action]:
        return self.validate_with_result(actions, state, event).approved_list

    def validate_with_result(
        self, actions: list[Action], state: TradingState, event: MarketEvent
    ) -> RiskResult:
        frame = StrategyInputFrame(primary=event)
        return self.validate_frame_with_result(actions, state, frame)

    def validate_frame_with_result(
        self,
        actions: list[Action],
        state: TradingState,
        frame: StrategyInputFrame,
        *,
        joint_target_book: JointTargetBook | None = None,
        ledger: JointLedgerSnapshot | None = None,
        live_execution: bool = False,
    ) -> RiskResult:
        from mm.core.adapters.joint import action_to_joint_target_book, joint_ledger_from_frame

        if joint_target_book is None and actions:
            joint_target_book = action_to_joint_target_book(actions[0])
        if joint_target_book is None:
            return RiskResult(
                approved=(),
                rejection_counts={"no_target_book": 1},
                exposure_before=_legacy_exposure(state),
                exposure_after=_legacy_exposure(state),
            )

        if ledger is None:
            ledger = joint_ledger_from_frame(frame, self.binding, created_at_ms=frame.created_at_ms)

        result = self.validate_target_book_with_result(
            joint_target_book,
            ledger,
            frame,
            live_execution=live_execution,
        )
        return target_book_result_to_risk_result(result, proposed_actions=actions)

    def validate_target_book_with_result(
        self,
        target_book: JointTargetBook,
        ledger: JointLedgerSnapshot,
        frame: StrategyInputFrame,
        *,
        live_execution: bool = False,
    ) -> TargetBookRiskResult:
        order_count = _target_book_order_count(target_book)
        exposure_before = _ledger_exposure_dict(ledger)

        if self.kill_switch:
            counter_inc("mm_risk_reject_total", labels={"reason": "kill_switch"})
            return TargetBookRiskResult(
                approved=None,
                frame_gate_reasons=("kill_switch",),
                rejection_counts={"kill_switch": max(order_count, 1)},
                exposure_before=exposure_before,
                exposure_after=exposure_before,
            )

        frame_gate_reasons = self._frame_gate_reasons(
            frame,
            target_book=target_book,
            ledger=ledger,
            live_execution=live_execution,
        )
        if frame_gate_reasons:
            rejection_counts = {reason: max(order_count, 1) for reason in frame_gate_reasons}
            for reason in frame_gate_reasons:
                counter_inc("mm_risk_reject_total", labels={"reason": reason})
            return TargetBookRiskResult(
                approved=None,
                frame_gate_reasons=tuple(frame_gate_reasons),
                rejection_counts=rejection_counts,
                exposure_before=exposure_before,
                exposure_after=exposure_before,
            )

        command_gate_reasons: list[str] = []
        target_book_gate_reasons: list[str] = []
        rejection_counts: dict[str, int] = {}
        venue_diagnostics: dict[str, Any] = {}

        local_ok, local_reasons, local_exposure = evaluate_venue_target_transition(
            ledger=ledger.local,
            target=target_book.local,
            limits=self.limits.primary,
        )
        venue_diagnostics["local"] = local_exposure.as_dict()
        for reason in local_reasons:
            target_book_gate_reasons.append(reason)
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            counter_inc("mm_risk_reject_total", labels={"reason": reason})

        global_ok = True
        if target_book.global_ is not None and target_book.global_.orders:
            if ledger.global_ is None:
                target_book_gate_reasons.append("missing_global_ledger")
                rejection_counts["missing_global_ledger"] = rejection_counts.get(
                    "missing_global_ledger", 0
                ) + 1
                global_ok = False
            else:
                global_limits = _hedge_limits_as_venue_limits(self.limits.hedge)
                global_ok, global_reasons, global_exposure = evaluate_venue_target_transition(
                    ledger=ledger.global_,
                    target=target_book.global_,
                    limits=global_limits,
                )
                venue_diagnostics["global"] = global_exposure.as_dict()
                for reason in global_reasons:
                    prefixed = f"global_{reason}" if not reason.startswith("global_") else reason
                    target_book_gate_reasons.append(prefixed)
                    rejection_counts[prefixed] = rejection_counts.get(prefixed, 0) + 1
                    counter_inc("mm_risk_reject_total", labels={"reason": prefixed})

        if local_ok and global_ok and not command_gate_reasons:
            exposure_after = _target_exposure_dict(ledger, target_book)
            return TargetBookRiskResult(
                approved=target_book,
                exposure_before=exposure_before,
                exposure_after=exposure_after,
                transition_diagnostics=venue_diagnostics,
            )

        return TargetBookRiskResult(
            approved=None,
            target_book_gate_reasons=tuple(target_book_gate_reasons),
            command_gate_reasons=tuple(command_gate_reasons),
            rejection_counts=rejection_counts,
            exposure_before=exposure_before,
            exposure_after=exposure_before,
            transition_diagnostics=venue_diagnostics,
        )

    def _frame_gate_reasons(
        self,
        frame: StrategyInputFrame,
        *,
        target_book: JointTargetBook,
        ledger: JointLedgerSnapshot,
        live_execution: bool,
    ) -> list[str]:
        now_ms = int(self.clock_ms())
        reasons: list[str] = []

        primary = self.limits.primary
        market_age_ms = now_ms - int(frame.primary.t_end_ms)
        if market_age_ms > primary.max_market_data_age_ms:
            reasons.append("stale_market_data")

        local_private_reason = _ledger_private_freshness_reason(
            ledger.local,
            now_ms=now_ms,
            max_age_ms=primary.max_private_state_age_ms,
            missing_reason="missing_primary_private",
            stale_reason="stale_primary_private",
        )
        if local_private_reason is not None:
            reasons.append(local_private_reason)

        sourcing_active = (
            self.liquidity_sourcing.enabled
            and self.liquidity_sourcing.mode == "liquidity_sourcing"
        )
        has_global_venue = self.binding.hedge is not None
        global_orders = (
            target_book.global_ is not None and bool(target_book.global_.orders)
        )

        if not sourcing_active or not has_global_venue:
            return reasons

        if frame.hedge is None or frame.hedge_status not in {"ready"}:
            reasons.append("missing_global_market_data")
        else:
            hedge_market_age = now_ms - int(frame.hedge.t_end_ms)
            if hedge_market_age > self.limits.hedge.max_market_data_age_ms:
                reasons.append("stale_global_market_data")

        require_global_private = False
        if live_execution and global_orders:
            require_global_private = True
        elif self.liquidity_sourcing.require_global_private and global_orders:
            require_global_private = True

        if require_global_private:
            global_ledger = ledger.global_
            global_private_reason = _ledger_private_freshness_reason(
                global_ledger,
                now_ms=now_ms,
                max_age_ms=self.limits.hedge.max_private_state_age_ms,
                missing_reason="missing_global_private",
                stale_reason="stale_global_private",
            )
            if global_private_reason is not None:
                reasons.append(global_private_reason)

        return reasons


def _ledger_private_freshness_reason(
    ledger: VenueLedger | None,
    *,
    now_ms: int,
    max_age_ms: int,
    missing_reason: str,
    stale_reason: str,
) -> str | None:
    if ledger is None or ledger.last_snapshot_ms <= 0:
        return missing_reason
    if not math.isfinite(max_age_ms) or max_age_ms == float("inf"):
        return None
    age_ms = max(0, now_ms - int(ledger.last_snapshot_ms))
    if age_ms > max_age_ms:
        return stale_reason
    return None


def _target_book_order_count(book: JointTargetBook) -> int:
    count = len(book.local.orders)
    if book.global_ is not None:
        count += len(book.global_.orders)
    return count


def _legacy_exposure(state: TradingState) -> dict[str, float]:
    return {
        "inventory": float(state.inventory),
        "cash": float(state.cash),
    }


def _ledger_exposure_dict(ledger: JointLedgerSnapshot) -> dict[str, float]:
    exposure = {
        "local_available_base": float(ledger.local.available_base),
        "local_available_quote": float(ledger.local.available_quote),
        "local_locked_base": float(ledger.local.locked_base),
        "local_locked_quote": float(ledger.local.locked_quote),
    }
    if ledger.global_ is not None:
        exposure.update(
            {
                "global_available_base": float(ledger.global_.available_base),
                "global_available_quote": float(ledger.global_.available_quote),
                "global_locked_base": float(ledger.global_.locked_base),
                "global_locked_quote": float(ledger.global_.locked_quote),
            }
        )
    exposure["consolidated_base_inventory"] = float(ledger.consolidated_base_inventory)
    exposure["consolidated_quote_wealth"] = float(ledger.consolidated_quote_wealth)
    return exposure


def _target_exposure_dict(
    ledger: JointLedgerSnapshot,
    target_book: JointTargetBook,
) -> dict[str, float]:
    exposure = dict(_ledger_exposure_dict(ledger))
    local_target_base, local_target_quote = _collateral_from_target_orders(target_book.local.orders)
    exposure["local_target_locked_base"] = local_target_base
    exposure["local_target_locked_quote"] = local_target_quote
    if target_book.global_ is not None:
        global_target_base, global_target_quote = _collateral_from_target_orders(
            target_book.global_.orders
        )
        exposure["global_target_locked_base"] = global_target_base
        exposure["global_target_locked_quote"] = global_target_quote
    return exposure


def target_book_result_to_risk_result(
    result: TargetBookRiskResult,
    *,
    proposed_actions: list[Action],
) -> RiskResult:
    if result.approved is not None:
        approved_action = joint_target_book_to_action(result.approved)
        approved = (approved_action,) if approved_action.orders else ()
    else:
        approved = ()

    rejection_counts = dict(result.rejection_counts)
    if not rejection_counts:
        for reason in (*result.frame_gate_reasons, *result.target_book_gate_reasons):
            rejection_counts[reason] = rejection_counts.get(reason, 0) + _proposed_order_count(
                proposed_actions
            ) or 1

    exposure_before = dict(result.exposure_before)
    exposure_after = dict(result.exposure_after)
    if not exposure_before and proposed_actions:
        exposure_before = {"order_count": float(_proposed_order_count(proposed_actions))}
    if not exposure_after and result.approved is None:
        exposure_after = dict(exposure_before)

    return RiskResult(
        approved=approved,
        rejection_counts=rejection_counts,
        exposure_before=exposure_before,
        exposure_after=exposure_after,
    )
