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
from mm.core.risk_limits import GlobalVenueRiskLimits, VenueRiskLimits
from mm.core.wallet import order_collateral_from_open_orders, reported_wallet_total
from mm.execution.plan import RiskResult
from mm.observability import counter_inc, gauge
from mm.portfolio.limits import AllocationCaps, wallet_initial_caps
from mm.portfolio.wallet import StrategyWallet, venue_ledger_from_strategy_wallet
from mm.risk.engine import RiskLimits, _proposed_order_count
from mm.risk.models import TargetBookRiskResult, VenueTransitionExposure
from mm.runtime.config import LiquiditySourcingRuntimeConfig


def _scope_actor_live_orders(
    live_orders: dict[str, OpenOrder],
    *,
    actor_id: str,
    exchange: str,
    symbol: str,
    include_unattributed: bool = False,
) -> dict[str, OpenOrder]:
    ex = str(exchange or "").lower().strip()
    sym = str(symbol or "").strip()
    actor = str(actor_id or "").strip()
    scoped: dict[str, OpenOrder] = {}
    for order_id, open_order in live_orders.items():
        order_actor = (open_order.actor_id or "").strip()
        if order_actor != actor and not (include_unattributed and not order_actor):
            continue
        order = open_order.order
        if str(order.exchange or "").lower().strip() != ex:
            continue
        if str(order.symbol or "").strip() != sym:
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


def _empty_venue_target_book(book: VenueTargetBook) -> VenueTargetBook:
    return VenueTargetBook(
        exchange=book.exchange,
        symbol=book.symbol,
        actor_id=book.actor_id,
        orders=(),
        policy=book.policy,
    )


def _venue_target_book_without_rejected_sides(
    book: VenueTargetBook,
    reasons: list[str],
) -> VenueTargetBook:
    rejected_sides = set().union(
        *(_risk_rejected_sides(reason) for reason in reasons)
    )
    return VenueTargetBook(
        exchange=book.exchange,
        symbol=book.symbol,
        actor_id=book.actor_id,
        orders=tuple(
            order
            for order in book.orders
            if str(order.side).lower().strip() not in rejected_sides
        ),
        policy=book.policy,
    )


def _can_approve_partial_book(reasons: list[str]) -> bool:
    return bool(reasons) and all(_risk_rejected_sides(reason) for reason in reasons)


def _risk_rejected_sides(reason: str) -> set[str]:
    normalized = reason.removeprefix("global_")
    if normalized in {"insufficient_quote_for_bid_orders", "min_cash", "max_inventory"}:
        return {"buy"}
    if normalized in {"insufficient_base_for_ask_orders", "min_inventory"}:
        return {"sell"}
    if normalized in {"allocation_quote_exceeded", "wallet_quote_deficit", "wallet_quote_locked_exceeds_held"}:
        return {"buy"}
    if normalized in {"allocation_base_exceeded", "wallet_base_deficit", "wallet_base_locked_exceeds_held"}:
        return {"buy"}
    return set()


def _global_limits_as_venue_limits(limits: GlobalVenueRiskLimits) -> VenueRiskLimits:
    return limits


def _venue_base_quote_totals(ledger: VenueLedger) -> tuple[float, float]:
    base_balance = ledger.balances.get(ledger.base_asset)
    quote_balance = ledger.balances.get(ledger.quote_asset)
    base_total = (
        reported_wallet_total(base_balance)
        if base_balance is not None
        else float(ledger.available_base) + float(ledger.locked_base)
    )
    quote_total = (
        reported_wallet_total(quote_balance)
        if quote_balance is not None
        else float(ledger.available_quote) + float(ledger.locked_quote)
    )
    return base_total, quote_total


def evaluate_venue_target_transition(
    *,
    ledger: VenueLedger,
    target: VenueTargetBook,
    limits: VenueRiskLimits,
    release_unattributed: bool = False,
    allocation: AllocationCaps | None = None,
) -> tuple[bool, list[str], VenueTransitionExposure]:
    """Evaluate whether ``target`` resting book is feasible from ``ledger`` state."""

    actor_live = _scope_actor_live_orders(
        ledger.live_orders,
        actor_id=target.actor_id,
        exchange=target.exchange,
        symbol=target.symbol,
        include_unattributed=release_unattributed,
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

    if target.policy == "replace":
        base_capacity = base_total
        quote_capacity = quote_total
    else:
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
        reasons.append("insufficient_base_for_ask_orders")
    if target_locked_quote > quote_capacity + 1e-9:
        reasons.append("insufficient_quote_for_bid_orders")

    free_quote_after_lock = quote_total - target_locked_quote
    if free_quote_after_lock < limits.min_cash:
        reasons.append("min_cash")

    inventory_after_sells = base_total - target_sell_qty
    if inventory_after_sells < limits.min_inventory:
        reasons.append("min_inventory")

    inventory_after_buys = base_total + target_buy_qty
    if inventory_after_buys > limits.max_inventory:
        reasons.append("max_inventory")

    if allocation is not None and allocation.max_base_inventory is not None:
        projected_base = allocation.attributed_base + target_locked_base
        if projected_base > allocation.max_base_inventory + 1e-9:
            reasons.append("allocation_base_exceeded")
    if allocation is not None and allocation.max_quote_exposure is not None:
        projected_quote = allocation.attributed_quote_used + target_locked_quote
        if projected_quote > allocation.max_quote_exposure + 1e-9:
            reasons.append("allocation_quote_exceeded")

    return not reasons, reasons, exposure


@dataclass(slots=True)
class NetDeltaRiskEngine(AbstractRiskEngine):
    """Approve or reject the full target-book transition against ledger snapshots."""

    limits: RiskLimits
    binding: SymbolBinding
    liquidity_sourcing: LiquiditySourcingRuntimeConfig
    clock_ms: Any = None
    kill_switch: bool = False
    allocation_caps_provider: Any | None = None
    strategy_wallet_provider: Any | None = None

    def __post_init__(self) -> None:
        if self.clock_ms is None:
            self.clock_ms = lambda: int(time.time() * 1000)

    def set_kill_switch(self, enabled: bool) -> None:
        self.kill_switch = bool(enabled)
        gauge("mm_kill_switch_state", 1.0 if self.kill_switch else 0.0)

    def _allocation_caps_for(self, actor_id: str) -> AllocationCaps | None:
        if self.strategy_wallet_provider is not None:
            wallet = self.strategy_wallet_provider()
            if wallet is not None:
                return wallet_initial_caps(wallet)
        if self.allocation_caps_provider is None:
            return None
        return self.allocation_caps_provider()

    def _local_ledger_for_risk(self, ledger: JointLedgerSnapshot) -> VenueLedger:
        if self.strategy_wallet_provider is None:
            return ledger.local
        wallet = self.strategy_wallet_provider()
        if wallet is None:
            return ledger.local
        return venue_ledger_from_strategy_wallet(
            wallet,
            live_orders=ledger.local.live_orders,
        )

    def validate(self, actions: list[Action], state: TradingState, event: MarketEvent) -> list[Action]:
        return self.validate_with_result(actions, state, event).approved_list

    def validate_with_result(
        self, actions: list[Action], state: TradingState, event: MarketEvent
    ) -> RiskResult:
        frame = StrategyInputFrame(local=event)
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
        global_live_execution: bool | None = None,
    ) -> RiskResult:
        from mm.core.adapters.joint import action_to_joint_target_book, joint_ledger_from_frame

        if joint_target_book is None and actions:
            joint_target_book = action_to_joint_target_book(actions[0])
        if joint_target_book is None:
            return RiskResult(
                approved=(),
                rejection_counts={"no_target_book": 1},
                exposure_before=_trading_state_exposure(state),
                exposure_after=_trading_state_exposure(state),
            )

        if ledger is None:
            ledger = joint_ledger_from_frame(frame, self.binding, created_at_ms=frame.created_at_ms)

        result = self.validate_target_book_with_result(
            joint_target_book,
            ledger,
            frame,
            live_execution=live_execution,
            global_live_execution=global_live_execution,
        )
        return target_book_result_to_risk_result(result, proposed_actions=actions)

    def validate_target_book_with_result(
        self,
        target_book: JointTargetBook,
        ledger: JointLedgerSnapshot,
        frame: StrategyInputFrame,
        *,
        live_execution: bool = False,
        global_live_execution: bool | None = None,
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
            global_live_execution=global_live_execution,
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
            ledger=self._local_ledger_for_risk(ledger),
            target=target_book.local,
            limits=self.limits.local,
            release_unattributed=target_book.local.policy == "replace",
            allocation=self._allocation_caps_for(target_book.actor_id),
        )
        venue_diagnostics["local"] = local_exposure.as_dict()
        for reason in local_reasons:
            target_book_gate_reasons.append(reason)
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            counter_inc("mm_risk_reject_total", labels={"reason": reason})

        global_ok = True
        global_reasons: list[str] = []
        if target_book.global_ is not None and target_book.global_.orders:
            global_stale = frame.global_ is None or frame.global_status not in {"ready"}
            if not global_stale and frame.global_ is not None:
                global_market_age = int(self.clock_ms()) - int(frame.global_.t_end_ms)
                global_stale = global_market_age > self.limits.global_.max_market_data_age_ms
            if global_stale:
                stale_reason = (
                    "stale_global_market_data"
                    if frame.global_ is not None and frame.global_status == "ready"
                    else "missing_global_market_data"
                )
                target_book_gate_reasons.append(stale_reason)
                rejection_counts[stale_reason] = rejection_counts.get(stale_reason, 0) + len(
                    target_book.global_.orders
                )
                global_ok = False
            elif ledger.global_ is None:
                target_book_gate_reasons.append("missing_global_ledger")
                rejection_counts["missing_global_ledger"] = rejection_counts.get(
                    "missing_global_ledger", 0
                ) + 1
                global_ok = False
            else:
                global_limits = _global_limits_as_venue_limits(self.limits.global_)
                global_ok, global_reasons, global_exposure = evaluate_venue_target_transition(
                    ledger=ledger.global_,
                    target=target_book.global_,
                    limits=global_limits,
                    release_unattributed=target_book.global_.policy == "replace",
                )
                venue_diagnostics["global"] = global_exposure.as_dict()
                for reason in global_reasons:
                    prefixed = f"global_{reason}" if not reason.startswith("global_") else reason
                    target_book_gate_reasons.append(prefixed)
                    rejection_counts[prefixed] = rejection_counts.get(prefixed, 0) + 1
                    counter_inc("mm_risk_reject_total", labels={"reason": prefixed})

        global_orders_rejected = (
            target_book.global_ is not None
            and bool(target_book.global_.orders)
            and not global_ok
        )

        approved_local: VenueTargetBook | None = None
        if local_ok:
            approved_local = target_book.local
        elif _can_approve_partial_book(local_reasons):
            approved_local = _venue_target_book_without_rejected_sides(
                target_book.local,
                local_reasons,
            )
        elif target_book.local.policy == "replace":
            approved_local = _empty_venue_target_book(target_book.local)

        approved_global: VenueTargetBook | None = None
        local_can_execute = approved_local is not None
        if target_book.global_ is not None:
            if local_can_execute and global_ok:
                approved_global = target_book.global_
            elif local_can_execute and _can_approve_partial_book(global_reasons):
                approved_global = _venue_target_book_without_rejected_sides(
                    target_book.global_,
                    global_reasons,
                )
            elif local_can_execute and global_orders_rejected:
                approved_global = _empty_venue_target_book(target_book.global_)
            elif not local_can_execute and target_book.global_.policy == "replace":
                approved_global = _empty_venue_target_book(target_book.global_)
            elif not target_book.global_.orders:
                approved_global = target_book.global_

        if approved_local is not None and not command_gate_reasons:
            approved_book = JointTargetBook(
                actor_id=target_book.actor_id,
                local=approved_local,
                global_=approved_global,
                mode=target_book.mode,
                diagnostics=dict(target_book.diagnostics),
            )
            exposure_after = _target_exposure_dict(ledger, approved_book)
            return TargetBookRiskResult(
                approved=approved_book,
                target_book_gate_reasons=tuple(target_book_gate_reasons),
                command_gate_reasons=tuple(command_gate_reasons),
                rejection_counts=rejection_counts,
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
        global_live_execution: bool | None = None,
    ) -> list[str]:
        now_ms = int(self.clock_ms())
        reasons: list[str] = []

        local_limits = self.limits.local
        market_age_ms = now_ms - int(frame.local.t_end_ms)
        if market_age_ms > local_limits.max_market_data_age_ms:
            reasons.append("stale_market_data")

        local_private_reason = _ledger_private_freshness_reason(
            ledger.local,
            now_ms=now_ms,
            max_age_ms=local_limits.max_private_state_age_ms,
            missing_reason="missing_local_private",
            stale_reason="stale_local_private",
        )
        if local_private_reason is not None:
            reasons.append(local_private_reason)

        sourcing_active = (
            self.liquidity_sourcing.enabled
            and self.liquidity_sourcing.mode == "liquidity_sourcing"
        )
        has_global_venue = self.binding.global_ is not None
        global_orders = (
            target_book.global_ is not None and bool(target_book.global_.orders)
        )

        if not sourcing_active or not has_global_venue:
            return reasons

        if global_live_execution:
            if frame.global_ is None or frame.global_status not in {"ready"}:
                reasons.append("missing_global_market_data")
            else:
                global_market_age = now_ms - int(frame.global_.t_end_ms)
                if global_market_age > self.limits.global_.max_market_data_age_ms:
                    reasons.append("stale_global_market_data")

        require_global_private = False
        global_live = (
            global_live_execution
            if global_live_execution is not None
            else (live_execution and global_orders)
        )
        if global_live and global_orders:
            require_global_private = True
        elif self.liquidity_sourcing.require_global_private and global_orders:
            require_global_private = True

        if require_global_private:
            global_ledger = ledger.global_
            global_private_reason = _ledger_private_freshness_reason(
                global_ledger,
                now_ms=now_ms,
                max_age_ms=self.limits.global_.max_private_state_age_ms,
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


def _trading_state_exposure(state: TradingState) -> dict[str, float]:
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
