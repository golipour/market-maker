"""Adapters between local/global strategy inputs, ledgers, and target books."""

from __future__ import annotations

from mm.core.models.execution import Action, Order
from mm.core.models.instrument import InstrumentSpec, SymbolBinding
from mm.core.models.joint_strategy import FrameSyncStatus, JointStrategyFrame
from mm.core.models.ledger import JointLedgerSnapshot, LedgerFreshness, VenueLedger
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.core.models.target_book import (
    JointTargetBook,
    LiquiditySourcingMode,
    TargetOrder,
    VenueTargetBook,
)
from mm.core.wallet import inventory_cash_from_snapshot, order_collateral_from_open_orders


def venue_ledger_from_private_snapshot(
    snapshot: PrivateStateSnapshot | None,
    spec: InstrumentSpec,
) -> VenueLedger:
    """Build a venue ledger snapshot from a private state snapshot."""

    if snapshot is None:
        return VenueLedger(
            exchange=spec.exchange,
            symbol=spec.symbol,
            base_asset=spec.base_asset,
            quote_asset=spec.quote_asset,
        )

    base_order_locked, quote_order_locked = order_collateral_from_open_orders(
        snapshot.open_orders,
        trading_symbol=spec.symbol,
        base_asset=spec.base_asset,
        quote_asset=spec.quote_asset,
    )
    base_balance = snapshot.balances.get(spec.base_asset)
    quote_balance = snapshot.balances.get(spec.quote_asset)
    available_base = float(base_balance.free) if base_balance is not None else 0.0
    available_quote = float(quote_balance.free) if quote_balance is not None else 0.0

    return VenueLedger(
        exchange=spec.exchange,
        symbol=spec.symbol,
        base_asset=spec.base_asset,
        quote_asset=spec.quote_asset,
        balances=dict(snapshot.balances),
        live_orders=dict(snapshot.open_orders),
        locked_base=base_order_locked,
        locked_quote=quote_order_locked,
        available_base=available_base,
        available_quote=available_quote,
        last_private_event_ms=int(snapshot.timestamp_ms),
        last_snapshot_ms=int(snapshot.timestamp_ms),
    )


def joint_ledger_from_frame(
    frame: StrategyInputFrame,
    binding: SymbolBinding,
    *,
    reference_price: float | None = None,
    created_at_ms: int | None = None,
) -> JointLedgerSnapshot:
    """Build a joint ledger snapshot from frame private state."""

    local = venue_ledger_from_private_snapshot(frame.local_private, binding.local)
    global_ledger = (
        venue_ledger_from_private_snapshot(frame.global_private, binding.global_)
        if binding.global_ is not None
        else None
    )

    local_inventory, local_cash = inventory_cash_from_snapshot(
        frame.local_private,
        binding.local.base_asset,
        binding.local.quote_asset,
        trading_symbol=binding.local.symbol,
    )
    global_inventory = 0.0
    global_cash = 0.0
    if global_ledger is not None and binding.global_ is not None:
        global_inventory, global_cash = inventory_cash_from_snapshot(
            frame.global_private,
            binding.global_.base_asset,
            binding.global_.quote_asset,
            trading_symbol=binding.global_.symbol,
        )

    ref_price = float(reference_price if reference_price is not None else frame.local.microprice)
    consolidated_base = local_inventory + global_inventory
    consolidated_quote = local_cash + global_cash

    now_ms = int(created_at_ms if created_at_ms is not None else frame.created_at_ms)
    local_private_age = (
        max(0, now_ms - int(frame.local_private.timestamp_ms))
        if frame.local_private is not None and now_ms > 0
        else None
    )
    global_private_age = (
        max(0, now_ms - int(frame.global_private.timestamp_ms))
        if frame.global_private is not None and now_ms > 0
        else None
    )
    local_market_age = (
        max(0, now_ms - int(frame.local.t_end_ms)) if now_ms > 0 else None
    )
    global_market_age = (
        max(0, now_ms - int(frame.global_.t_end_ms))
        if frame.global_ is not None and now_ms > 0
        else None
    )

    return JointLedgerSnapshot(
        local=local,
        global_=global_ledger,
        reference_price=ref_price,
        consolidated_base_inventory=consolidated_base,
        consolidated_quote_wealth=consolidated_quote + consolidated_base * ref_price,
        created_at_ms=now_ms,
        freshness=LedgerFreshness(
            local_private_age_ms=local_private_age,
            global_private_age_ms=global_private_age,
            local_market_age_ms=local_market_age,
            global_market_age_ms=global_market_age,
            ledger_fresh=frame.local_private is not None,
        ),
    )


from mm.runtime.config import LiquiditySourcingRuntimeConfig


def _reference_price_from_frame(
    frame: StrategyInputFrame,
    *,
    reference_venue: str = "global",
) -> float:
    if reference_venue == "global" and frame.global_ is not None:
        return float(frame.global_.microprice)
    return float(frame.local.microprice)


def _venue_private_ready(
    ledger: VenueLedger | None,
    *,
    max_private_age_ms: int,
    now_ms: int,
) -> bool:
    if ledger is None or ledger.last_snapshot_ms <= 0:
        return False
    if max_private_age_ms <= 0:
        return True
    age_ms = max(0, now_ms - int(ledger.last_snapshot_ms))
    return age_ms <= max_private_age_ms


def build_frame_sync_status(
    frame: StrategyInputFrame,
    *,
    ledger: JointLedgerSnapshot | None,
    binding: SymbolBinding,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig,
    now_ms: int,
    max_local_private_age_ms: int = 120_000,
    max_global_private_age_ms: int = 120_000,
) -> FrameSyncStatus:
    """Compute runtime synchronization readiness for a joint strategy frame."""

    has_global_venue = binding.global_ is not None
    sourcing_active = (
        liquidity_sourcing.enabled and liquidity_sourcing.mode == "liquidity_sourcing"
    )
    global_market_required = sourcing_active and has_global_venue

    local_market_ready = True
    if global_market_required:
        global_market_ready = frame.global_ is not None and frame.global_status == "ready"
    elif has_global_venue:
        global_market_ready = frame.global_status in {"ready", "disabled", "missing", "stale"}
    else:
        global_market_ready = True

    local_private_ready = _venue_private_ready(
        ledger.local if ledger is not None else None,
        max_private_age_ms=max_local_private_age_ms,
        now_ms=now_ms,
    )

    if not has_global_venue:
        global_private_ready = True
    elif sourcing_active and liquidity_sourcing.require_global_private:
        global_private_ready = _venue_private_ready(
            ledger.global_ if ledger is not None else None,
            max_private_age_ms=max_global_private_age_ms,
            now_ms=now_ms,
        )
    elif sourcing_active:
        global_ledger = ledger.global_ if ledger is not None else None
        global_private_ready = (
            global_ledger is None
            or global_ledger.last_snapshot_ms <= 0
            or _venue_private_ready(
                global_ledger,
                max_private_age_ms=max_global_private_age_ms,
                now_ms=now_ms,
            )
        )
    else:
        global_private_ready = True

    ledger_fresh = ledger.freshness.ledger_fresh if ledger is not None else False

    return FrameSyncStatus(
        local_market_ready=local_market_ready,
        global_market_ready=global_market_ready,
        local_private_ready=local_private_ready,
        global_private_ready=global_private_ready,
        ledger_fresh=ledger_fresh,
        latency_skew_ms=frame.global_lag_ms,
        global_status=frame.global_status,
    )


def joint_strategy_frame_from_input(
    frame: StrategyInputFrame,
    binding: SymbolBinding,
    *,
    ledger: JointLedgerSnapshot | None,
    liquidity_sourcing: LiquiditySourcingRuntimeConfig,
    now_ms: int | None = None,
    max_local_private_age_ms: int = 120_000,
    max_global_private_age_ms: int = 120_000,
    input_frame: StrategyInputFrame | None = None,
) -> JointStrategyFrame:
    """Build a ledger-backed joint frame from aligned market events."""

    created_at_ms = int(now_ms if now_ms is not None else frame.created_at_ms)
    mode: LiquiditySourcingMode = (
        "liquidity_sourcing"
        if liquidity_sourcing.enabled and liquidity_sourcing.mode == "liquidity_sourcing"
        else "single_venue"
    )
    if ledger is None:
        ledger = joint_ledger_from_frame(
            frame,
            binding,
            reference_price=_reference_price_from_frame(
                frame,
                reference_venue=liquidity_sourcing.reference_venue,
            ),
            created_at_ms=created_at_ms,
        )
    sync_status = build_frame_sync_status(
        frame,
        ledger=ledger,
        binding=binding,
        liquidity_sourcing=liquidity_sourcing,
        now_ms=created_at_ms,
        max_local_private_age_ms=max_local_private_age_ms,
        max_global_private_age_ms=max_global_private_age_ms,
    )
    return JointStrategyFrame(
        local_market=frame.local,
        global_market=frame.global_,
        ledger=ledger,
        mode=mode,
        sync_status=sync_status,
        created_at_ms=created_at_ms,
        input_frame=input_frame or frame,
    )


def _order_to_target_order(order: Order) -> TargetOrder:
    return TargetOrder(
        exchange=order.exchange,
        symbol=order.symbol,
        side=order.side,
        price=float(order.price),
        qty=float(order.qty),
        order_type=order.order_type,
    )


def _target_order_to_order(order: TargetOrder, *, client_order_id: str | None = None) -> Order:
    return Order(
        exchange=order.exchange,
        symbol=order.symbol,
        side=order.side,
        price=float(order.price),
        qty=float(order.qty),
        order_type=order.order_type,
        client_order_id=client_order_id,
    )


def action_to_venue_target_book(
    action: Action,
    *,
    policy: str = "reconcile",
) -> VenueTargetBook:
    """Convert an ``Action`` into a single-venue target book."""

    if not action.orders:
        return VenueTargetBook(
            exchange="",
            symbol="",
            actor_id=action.actor_id,
            policy="reconcile" if policy != "replace" else "replace",
        )
    first = action.orders[0]
    exchange = first.exchange
    symbol = first.symbol
    orders = tuple(_order_to_target_order(order) for order in action.orders)
    exec_policy = "replace" if policy == "replace" else "reconcile"
    return VenueTargetBook(
        exchange=exchange,
        symbol=symbol,
        actor_id=action.actor_id,
        orders=orders,
        policy=exec_policy,
    )


def venue_target_book_to_action(book: VenueTargetBook) -> Action:
    """Convert a venue target book into an executable action."""

    return Action(
        actor_id=book.actor_id,
        orders=tuple(_target_order_to_order(order) for order in book.orders),
    )


def action_to_joint_target_book(
    action: Action,
    *,
    mode: LiquiditySourcingMode = "single_venue",
    global_book: VenueTargetBook | None = None,
    diagnostics: dict | None = None,
) -> JointTargetBook:
    """Wrap a local action as a joint target book."""

    return JointTargetBook(
        actor_id=action.actor_id,
        local=action_to_venue_target_book(action),
        global_=global_book,
        mode=mode,
        diagnostics=dict(diagnostics or {}),
    )


def joint_target_book_to_action(book: JointTargetBook) -> Action:
    """Flatten a joint target book into one executable action."""

    orders: list[Order] = [
        _target_order_to_order(order) for order in book.local.orders
    ]
    if book.global_ is not None:
        orders.extend(_target_order_to_order(order) for order in book.global_.orders)
    return Action(actor_id=book.actor_id, orders=tuple(orders))
