"""Compatibility adapters between legacy and joint-venue domain models."""

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
    """Build a joint ledger snapshot from legacy frame private state."""

    local = venue_ledger_from_private_snapshot(frame.primary_private, binding.primary)
    global_ledger = (
        venue_ledger_from_private_snapshot(frame.hedge_private, binding.hedge)
        if binding.hedge is not None
        else None
    )

    local_inventory, local_cash = inventory_cash_from_snapshot(
        frame.primary_private,
        binding.primary.base_asset,
        binding.primary.quote_asset,
        trading_symbol=binding.primary.symbol,
    )
    global_inventory = 0.0
    global_cash = 0.0
    if global_ledger is not None and binding.hedge is not None:
        global_inventory, global_cash = inventory_cash_from_snapshot(
            frame.hedge_private,
            binding.hedge.base_asset,
            binding.hedge.quote_asset,
            trading_symbol=binding.hedge.symbol,
        )

    ref_price = float(reference_price if reference_price is not None else frame.primary.microprice)
    consolidated_base = local_inventory + global_inventory
    consolidated_quote = local_cash + global_cash

    now_ms = int(created_at_ms if created_at_ms is not None else frame.created_at_ms)
    local_private_age = (
        max(0, now_ms - int(frame.primary_private.timestamp_ms))
        if frame.primary_private is not None and now_ms > 0
        else None
    )
    global_private_age = (
        max(0, now_ms - int(frame.hedge_private.timestamp_ms))
        if frame.hedge_private is not None and now_ms > 0
        else None
    )
    local_market_age = (
        max(0, now_ms - int(frame.primary.t_end_ms)) if now_ms > 0 else None
    )
    global_market_age = (
        max(0, now_ms - int(frame.hedge.t_end_ms))
        if frame.hedge is not None and now_ms > 0
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
            ledger_fresh=frame.primary_private is not None,
        ),
    )


def strategy_input_frame_to_joint_strategy_frame(
    frame: StrategyInputFrame,
    binding: SymbolBinding,
    *,
    liquidity_sourcing_enabled: bool = False,
    liquidity_sourcing_mode: LiquiditySourcingMode = "single_venue",
) -> JointStrategyFrame:
    """Adapt legacy ``StrategyInputFrame`` into ``JointStrategyFrame``."""

    mode: LiquiditySourcingMode = (
        "liquidity_sourcing"
        if liquidity_sourcing_enabled and liquidity_sourcing_mode == "liquidity_sourcing"
        else "single_venue"
    )
    ledger = joint_ledger_from_frame(frame, binding, created_at_ms=frame.created_at_ms)
    sync_status = FrameSyncStatus(
        local_market_ready=True,
        global_market_ready=frame.hedge is not None and frame.hedge_status == "ready",
        local_private_ready=frame.primary_private is not None,
        global_private_ready=frame.hedge_private is not None,
        ledger_fresh=ledger.freshness.ledger_fresh,
        latency_skew_ms=frame.hedge_lag_ms,
        hedge_status=frame.hedge_status,
    )
    return JointStrategyFrame(
        local_market=frame.primary,
        global_market=frame.hedge,
        ledger=ledger,
        mode=mode,
        sync_status=sync_status,
        created_at_ms=frame.created_at_ms,
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
    """Convert a legacy ``Action`` into a single-venue target book."""

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
    """Convert a venue target book back to a legacy ``Action``."""

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
    """Wrap a legacy primary ``Action`` as a joint target book."""

    return JointTargetBook(
        actor_id=action.actor_id,
        local=action_to_venue_target_book(action),
        global_=global_book,
        mode=mode,
        diagnostics=dict(diagnostics or {}),
    )


def joint_target_book_to_action(book: JointTargetBook) -> Action:
    """Flatten a joint target book into one legacy ``Action`` for current execution."""

    orders: list[Order] = [
        _target_order_to_order(order) for order in book.local.orders
    ]
    if book.global_ is not None:
        orders.extend(_target_order_to_order(order) for order in book.global_.orders)
    return Action(actor_id=book.actor_id, orders=tuple(orders))
