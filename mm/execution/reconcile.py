"""Reconcile declarative target orders against live open orders for one actor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mm.core.models.execution import Order
from mm.core.models.private import OpenOrder


@dataclass(frozen=True, slots=True)
class ReconcileConfig:
    """Tolerance for matching target orders to live orders (same side)."""

    price_tol_bps: float = 0.0
    qty_tol_rel: float = 0.0


@dataclass(frozen=True, slots=True)
class ReconcilePlan:
    """Minimal API plan: keep matched live orders, cancel orphans, place new targets."""

    keep: tuple[OpenOrder, ...] = ()
    cancel_ids: tuple[str, ...] = ()
    place: tuple[Order, ...] = ()


def reconcile_target_orders(
    *,
    actor_id: str,
    exchange: str,
    symbol: str | None,
    desired: Sequence[Order],
    live_orders: Mapping[str, OpenOrder],
    config: ReconcileConfig | None = None,
    include_unattributed: bool = True,
) -> ReconcilePlan:
    """Match *desired* to *live_orders* for one actor; cancel unmatched live, place unmatched targets."""

    cfg = config or ReconcileConfig()
    actor = actor_id.strip()
    ex = exchange.lower().strip()
    sym = symbol.lower().strip() if symbol else None

    scoped_desired = [
        order
        for order in desired
        if order.exchange.lower().strip() == ex
        and (sym is None or order.symbol.lower().strip() == sym)
    ]
    scoped_live = [
        open_order
        for open_order in live_orders.values()
        if _live_matches_scope(
            open_order,
            actor_id=actor,
            exchange=ex,
            symbol=sym,
            include_unattributed=include_unattributed,
        )
    ]

    if not scoped_desired:
        return ReconcilePlan(
            keep=(),
            cancel_ids=tuple(o.exchange_order_id for o in scoped_live),
            place=(),
        )

    used_live: set[str] = set()
    keep: list[OpenOrder] = []
    place: list[Order] = []

    for side in ("buy", "sell"):
        side_desired = [o for o in scoped_desired if o.side.lower().strip() == side]
        side_live = [
            o
            for o in scoped_live
            if o.order.side.lower().strip() == side and o.exchange_order_id not in used_live
        ]
        side_desired.sort(key=lambda o: o.price, reverse=(side == "buy"))
        for target in side_desired:
            best: OpenOrder | None = None
            best_score = float("inf")
            for candidate in side_live:
                if candidate.exchange_order_id in used_live:
                    continue
                if not _within_tolerance(target, candidate, cfg):
                    continue
                score = _match_score(target, candidate)
                if score < best_score:
                    best_score = score
                    best = candidate
            if best is not None:
                used_live.add(best.exchange_order_id)
                keep.append(best)
            else:
                place.append(target)

    cancel_ids = tuple(
        o.exchange_order_id for o in scoped_live if o.exchange_order_id not in used_live
    )
    return ReconcilePlan(keep=tuple(keep), cancel_ids=cancel_ids, place=tuple(place))


def _live_matches_scope(
    open_order: OpenOrder,
    *,
    actor_id: str,
    exchange: str,
    symbol: str | None,
    include_unattributed: bool = False,
) -> bool:
    order_actor = (open_order.actor_id or "").strip()
    if order_actor != actor_id and not (include_unattributed and not order_actor):
        return False
    order = open_order.order
    if order.exchange.lower().strip() != exchange:
        return False
    if symbol is not None and order.symbol.lower().strip() != symbol:
        return False
    return True


def _match_score(desired: Order, live: OpenOrder) -> float:
    live_order = live.order
    ref_price = max(abs(desired.price), abs(live_order.price), 1e-12)
    price_bps = abs(desired.price - live_order.price) / ref_price * 10_000.0
    ref_qty = max(desired.qty, live.remaining_qty, 1e-12)
    qty_rel = abs(desired.qty - live.remaining_qty) / ref_qty
    return price_bps + qty_rel * 100.0


def _within_tolerance(desired: Order, live: OpenOrder, config: ReconcileConfig) -> bool:
    live_order = live.order
    if desired.side.lower().strip() != live_order.side.lower().strip():
        return False
    if desired.symbol.lower().strip() != live_order.symbol.lower().strip():
        return False
    ref_price = max(abs(desired.price), abs(live_order.price), 1e-12)
    price_bps = abs(desired.price - live_order.price) / ref_price * 10_000.0
    ref_qty = max(desired.qty, live.remaining_qty, 1e-12)
    qty_rel = abs(desired.qty - live.remaining_qty) / ref_qty
    return price_bps <= config.price_tol_bps and qty_rel <= config.qty_tol_rel
