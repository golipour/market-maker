from __future__ import annotations

from mm.core.models.execution import Order
from mm.core.models.private import OpenOrder
from mm.execution.reconcile import ReconcileConfig, reconcile_target_orders


def _order(side: str, price: float, qty: float, *, client_order_id: str | None = None) -> Order:
    return Order("nobitex", "BTC_IRT", side, price, qty, client_order_id=client_order_id)


def _live(
    order_id: str,
    order: Order,
    *,
    actor_id: str = "actor-1",
    remaining_qty: float | None = None,
) -> OpenOrder:
    qty = remaining_qty if remaining_qty is not None else order.qty
    return OpenOrder(
        exchange_order_id=order_id,
        actor_id=actor_id,
        order=order,
        remaining_qty=qty,
        filled_qty=0.0,
        status="open",
        timestamp_ms=1000,
    )


def test_reconcile_keeps_exact_match_and_places_new() -> None:
    live = {
        "live-1": _live("live-1", _order("buy", 100.0, 1.0, client_order_id="old")),
    }
    desired = (
        _order("buy", 100.0, 1.0, client_order_id="new-id"),
        _order("sell", 101.0, 2.0, client_order_id="sell-1"),
    )
    plan = reconcile_target_orders(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        desired=desired,
        live_orders=live,
    )
    assert plan.keep == (live["live-1"],)
    assert plan.cancel_ids == ()
    assert len(plan.place) == 1
    assert plan.place[0].side == "sell"


def test_reconcile_cancels_orphan_live_orders() -> None:
    live = {
        "live-1": _live("live-1", _order("buy", 100.0, 1.0)),
        "live-2": _live("live-2", _order("sell", 105.0, 0.5)),
    }
    desired = (_order("buy", 100.0, 1.0),)
    plan = reconcile_target_orders(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        desired=desired,
        live_orders=live,
    )
    assert plan.keep == (live["live-1"],)
    assert plan.cancel_ids == ("live-2",)
    assert plan.place == ()


def test_reconcile_empty_target_cancels_all_actor_orders() -> None:
    live = {
        "live-1": _live("live-1", _order("buy", 100.0, 1.0)),
    }
    plan = reconcile_target_orders(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        desired=(),
        live_orders=live,
    )
    assert plan.keep == ()
    assert plan.cancel_ids == ("live-1",)
    assert plan.place == ()


def test_reconcile_idempotent_second_sync() -> None:
    order = _order("buy", 100.0, 1.0)
    live = {"live-1": _live("live-1", order)}
    plan = reconcile_target_orders(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        desired=(order,),
        live_orders=live,
    )
    assert plan.keep == (live["live-1"],)
    assert plan.cancel_ids == ()
    assert plan.place == ()


def test_reconcile_respects_price_and_qty_tolerance() -> None:
    live = {"live-1": _live("live-1", _order("buy", 100.0, 1.0))}
    desired = (_order("buy", 100.05, 1.005),)
    strict = reconcile_target_orders(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        desired=desired,
        live_orders=live,
        config=ReconcileConfig(price_tol_bps=0.0, qty_tol_rel=0.0),
    )
    assert strict.place == desired
    assert strict.cancel_ids == ("live-1",)

    loose = reconcile_target_orders(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        desired=desired,
        live_orders=live,
        config=ReconcileConfig(price_tol_bps=10.0, qty_tol_rel=0.01),
    )
    assert loose.keep == (live["live-1"],)
    assert loose.place == ()


def test_reconcile_ignores_other_actors() -> None:
    live = {
        "live-1": _live("live-1", _order("buy", 100.0, 1.0), actor_id="other"),
    }
    plan = reconcile_target_orders(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        desired=(_order("buy", 100.0, 1.0),),
        live_orders=live,
    )
    assert plan.keep == ()
    assert plan.place[0].price == 100.0
    assert plan.cancel_ids == ()


def test_reconcile_cleans_up_unattributed_orders() -> None:
    live = {
        "live-1": _live("live-1", _order("buy", 100.0, 1.0), actor_id=""),
        "live-2": _live("live-2", _order("sell", 110.0, 1.0), actor_id=""),
    }
    # Matching unattributed order is kept, non-matching unattributed order is cancelled
    plan = reconcile_target_orders(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        desired=(_order("buy", 100.0, 1.0),),
        live_orders=live,
        include_unattributed=True,
    )
    assert plan.keep == (live["live-1"],)
    assert plan.cancel_ids == ("live-2",)
    assert plan.place == ()


def test_reconcile_config_can_leave_unattributed_orders_alone() -> None:
    live = {
        "live-1": _live("live-1", _order("buy", 100.0, 1.0), actor_id=""),
    }
    plan = reconcile_target_orders(
        actor_id="actor-1",
        exchange="nobitex",
        symbol="BTC_IRT",
        desired=(),
        live_orders=live,
        config=ReconcileConfig(include_unattributed=False),
    )
    assert plan.cancel_ids == ()
    assert plan.keep == ()

