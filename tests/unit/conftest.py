"""Shared helpers for unit tests (venues.local/global config shape)."""

from __future__ import annotations

from typing import Any

from mm.core.models.execution import Action, Order
from mm.core.models.market import MarketEvent
from mm.core.models.strategy import StrategyDecision
from mm.core.models.target_book import JointTargetBook
from mm.strategies.target_book_builder import build_venue_target_book_from_orders


def venues_payload(
    *,
    local: dict[str, Any] | None = None,
    global_: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    local_cfg = dict(local or {"exchange": "nobitex", "symbol": "BTC_IRT"})
    venues: dict[str, Any] = {"local": local_cfg}
    if global_ is False:
        pass
    elif global_ is not None:
        venues["global"] = dict(global_)
    return {"venues": venues, **extra}


def strategy_decision_with_joint(
    event: MarketEvent,
    *,
    frame_step: int,
    actor_id: str = "test-runner",
    order: Order | None = None,
) -> StrategyDecision:
    if order is None:
        order = Order(
            event.exchange,
            event.symbol,
            "buy",
            99.0,
            1.0,
            client_order_id="c1",
        )
    joint_book = JointTargetBook(
        actor_id=actor_id,
        local=build_venue_target_book_from_orders(actor_id=actor_id, orders=(order,)),
    )
    return StrategyDecision(
        actor_id=actor_id,
        exchange=event.exchange,
        symbol=event.symbol,
        t_end_ms=event.t_end_ms,
        frame_step=frame_step,
        action=Action(actor_id=actor_id, orders=(order,)),
        joint_target_book=joint_book,
    )
