"""Route approved actions to per-exchange order managers."""

from __future__ import annotations

from collections import defaultdict

from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Action


async def route_actions_to_managers(
    actions: list[Action],
    order_managers: dict[str, AbstractOrderManager],
) -> None:
    """Group creates/cancels by exchange and call ``OrderManager.sync``."""

    by_exchange: dict[str, list[Action]] = defaultdict(list)
    for action in actions:
        creates_by_exchange: dict[str, list] = defaultdict(list)
        for order in action.orders_to_create:
            creates_by_exchange[order.exchange.lower().strip()].append(order)
        cancel_ids = action.orders_to_cancel
        for exchange, orders in creates_by_exchange.items():
            by_exchange[exchange].append(
                Action(
                    actor_id=action.actor_id,
                    orders_to_create=tuple(orders),
                    orders_to_cancel=cancel_ids,
                )
            )
        if cancel_ids and not creates_by_exchange:
            for exchange in order_managers:
                by_exchange[exchange].append(
                    Action(actor_id=action.actor_id, orders_to_create=(), orders_to_cancel=cancel_ids)
                )
    for exchange, routed in by_exchange.items():
        manager = order_managers.get(exchange)
        if manager is not None:
            await manager.sync(routed)
