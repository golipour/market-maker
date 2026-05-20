"""Route approved actions to per-exchange order managers."""

from __future__ import annotations

from collections import defaultdict

from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Action
from mm.execution.plan import ExecutionSyncReport


async def route_actions_to_managers(
    actions: list[Action],
    order_managers: dict[str, AbstractOrderManager],
) -> None:
    """Group target orders by exchange and call ``OrderManager.sync``."""
    await route_actions_to_managers_with_report(actions, order_managers)


async def route_actions_to_managers_with_report(
    actions: list[Action],
    order_managers: dict[str, AbstractOrderManager],
) -> ExecutionSyncReport:
    """Route actions and aggregate reconcile reports."""

    by_exchange: dict[str, list[Action]] = defaultdict(list)
    for action in actions:
        creates_by_exchange: dict[str, list] = defaultdict(list)
        for order in action.orders:
            creates_by_exchange[order.exchange.lower().strip()].append(order)
        for exchange, orders in creates_by_exchange.items():
            by_exchange[exchange].append(
                Action(actor_id=action.actor_id, orders=tuple(orders))
            )

    combined = ExecutionSyncReport()
    for exchange, routed in by_exchange.items():
        manager = order_managers.get(exchange)
        if manager is None:
            continue
        sync_with_report = getattr(manager, "sync_with_report", None)
        if sync_with_report is not None:
            report = await sync_with_report(routed)
            combined = combined.merge(report)
        else:
            await manager.sync(routed)
    return combined
