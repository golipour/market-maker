"""Stateless order router: dispatch validated execution plans to exchange adapters."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.models.execution import Order
from mm.core.payload_schema import (
    cancel_orders_accepted_event,
    cancel_orders_failed_event,
    cancel_orders_requested_event,
    execution_lineage_fields,
    execution_order_summary,
    order_validation_rejected_event,
    place_orders_accepted_event,
    place_orders_failed_event,
    place_orders_requested_event,
)
from mm.execution.target_book import CommandRejection, ExecutionPlan, RouterResult
from mm.exchanges.shared.provider_adapter import now_ms
from mm.observability import counter_inc, log_event
from mm.observability import schema as obs_schema


class OrderRouter:
    """Route cancel/place commands and publish lifecycle events.

    The router is deliberately stateless: it does not mutate live-order truth.
    OrderManager records accepted commands as pending until private confirmation.
    """

    def __init__(
        self,
        exchange: str,
        command_adapter: AbstractOrderCommandAdapter,
        *,
        bus: AbstractEventBus | None = None,
        clock_ms: Callable[[], int] = now_ms,
    ) -> None:
        self.exchange = exchange.lower().strip()
        self.command_adapter = command_adapter
        self.bus = bus
        self.clock_ms = clock_ms

    async def execute(self, plan: ExecutionPlan) -> RouterResult:
        """Execute cancels before places for replace and reconcile policies."""

        cancelled: list[str] = []
        failed_cancels: list[str] = []
        placed: list[tuple[str, Order]] = []
        failed_places: list[CommandRejection] = []
        lineage = execution_lineage_fields(
            frame_id=plan.frame_id,
            decision_id=plan.decision_id,
            target_book_id=plan.target_book_id,
            execution_plan_id=plan.execution_plan_id,
        )

        cancel_ms = 0.0
        place_ms = 0.0

        cancel_ids = [command.exchange_order_id for command in plan.cancel]
        if cancel_ids:
            cancel_started_ns = time.monotonic_ns()
            await self._publish(
                "execution:orders",
                cancel_orders_requested_event(
                    exchange=self.exchange,
                    order_ids=cancel_ids,
                    policy=plan.policy,
                    timestamp_ms=self.clock_ms(),
                    **lineage,
                ),
            )
            try:
                await self.command_adapter.cancel_orders(cancel_ids)
                cancelled.extend(cancel_ids)
                await self._publish(
                    "execution:orders",
                    cancel_orders_accepted_event(
                        exchange=self.exchange,
                        order_ids=cancel_ids,
                        timestamp_ms=self.clock_ms(),
                        **lineage,
                    ),
                )
            except Exception as exc:
                failed_cancels.extend(cancel_ids)
                self._observe_adapter_failure(plan, operation="cancel_orders", exc=exc)
                await self._publish(
                    "execution:orders",
                    cancel_orders_failed_event(
                        exchange=self.exchange,
                        order_ids=cancel_ids,
                        error=type(exc).__name__,
                        message=str(exc),
                        timestamp_ms=self.clock_ms(),
                        **lineage,
                    ),
                )
                cancel_ms = (time.monotonic_ns() - cancel_started_ns) / 1_000_000
                if plan.policy in ("replace", "reconcile"):
                    return RouterResult(
                        cancelled_ids=tuple(cancelled),
                        failed_cancel_ids=tuple(failed_cancels),
                        cancel_ms=cancel_ms,
                        place_ms=place_ms,
                    )
            else:
                cancel_ms = (time.monotonic_ns() - cancel_started_ns) / 1_000_000

        if plan.place:
            place_started_ns = time.monotonic_ns()
            orders = [command.order for command in plan.place]
            await self._publish(
                "execution:orders",
                place_orders_requested_event(
                    exchange=self.exchange,
                    orders=[execution_order_summary(order) for order in orders],
                    policy=plan.policy,
                    timestamp_ms=self.clock_ms(),
                    **lineage,
                ),
            )
            try:
                order_ids = await self.command_adapter.place_orders(orders)
            except Exception as exc:
                place_ms = (time.monotonic_ns() - place_started_ns) / 1_000_000
                self._observe_adapter_failure(plan, operation="place_orders", exc=exc)
                failed_places.extend(
                    CommandRejection(
                        order=command.order,
                        reason="router_adapter_failure",
                        stage="router",
                    )
                    for command in plan.place
                )
                await self._publish(
                    "execution:orders",
                    place_orders_failed_event(
                        exchange=self.exchange,
                        orders=[execution_order_summary(command.order) for command in plan.place],
                        error=type(exc).__name__,
                        message=str(exc),
                        timestamp_ms=self.clock_ms(),
                        **lineage,
                    ),
                )
            else:
                timestamp_ms = self.clock_ms()
                accepted_orders = []
                for command, order_id in zip(plan.place, order_ids):
                    placed.append((order_id, command.order))
                    accepted_orders.append(
                        {
                            **execution_order_summary(command.order),
                            "exchange_order_id": order_id,
                        }
                    )
                await self._publish(
                    "execution:orders",
                    place_orders_accepted_event(
                        exchange=self.exchange,
                        order_ids=order_ids,
                        orders=accepted_orders,
                        timestamp_ms=timestamp_ms,
                        **lineage,
                    ),
                )
                place_ms = (time.monotonic_ns() - place_started_ns) / 1_000_000

        for rejection in plan.rejected:
            await self._publish(
                "execution:orders",
                order_validation_rejected_event(
                    exchange=self.exchange,
                    symbol=rejection.order.symbol,
                    reason=rejection.reason,
                    side=rejection.order.side,
                    price=float(rejection.order.price),
                    qty=float(rejection.order.qty),
                    stage=rejection.stage,
                    timestamp_ms=self.clock_ms(),
                    **lineage,
                ),
            )

        return RouterResult(
            cancelled_ids=tuple(cancelled),
            placed=tuple(placed),
            failed_cancel_ids=tuple(failed_cancels),
            failed_places=tuple(failed_places),
            cancel_ms=cancel_ms,
            place_ms=place_ms,
        )

    async def _publish(self, topic_prefix: str, event: dict[str, Any]) -> None:
        if self.bus is not None:
            await self.bus.publish(f"{topic_prefix}:{self.exchange}", event)

    def _observe_adapter_failure(self, plan: ExecutionPlan, *, operation: str, exc: BaseException) -> None:
        labels = {
            "exchange": self.exchange,
            "operation": operation,
            "error": type(exc).__name__,
        }
        counter_inc(
            obs_schema.MM_ORDER_ROUTER_FAILURES_TOTAL,
            labels=labels,
            help="Order router adapter failures by operation and exception class",
        )
        log_event(
            obs_schema.EVENT_ADAPTER_REQUEST_FAILED,
            {
                "exchange": self.exchange,
                "symbol": plan.symbol,
                "actor_id": plan.actor_id,
                "operation": operation,
                "error": type(exc).__name__,
                "message": str(exc),
                "retryable": True,
            },
        )
