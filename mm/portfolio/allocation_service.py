"""Portfolio allocation tracking and enforcement."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Fill
from mm.core.models.instrument import SymbolBinding
from mm.core.wallet import order_collateral_from_open_orders
from mm.observability import counter_inc, gauge, log_event
from mm.observability import schema as obs_schema
from mm.portfolio.limits import AllocationCaps, AllocationUsage
from mm.registry.models import AllocationRecord
from mm.runtime.control import RuntimeControlPlane


@dataclass(slots=True)
class ActorAllocationTracker:
    record: AllocationRecord
    binding: SymbolBinding
    usage: AllocationUsage | None = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = AllocationUsage(actor_id=self.record.actor_id)

    def apply_fill(self, fill: Fill) -> None:
        if fill.actor_id.strip() != self.record.actor_id:
            return
        qty = max(0.0, float(fill.qty))
        price = max(0.0, float(fill.price))
        fee = max(0.0, float(fill.fee))
        side = str(fill.side).lower().strip()
        if side == "buy":
            self.usage.attributed_base += qty
            self.usage.attributed_quote -= price * qty + fee
        elif side == "sell":
            self.usage.attributed_base -= qty
            self.usage.attributed_quote += price * qty - fee
        self.usage.realized_pnl_quote = self.usage.attributed_quote

    def refresh_open_order_locks(self, order_managers: dict[str, AbstractOrderManager]) -> None:
        manager = order_managers.get(self.binding.local.exchange.lower().strip())
        if manager is None:
            self.usage.locked_base = 0.0
            self.usage.locked_quote = 0.0
            return
        actor_orders = {
            order_id: open_order
            for order_id, open_order in manager.live_orders.items()
            if (open_order.actor_id or "").strip() == self.record.actor_id
        }
        locked_base, locked_quote = order_collateral_from_open_orders(
            actor_orders,
            trading_symbol=self.binding.local.symbol,
            base_asset=self.binding.local.base_asset,
            quote_asset=self.binding.local.quote_asset,
        )
        self.usage.locked_base = locked_base
        self.usage.locked_quote = locked_quote

    def caps(self) -> AllocationCaps:
        return AllocationCaps.from_record(self.record, self.usage)

    def breach_reasons(self) -> tuple[str, ...]:
        if self.record.status != "active":
            return ()
        reasons: list[str] = []
        if self.record.allocated_base > 0 and self.usage.used_base > self.record.allocated_base + 1e-9:
            reasons.append("allocation_base_exceeded")
        if self.record.allocated_quote > 0 and self.usage.used_quote > self.record.allocated_quote + 1e-9:
            reasons.append("allocation_quote_exceeded")
        return tuple(reasons)

    def snapshot_payload(self) -> dict[str, Any]:
        caps = self.caps()
        usage = self.usage.as_dict()
        limits = {
            "allocated_base": self.record.allocated_base,
            "allocated_quote": self.record.allocated_quote,
        }
        return {
            "type": "PortfolioAllocationSnapshot",
            "allocation_id": self.record.allocation_id,
            "strategy_id": self.record.strategy_id,
            "actor_id": self.record.actor_id,
            "account_id": self.record.account_id,
            "limits": limits,
            "usage": usage,
            "breach_reasons": list(self.breach_reasons()),
            "halt_on_breach": self.record.halt_on_breach,
            "timestamp_ms": int(time.time() * 1000),
            "caps": {
                "max_base_inventory": caps.max_base_inventory,
                "max_quote_exposure": caps.max_quote_exposure,
            },
        }


class PortfolioAllocationService:
    """Track per-actor allocation usage, publish portfolio views, and enforce halts."""

    def __init__(
        self,
        *,
        bus: AbstractEventBus,
        allocations: tuple[AllocationRecord, ...],
        bindings: dict[str, SymbolBinding],
        order_managers: dict[str, AbstractOrderManager],
        control_plane: RuntimeControlPlane | None = None,
    ) -> None:
        self.bus = bus
        self.order_managers = order_managers
        self.control_plane = control_plane
        self._trackers: dict[str, ActorAllocationTracker] = {}
        for record in allocations:
            if record.status != "active":
                continue
            binding = bindings.get(record.actor_id)
            if binding is None:
                continue
            self._trackers[record.actor_id] = ActorAllocationTracker(
                record=record,
                binding=binding,
                usage=AllocationUsage(actor_id=record.actor_id),
            )
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._tracked_exchanges = tuple(
            dict.fromkeys(tracker.binding.local.exchange.lower().strip() for tracker in self._trackers.values())
        )

    def usage_for(self, actor_id: str) -> AllocationUsage | None:
        tracker = self._trackers.get(actor_id.strip())
        return tracker.usage if tracker is not None else None

    def caps_for(self, actor_id: str) -> AllocationCaps | None:
        tracker = self._trackers.get(actor_id.strip())
        return tracker.caps() if tracker is not None else None

    def allocation_for(self, actor_id: str) -> AllocationRecord | None:
        tracker = self._trackers.get(actor_id.strip())
        return tracker.record if tracker is not None else None

    async def start(self) -> None:
        self._stopped.clear()
        for exchange in self._tracked_exchanges:
            self._tasks.append(asyncio.create_task(self._listen_fills(exchange)))
        self._tasks.append(asyncio.create_task(self._publish_loop()))

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _listen_fills(self, exchange: str) -> None:
        try:
            async for event in self.bus.subscribe(f"private:state:{exchange}:fills"):
                if not isinstance(event, Fill):
                    continue
                tracker = self._trackers.get((event.actor_id or "").strip())
                if tracker is None:
                    continue
                tracker.apply_fill(event)
                await self._after_usage_update(tracker)
        except asyncio.CancelledError:
            raise

    async def _publish_loop(self) -> None:
        try:
            while not self._stopped.is_set():
                await self._publish_snapshots()
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise

    async def _publish_snapshots(self) -> None:
        portfolio: list[dict[str, Any]] = []
        for tracker in self._trackers.values():
            tracker.refresh_open_order_locks(self.order_managers)
            payload = tracker.snapshot_payload()
            portfolio.append(payload)
            self._export_metrics(tracker)
            await self._enforce_breach(tracker)
        await self.bus.publish(
            "portfolio:allocations",
            {
                "type": "PortfolioAllocationView",
                "actors": portfolio,
                "timestamp_ms": int(time.time() * 1000),
            },
        )

    async def _after_usage_update(self, tracker: ActorAllocationTracker) -> None:
        tracker.refresh_open_order_locks(self.order_managers)
        self._export_metrics(tracker)
        payload = tracker.snapshot_payload()
        await self.bus.publish(f"portfolio:allocation:{tracker.record.actor_id}", payload)
        await self._enforce_breach(tracker)

    async def _enforce_breach(self, tracker: ActorAllocationTracker) -> None:
        reasons = tracker.breach_reasons()
        if not reasons:
            if self.control_plane is not None:
                self.control_plane.state.allocation_breached_actors.discard(tracker.record.actor_id)
            return
        for reason in reasons:
            counter_inc(
                obs_schema.MM_ALLOCATION_BREACH_TOTAL,
                labels={"actor_id": tracker.record.actor_id, "reason": reason},
            )
        if not tracker.record.halt_on_breach or self.control_plane is None:
            return
        self.control_plane.halt_actor_for_allocation_breach(
            tracker.record.actor_id,
            reasons=reasons,
        )

    def _export_metrics(self, tracker: ActorAllocationTracker) -> None:
        labels = {
            "actor_id": tracker.record.actor_id,
            "account_id": tracker.record.account_id,
            "strategy_id": tracker.record.strategy_id,
        }
        gauge(
            obs_schema.MM_ALLOCATION_USED_BASE,
            tracker.usage.used_base,
            labels=labels,
            help="Attributed base usage for a strategy allocation",
        )
        gauge(
            obs_schema.MM_ALLOCATION_USED_QUOTE,
            tracker.usage.used_quote,
            labels=labels,
            help="Attributed quote usage for a strategy allocation",
        )
        gauge(
            obs_schema.MM_ALLOCATION_LIMIT_BASE,
            tracker.record.allocated_base,
            labels=labels,
            help="Allocated base inventory cap",
        )
        gauge(
            obs_schema.MM_ALLOCATION_LIMIT_QUOTE,
            tracker.record.allocated_quote,
            labels=labels,
            help="Allocated quote capital cap",
        )
        gauge(
            obs_schema.MM_ALLOCATION_REALIZED_PNL_QUOTE,
            tracker.usage.realized_pnl_quote,
            labels=labels,
            help="Attributed realized PnL in quote terms from fills",
        )
