"""Portfolio ledger service: subscribe to private topics and publish joint snapshots."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Fill
from mm.core.models.instrument import SymbolBinding
from mm.core.models.ledger import JointLedgerSnapshot, VenueLedger
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.ledger.models import (
    LedgerConsistencyDiagnostic,
    LedgerOrderManagerMismatch,
    LedgerOrderMismatch,
)
from mm.ledger.state_machine import VenueLedgerStateMachine, build_joint_ledger_snapshot
from mm.runtime.config import MarketMakerRuntimeConfig


def _compare_live_orders(
    ledger_orders: dict[str, OpenOrder],
    manager_orders: dict[str, OpenOrder],
) -> tuple[bool, tuple[LedgerOrderMismatch, ...]]:
    mismatches: list[LedgerOrderMismatch] = []
    ledger_ids = set(ledger_orders)
    manager_ids = set(manager_orders)

    for order_id in sorted(ledger_ids - manager_ids):
        mismatches.append(
            LedgerOrderMismatch(
                exchange_order_id=order_id,
                reason="missing_in_manager",
                ledger_value=ledger_orders[order_id].remaining_qty,
            )
        )
    for order_id in sorted(manager_ids - ledger_ids):
        mismatches.append(
            LedgerOrderMismatch(
                exchange_order_id=order_id,
                reason="missing_in_ledger",
                manager_value=manager_orders[order_id].remaining_qty,
            )
        )
    for order_id in sorted(ledger_ids & manager_ids):
        ledger_order = ledger_orders[order_id]
        manager_order = manager_orders[order_id]
        if ledger_order.remaining_qty != manager_order.remaining_qty:
            mismatches.append(
                LedgerOrderMismatch(
                    exchange_order_id=order_id,
                    reason="remaining_qty",
                    ledger_value=ledger_order.remaining_qty,
                    manager_value=manager_order.remaining_qty,
                )
            )
        elif ledger_order.order.side != manager_order.order.side:
            mismatches.append(
                LedgerOrderMismatch(
                    exchange_order_id=order_id,
                    reason="side",
                    ledger_value=ledger_order.order.side,
                    manager_value=manager_order.order.side,
                )
            )
        elif ledger_order.order.price != manager_order.order.price:
            mismatches.append(
                LedgerOrderMismatch(
                    exchange_order_id=order_id,
                    reason="price",
                    ledger_value=ledger_order.order.price,
                    manager_value=manager_order.order.price,
                )
            )
    return not mismatches, tuple(mismatches)


@dataclass(slots=True)
class _MismatchTracker:
    signature: tuple[tuple[str, str, str, str], ...]
    first_seen_ms: int
    consecutive_count: int = 1


class PortfolioLedgerService:
    """Maintain the strategy-facing ledger from private state snapshots."""

    def __init__(
        self,
        config: MarketMakerRuntimeConfig,
        binding: SymbolBinding,
        bus: AbstractEventBus,
        *,
        order_managers: dict[str, AbstractOrderManager] | None = None,
        clock_ms: Any | None = None,
    ) -> None:
        self.config = config
        self.binding = binding
        self.bus = bus
        self.order_managers = order_managers or {}
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._local = VenueLedgerStateMachine(
            config.local.normalized_exchange(),
            binding.local,
        )
        self._global: VenueLedgerStateMachine | None = None
        if config.global_ is not None and binding.global_ is not None:
            self._global = VenueLedgerStateMachine(
                config.global_.normalized_exchange(),
                binding.global_,
            )
        self._tracked_exchanges = self._configured_exchanges()
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._reference_price = 0.0
        self._mismatch_trackers: dict[str, _MismatchTracker] = {}
        self._mismatch_alert_after_count = 3

    @property
    def local_ledger(self) -> VenueLedger:
        return self._local.snapshot()

    @property
    def global_ledger(self) -> VenueLedger | None:
        return self._global.snapshot() if self._global is not None else None

    def joint_snapshot(
        self,
        *,
        reference_price: float | None = None,
        created_at_ms: int | None = None,
    ) -> JointLedgerSnapshot:
        ref = self._reference_price if reference_price is None else float(reference_price)
        return build_joint_ledger_snapshot(
            self.local_ledger,
            self.global_ledger,
            self.binding,
            reference_price=ref,
            created_at_ms=created_at_ms if created_at_ms is not None else self._clock_ms(),
        )

    def consistency_diagnostic(self, exchange: str) -> LedgerConsistencyDiagnostic | None:
        manager = self.order_managers.get(exchange.lower().strip())
        if manager is None:
            return None
        ledger = self._ledger_for_exchange(exchange)
        if ledger is None:
            return None
        manager_orders = dict(manager.live_orders)
        matches, mismatches = _compare_live_orders(dict(ledger.live_orders), manager_orders)
        return LedgerConsistencyDiagnostic(
            exchange=exchange.lower().strip(),
            timestamp_ms=self._clock_ms(),
            ledger_order_count=len(ledger.live_orders),
            manager_order_count=len(manager_orders),
            matches=matches,
            mismatches=mismatches,
        )

    async def start(self) -> None:
        if not self._tracked_exchanges:
            return
        self._stopped.clear()
        for exchange in self._tracked_exchanges:
            self._tasks.extend(
                [
                    asyncio.create_task(self._listen_snapshot(exchange)),
                    asyncio.create_task(self._listen_balances(exchange)),
                    asyncio.create_task(self._listen_open_orders(exchange)),
                    asyncio.create_task(self._listen_order_updates(exchange)),
                    asyncio.create_task(self._listen_fills(exchange)),
                ]
            )

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def _configured_exchanges(self) -> tuple[str, ...]:
        exchanges: list[str] = []
        for source in self.config.sources:
            private = self.config.private_config_for(source.role)
            if private is None:
                continue
            if not private.resolved_api_credentials(source.normalized_exchange()):
                continue
            exchanges.append(source.normalized_exchange())
        return tuple(exchanges)

    def _ledger_for_exchange(self, exchange: str) -> VenueLedger | None:
        key = exchange.lower().strip()
        if key == self.config.local.normalized_exchange():
            return self.local_ledger
        if self.config.global_ is not None and key == self.config.global_.normalized_exchange():
            return self.global_ledger
        return None

    def _machine_for_exchange(self, exchange: str) -> VenueLedgerStateMachine | None:
        key = exchange.lower().strip()
        if key == self.config.local.normalized_exchange():
            return self._local
        if self._global is not None and self.config.global_ is not None:
            if key == self.config.global_.normalized_exchange():
                return self._global
        return None

    async def _listen_snapshot(self, exchange: str) -> None:
        try:
            async for event in self.bus.subscribe(f"private:state:{exchange}:snapshot"):
                if not isinstance(event, PrivateStateSnapshot):
                    continue
                machine = self._machine_for_exchange(exchange)
                if machine is None:
                    continue
                machine.apply_snapshot(event)
                await self._publish_after_update(exchange)
        except asyncio.CancelledError:
            raise

    async def _listen_balances(self, exchange: str) -> None:
        try:
            async for event in self.bus.subscribe(f"private:state:{exchange}:balances"):
                if not isinstance(event, Balance):
                    continue
                machine = self._machine_for_exchange(exchange)
                if machine is None:
                    continue
                machine.apply_balance(event)
                await self._publish_after_update(exchange)
        except asyncio.CancelledError:
            raise

    async def _listen_open_orders(self, exchange: str) -> None:
        try:
            async for event in self.bus.subscribe(f"private:state:{exchange}:open_orders"):
                if not isinstance(event, OpenOrder):
                    continue
                machine = self._machine_for_exchange(exchange)
                if machine is None:
                    continue
                machine.apply_open_order(event)
                await self._publish_after_update(exchange)
        except asyncio.CancelledError:
            raise

    async def _listen_order_updates(self, exchange: str) -> None:
        try:
            async for event in self.bus.subscribe(f"private:state:{exchange}:order_updates"):
                if not isinstance(event, OrderUpdate):
                    continue
                machine = self._machine_for_exchange(exchange)
                if machine is None:
                    continue
                machine.apply_order_update(event)
                await self._publish_after_update(exchange)
        except asyncio.CancelledError:
            raise

    async def _listen_fills(self, exchange: str) -> None:
        try:
            async for event in self.bus.subscribe(f"private:state:{exchange}:fills"):
                if not isinstance(event, Fill):
                    continue
                machine = self._machine_for_exchange(exchange)
                if machine is None:
                    continue
                machine.apply_fill(event)
                await self._publish_after_update(exchange)
        except asyncio.CancelledError:
            raise

    async def _publish_after_update(self, exchange: str) -> None:
        ledger = self._ledger_for_exchange(exchange)
        if ledger is None:
            return
        await self.bus.publish(f"ledger:snapshot:{exchange}", ledger.as_payload())
        joint = self.joint_snapshot(created_at_ms=self._clock_ms())
        await self.bus.publish("ledger:snapshot", joint.as_payload())
        diagnostic = self.consistency_diagnostic(exchange)
        if diagnostic is not None:
            await self.bus.publish(
                f"ledger:diagnostics:consistency:{exchange}",
                diagnostic.as_payload(),
            )
            mismatch = self._persistent_mismatch_event(diagnostic)
            if mismatch is not None:
                await self.bus.publish(
                    f"ledger:diagnostics:mismatch:{exchange}",
                    mismatch.as_payload(),
                )

    def set_reference_price(self, price: float) -> None:
        self._reference_price = float(price)

    def _persistent_mismatch_event(
        self,
        diagnostic: LedgerConsistencyDiagnostic,
    ) -> LedgerOrderManagerMismatch | None:
        exchange = diagnostic.exchange.lower().strip()
        if diagnostic.matches:
            self._mismatch_trackers.pop(exchange, None)
            return None
        signature = _mismatch_signature(diagnostic.mismatches)
        tracker = self._mismatch_trackers.get(exchange)
        if tracker is None or tracker.signature != signature:
            tracker = _MismatchTracker(
                signature=signature,
                first_seen_ms=diagnostic.timestamp_ms,
            )
            self._mismatch_trackers[exchange] = tracker
        else:
            tracker.consecutive_count += 1
        if tracker.consecutive_count < self._mismatch_alert_after_count:
            return None
        return LedgerOrderManagerMismatch(
            exchange=exchange,
            timestamp_ms=diagnostic.timestamp_ms,
            first_seen_ms=tracker.first_seen_ms,
            consecutive_count=tracker.consecutive_count,
            ledger_order_count=diagnostic.ledger_order_count,
            manager_order_count=diagnostic.manager_order_count,
            mismatches=diagnostic.mismatches,
        )


def _mismatch_signature(
    mismatches: tuple[LedgerOrderMismatch, ...],
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            mismatch.exchange_order_id,
            mismatch.reason,
            repr(mismatch.ledger_value),
            repr(mismatch.manager_value),
        )
        for mismatch in mismatches
    )


def build_portfolio_ledger_service(
    config: MarketMakerRuntimeConfig,
    *,
    bus: AbstractEventBus,
    order_managers: dict[str, AbstractOrderManager] | None = None,
) -> PortfolioLedgerService | None:
    """Build ledger service when at least one private venue is configured."""

    service = PortfolioLedgerService(
        config,
        config.binding,
        bus,
        order_managers=order_managers,
    )
    if not service._tracked_exchanges:
        return None
    return service
