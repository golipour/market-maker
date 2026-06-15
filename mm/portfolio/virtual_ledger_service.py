"""Per-strategy virtual ledgers backed by registry wallet allocations."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.order_manager import AbstractOrderManager
from mm.core.models.execution import Fill
from mm.core.models.instrument import SymbolBinding
from mm.core.models.ledger import JointLedgerSnapshot
from mm.observability import counter_inc, gauge, log_event
from mm.observability import schema as obs_schema
from mm.portfolio.wallet import StrategyWallet, joint_ledger_with_virtual_local
from mm.registry.models import AllocationRecord
from mm.runtime.control import RuntimeControlPlane


@dataclass(slots=True)
class ActorWalletTracker:
    record: AllocationRecord
    binding: SymbolBinding
    wallet: StrategyWallet | None = None

    def __post_init__(self) -> None:
        if self.wallet is None:
            self.wallet = StrategyWallet.from_allocation(self.record, self.binding)

    def apply_fill(self, fill: Fill) -> None:
        assert self.wallet is not None
        self.wallet.apply_fill(fill)

    def refresh_open_order_locks(self, order_managers: dict[str, AbstractOrderManager]) -> None:
        assert self.wallet is not None
        self.wallet.refresh_open_order_locks(order_managers)

    def breach_reasons(self) -> tuple[str, ...]:
        if self.record.status != "active":
            return ()
        assert self.wallet is not None
        return self.wallet.breach_reasons()

    def snapshot_payload(self) -> dict[str, Any]:
        assert self.wallet is not None
        wallet_view = self.wallet.as_dict()
        return {
            "type": "VirtualLedgerSnapshot",
            "allocation_id": self.record.allocation_id,
            "strategy_id": self.record.strategy_id,
            "actor_id": self.record.actor_id,
            "account_id": self.record.account_id,
            "wallet": wallet_view,
            "initial_wallet": {
                "base": self.wallet.initial_base,
                "quote": self.wallet.initial_quote,
                "base_asset": self.wallet.base_asset,
                "quote_asset": self.wallet.quote_asset,
            },
            "breach_reasons": list(self.breach_reasons()),
            "halt_on_breach": self.record.halt_on_breach,
            "timestamp_ms": int(time.time() * 1000),
        }


# Backward-compatible alias.
ActorAllocationTracker = ActorWalletTracker


class VirtualLedgerService:
    """Maintain per-actor virtual ledgers, publish snapshots, and enforce halts."""

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
        self._trackers: dict[str, ActorWalletTracker] = {}
        for record in allocations:
            if record.status != "active":
                continue
            binding = bindings.get(record.actor_id)
            if binding is None:
                continue
            self._trackers[record.actor_id] = ActorWalletTracker(
                record=record,
                binding=binding,
            )
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._tracked_exchanges = tuple(
            dict.fromkeys(tracker.binding.local.exchange.lower().strip() for tracker in self._trackers.values())
        )

    def wallet_for(self, actor_id: str) -> StrategyWallet | None:
        tracker = self._trackers.get(actor_id.strip())
        if tracker is None:
            return None
        tracker.refresh_open_order_locks(self.order_managers)
        return tracker.wallet

    def joint_ledger_for(
        self,
        actor_id: str,
        physical_ledger: JointLedgerSnapshot,
    ) -> JointLedgerSnapshot:
        """Strategy ledger: virtual local balances with physical global + freshness."""

        wallet = self.wallet_for(actor_id)
        if wallet is None:
            return physical_ledger
        return joint_ledger_with_virtual_local(physical_ledger, wallet)

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
                "type": "VirtualLedgerView",
                "actors": portfolio,
                "timestamp_ms": int(time.time() * 1000),
            },
        )

    async def _after_usage_update(self, tracker: ActorWalletTracker) -> None:
        tracker.refresh_open_order_locks(self.order_managers)
        self._export_metrics(tracker)
        payload = tracker.snapshot_payload()
        await self.bus.publish(f"portfolio:allocation:{tracker.record.actor_id}", payload)
        await self._enforce_breach(tracker)

    async def _enforce_breach(self, tracker: ActorWalletTracker) -> None:
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

    def _export_metrics(self, tracker: ActorWalletTracker) -> None:
        assert tracker.wallet is not None
        wallet = tracker.wallet
        labels = {
            "actor_id": tracker.record.actor_id,
            "account_id": tracker.record.account_id,
            "strategy_id": tracker.record.strategy_id,
        }
        gauge(
            obs_schema.MM_ALLOCATION_USED_BASE,
            wallet.held_base,
            labels=labels,
            help="Strategy wallet held base",
        )
        gauge(
            obs_schema.MM_ALLOCATION_USED_QUOTE,
            wallet.held_quote,
            labels=labels,
            help="Strategy wallet held quote",
        )
        gauge(
            obs_schema.MM_ALLOCATION_LIMIT_BASE,
            wallet.initial_base,
            labels=labels,
            help="Strategy wallet initial base",
        )
        gauge(
            obs_schema.MM_ALLOCATION_LIMIT_QUOTE,
            wallet.initial_quote,
            labels=labels,
            help="Strategy wallet initial quote",
        )
        gauge(
            obs_schema.MM_ALLOCATION_REALIZED_PNL_QUOTE,
            wallet.held_quote - wallet.initial_quote,
            labels=labels,
            help="Wallet quote PnL vs initial allocation",
        )


# Backward-compatible alias for existing imports.
PortfolioAllocationService = VirtualLedgerService
