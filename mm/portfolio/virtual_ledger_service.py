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
from mm.core.payload_schema import private_state_snapshot_from_payload
from mm.core.wallet import reported_wallet_total
from mm.observability import counter_inc, gauge, log_event
from mm.observability import schema as obs_schema
from mm.portfolio.wallet import StrategyWallet, joint_ledger_with_virtual_local
from mm.portfolio.wallet_store import WalletBalance, WalletStore
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
        wallet_store: WalletStore | None = None,
    ) -> None:
        self.bus = bus
        self.order_managers = order_managers
        self.control_plane = control_plane
        self._wallet_store = wallet_store
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
        self._wallet_lock = asyncio.Lock()
        self._physical_totals: dict[str, dict[str, float]] = {}
        self._tracked_exchanges = tuple(
            dict.fromkeys(tracker.binding.local.exchange.lower().strip() for tracker in self._trackers.values())
        )

    def wallet_for(self, actor_id: str) -> StrategyWallet | None:
        tracker = self._trackers.get(actor_id.strip())
        if tracker is None:
            return None
        tracker.refresh_open_order_locks(self.order_managers)
        self._mark_global_isolation(tracker)
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
        await self._restore_wallets()
        for exchange in self._tracked_exchanges:
            self._tasks.append(asyncio.create_task(self._listen_fills(exchange)))
            self._tasks.append(asyncio.create_task(self._listen_snapshots(exchange)))
        self._tasks.append(asyncio.create_task(self._publish_loop()))

    async def stop(self) -> None:
        self._stopped.set()
        async with self._wallet_lock:
            await self._persist_wallets()
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
                async with self._wallet_lock:
                    tracker.apply_fill(event)
                    await self._persist_wallets(tracker)
                await self._after_usage_update(tracker)
        except asyncio.CancelledError:
            raise

    async def _listen_snapshots(self, exchange: str) -> None:
        try:
            async for event in self.bus.subscribe(f"private:state:{exchange}:snapshot"):
                self._remember_physical_totals(exchange, event)
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
        self._apply_physical_coverage()
        for tracker in self._trackers.values():
            tracker.refresh_open_order_locks(self.order_managers)
            self._mark_global_isolation(tracker)
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
        self._apply_physical_coverage()
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

    async def _restore_wallets(self) -> None:
        if self._wallet_store is None:
            return
        try:
            saved = await asyncio.to_thread(self._wallet_store.load)
        except Exception as exc:
            log_event(
                "virtual_wallet_restore_failed",
                {"error": type(exc).__name__, "message": str(exc)},
            )
            return
        for actor_id, row in saved.items():
            tracker = self._trackers.get(actor_id)
            if tracker is None or tracker.wallet is None:
                continue
            tracker.wallet.restore_held(
                exchange=row.exchange,
                symbol=row.symbol,
                initial_base=row.initial_base,
                initial_quote=row.initial_quote,
                held_base=row.held_base,
                held_quote=row.held_quote,
            )
        await self._persist_wallets()

    async def _persist_wallets(self, tracker: ActorWalletTracker | None = None) -> None:
        if self._wallet_store is None:
            return
        trackers = (tracker,) if tracker is not None else tuple(self._trackers.values())
        rows = [
            WalletBalance.from_wallet(item.wallet)
            for item in trackers
            if item.wallet is not None
        ]
        if not rows:
            return
        try:
            await asyncio.to_thread(self._wallet_store.save, rows)
        except Exception as exc:
            log_event(
                "virtual_wallet_persist_failed",
                {"error": type(exc).__name__, "message": str(exc)},
            )

    def _remember_physical_totals(self, exchange: str, event: Any) -> None:
        try:
            snapshot = private_state_snapshot_from_payload(event)
        except (TypeError, ValueError, KeyError, AssertionError):
            return
        totals = {
            str(asset): reported_wallet_total(balance)
            for asset, balance in snapshot.balances.items()
        }
        self._physical_totals[exchange.lower().strip()] = totals

    def _apply_physical_coverage(self) -> None:
        sums: dict[tuple[str, str], float] = {}
        for tracker in self._trackers.values():
            wallet = tracker.wallet
            if wallet is None:
                continue
            _add_held(sums, wallet.exchange, wallet.base_asset, wallet.held_base)
            if wallet.quote_asset != wallet.base_asset:
                _add_held(sums, wallet.exchange, wallet.quote_asset, wallet.held_quote)
        for tracker in self._trackers.values():
            wallet = tracker.wallet
            if wallet is None:
                continue
            physical = self._physical_totals.get(wallet.exchange)
            if not physical:
                wallet.coverage_reasons = ()
                continue
            reasons: list[str] = []
            for asset in (wallet.base_asset, wallet.quote_asset):
                if asset not in physical:
                    continue
                physical_total = physical[asset]
                virtual_sum = sums.get((wallet.exchange, asset), 0.0)
                epsilon = max(1e-6, abs(physical_total) * 1e-6)
                if virtual_sum > physical_total + epsilon:
                    reasons.append("wallet_exceeds_physical")
                    break
            wallet.coverage_reasons = tuple(reasons)

    def _mark_global_isolation(self, tracker: ActorWalletTracker) -> None:
        wallet = tracker.wallet
        if wallet is None:
            return
        global_spec = tracker.binding.global_
        if global_spec is None:
            wallet.isolate_global_balances = False
            return
        exchange = global_spec.exchange.lower().strip()
        wallet.isolate_global_balances = self._actor_count_on_exchange(exchange) > 1

    def _actor_count_on_exchange(self, exchange: str) -> int:
        key = exchange.lower().strip()
        count = 0
        for tracker in self._trackers.values():
            touched = {tracker.binding.local.exchange.lower().strip()}
            if tracker.binding.global_ is not None:
                touched.add(tracker.binding.global_.exchange.lower().strip())
            if key in touched:
                count += 1
        return count

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


def _add_held(sums: dict[tuple[str, str], float], exchange: str, asset: str, held: float) -> None:
    key = (exchange, asset)
    sums[key] = sums.get(key, 0.0) + max(0.0, float(held))


# Backward-compatible alias for existing imports.
PortfolioAllocationService = VirtualLedgerService
