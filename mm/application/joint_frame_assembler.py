"""Ledger-backed joint strategy frame assembly."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace

from mm.application.frame_freshness import (
    local_market_age_ms,
    stale_market_frame_reason,
)
from mm.application.frame_assembler import StrategyInputFrameAssembler
from mm.bus.base import AbstractEventBus
from mm.core.adapters.joint import (
    _reference_price_from_frame,
    joint_strategy_frame_from_input,
)
from mm.core.models.instrument import SymbolBinding
from mm.core.models.joint_strategy import JointStrategyFrame
from mm.core.models.ledger import JointLedgerSnapshot, LedgerFreshness
from mm.core.models.strategy import StrategyInputFrame
from mm.exchanges.shared.provider_adapter import now_ms
from mm.observability import log_event
from mm.ledger.service import PortfolioLedgerService
from mm.portfolio.wallet import StrategyWallet, joint_ledger_with_virtual_local
from mm.runtime.config import LiquiditySourcingRuntimeConfig


def _ledger_ahead_of_market_ms(
    ledger: JointLedgerSnapshot,
    *,
    market_t_end_ms: int,
) -> int | None:
    private_ms = int(ledger.local.last_private_event_ms or 0)
    if private_ms <= 0 or private_ms <= int(market_t_end_ms):
        return None
    return max(0, private_ms - int(market_t_end_ms))


class JointStrategyFrameAssembler:
    """Publish ledger-backed ``JointStrategyFrame`` on the joint topic."""

    def __init__(
        self,
        *,
        input_assembler: StrategyInputFrameAssembler,
        binding: SymbolBinding,
        liquidity_sourcing: LiquiditySourcingRuntimeConfig,
        bus: AbstractEventBus | None = None,
        joint_output_topic: str | None = None,
        ledger_service: PortfolioLedgerService | None = None,
        ledger_queue: asyncio.Queue | None = None,
        max_local_private_age_ms: int = 120_000,
        max_global_private_age_ms: int = 120_000,
        max_local_market_age_ms: int = 0,
        max_global_market_age_ms: int = 0,
        clock_ms: Callable[[], int] = now_ms,
        strategy_wallet_provider: Callable[[], StrategyWallet | None] | None = None,
    ) -> None:
        self.input_assembler = input_assembler
        self.binding = binding
        self.liquidity_sourcing = liquidity_sourcing
        self.bus = bus
        self.joint_output_topic = joint_output_topic
        self.ledger_service = ledger_service
        self.ledger_queue = ledger_queue
        self.max_local_private_age_ms = max(0, int(max_local_private_age_ms))
        self.max_global_private_age_ms = max(0, int(max_global_private_age_ms))
        self.max_local_market_age_ms = max(0, int(max_local_market_age_ms))
        self.max_global_market_age_ms = max(0, int(max_global_market_age_ms))
        self.clock_ms = clock_ms
        self.strategy_wallet_provider = strategy_wallet_provider
        self._latest_ledger: JointLedgerSnapshot | None = None
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._stopped.clear()
        await self.input_assembler.start_listeners()
        if self.ledger_queue is not None:
            self._tasks.append(asyncio.create_task(self._run_ledger()))
        self._tasks.append(asyncio.create_task(self._run_publish()))

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.input_assembler.stop()

    async def next_joint_frame(self) -> JointStrategyFrame:
        input_frame = await self.input_assembler.next_frame()
        return self.build_joint_frame(input_frame)

    def should_skip_input_frame(self, input_frame: StrategyInputFrame) -> str | None:
        now_ms = self.clock_ms()
        return stale_market_frame_reason(
            input_frame,
            now_ms=now_ms,
            max_local_market_age_ms=self.max_local_market_age_ms,
            max_global_market_age_ms=self.max_global_market_age_ms,
        )

    def build_joint_frame(self, input_frame: StrategyInputFrame) -> JointStrategyFrame:
        now_ms = self.clock_ms()
        reference_price = _reference_price_from_frame(
            input_frame,
            reference_venue=self.liquidity_sourcing.reference_venue,
        )
        physical_ledger: JointLedgerSnapshot | None = None
        if self.ledger_service is not None:
            self.ledger_service.set_reference_price(reference_price)
            physical_ledger = self._ledger_with_market_freshness(
                self.ledger_service.joint_snapshot(
                    reference_price=reference_price,
                    created_at_ms=now_ms,
                ),
                input_frame=input_frame,
                now_ms=now_ms,
            )
        elif self._latest_ledger is not None:
            physical_ledger = self._ledger_with_market_freshness(
                replace(
                    self._latest_ledger,
                    reference_price=reference_price,
                    created_at_ms=now_ms,
                ),
                input_frame=input_frame,
                now_ms=now_ms,
            )

        wallet = self.strategy_wallet_provider() if self.strategy_wallet_provider is not None else None
        if wallet is not None and physical_ledger is not None:
            strategy_ledger = joint_ledger_with_virtual_local(physical_ledger, wallet)
            ledger_source = "virtual"
        else:
            strategy_ledger = physical_ledger
            ledger_source = "physical"

        ledger_ahead_ms = (
            _ledger_ahead_of_market_ms(physical_ledger, market_t_end_ms=input_frame.local.t_end_ms)
            if physical_ledger is not None
            else None
        )

        joint = joint_strategy_frame_from_input(
            input_frame,
            self.binding,
            ledger=strategy_ledger,
            liquidity_sourcing=self.liquidity_sourcing,
            now_ms=now_ms,
            max_local_private_age_ms=self.max_local_private_age_ms,
            max_global_private_age_ms=self.max_global_private_age_ms,
            max_local_market_age_ms=self.max_local_market_age_ms,
            max_global_market_age_ms=self.max_global_market_age_ms,
            input_frame=input_frame,
            physical_ledger=physical_ledger if wallet is not None else None,
            ledger_source=ledger_source,
            sync_ledger=physical_ledger,
        )
        if ledger_ahead_ms is not None:
            return replace(
                joint,
                sync_status=replace(joint.sync_status, ledger_ahead_of_market_ms=ledger_ahead_ms),
            )
        return joint

    async def _run_publish(self) -> None:
        try:
            while not self._stopped.is_set():
                input_frame = await self.input_assembler.next_frame()
                skip_reason = self.should_skip_input_frame(input_frame)
                if skip_reason is not None:
                    log_event(
                        "joint_frame_skipped",
                        {
                            "reason": skip_reason,
                            "exchange": input_frame.local.exchange,
                            "symbol": input_frame.local.symbol,
                            "t_end_ms": input_frame.local.t_end_ms,
                            "local_market_age_ms": local_market_age_ms(
                                input_frame,
                                now_ms=self.clock_ms(),
                            ),
                            "max_local_market_age_ms": self.max_local_market_age_ms,
                            "max_global_market_age_ms": self.max_global_market_age_ms,
                        },
                    )
                    continue
                joint = self.build_joint_frame(input_frame)
                if self.bus is not None and self.joint_output_topic:
                    await self.bus.publish(self.joint_output_topic, joint)
        except asyncio.CancelledError:
            raise

    async def _run_ledger(self) -> None:
        if self.ledger_queue is None:
            return
        try:
            while not self._stopped.is_set():
                payload = await self.ledger_queue.get()
                if isinstance(payload, JointLedgerSnapshot):
                    self._latest_ledger = payload
        except asyncio.CancelledError:
            raise

    def _ledger_with_market_freshness(
        self,
        ledger: JointLedgerSnapshot,
        *,
        input_frame: StrategyInputFrame,
        now_ms: int,
    ) -> JointLedgerSnapshot:
        global_market_age = (
            max(0, now_ms - int(input_frame.global_.t_end_ms))
            if input_frame.global_ is not None
            else None
        )
        freshness = LedgerFreshness(
            local_private_age_ms=ledger.freshness.local_private_age_ms,
            global_private_age_ms=ledger.freshness.global_private_age_ms,
            local_market_age_ms=max(0, now_ms - int(input_frame.local.t_end_ms)),
            global_market_age_ms=global_market_age,
            ledger_fresh=ledger.freshness.ledger_fresh,
        )
        return replace(ledger, freshness=freshness, created_at_ms=now_ms)
