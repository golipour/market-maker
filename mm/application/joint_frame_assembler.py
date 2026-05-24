"""Ledger-backed joint strategy frame assembly alongside legacy frames."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace

from mm.application.frame_assembler import StrategyInputFrameAssembler
from mm.bus.base import AbstractEventBus
from mm.core.adapters.joint import (
    _reference_price_from_frame,
    joint_strategy_frame_from_legacy,
)
from mm.core.models.instrument import SymbolBinding
from mm.core.models.joint_strategy import JointStrategyFrame
from mm.core.models.ledger import JointLedgerSnapshot, LedgerFreshness
from mm.core.models.strategy import StrategyInputFrame
from mm.exchanges.shared.provider_adapter import now_ms
from mm.ledger.service import PortfolioLedgerService
from mm.runtime.config import LiquiditySourcingRuntimeConfig


class JointStrategyFrameAssembler:
    """Publish legacy ``StrategyInputFrame`` and ledger-backed ``JointStrategyFrame``."""

    def __init__(
        self,
        *,
        legacy_assembler: StrategyInputFrameAssembler,
        binding: SymbolBinding,
        liquidity_sourcing: LiquiditySourcingRuntimeConfig,
        bus: AbstractEventBus | None = None,
        legacy_output_topic: str | None = None,
        joint_output_topic: str | None = None,
        ledger_service: PortfolioLedgerService | None = None,
        ledger_queue: asyncio.Queue | None = None,
        max_local_private_age_ms: int = 120_000,
        max_global_private_age_ms: int = 120_000,
        clock_ms: Callable[[], int] = now_ms,
    ) -> None:
        self.legacy_assembler = legacy_assembler
        self.binding = binding
        self.liquidity_sourcing = liquidity_sourcing
        self.bus = bus
        self.legacy_output_topic = legacy_output_topic
        self.joint_output_topic = joint_output_topic
        self.ledger_service = ledger_service
        self.ledger_queue = ledger_queue
        self.max_local_private_age_ms = max(0, int(max_local_private_age_ms))
        self.max_global_private_age_ms = max(0, int(max_global_private_age_ms))
        self.clock_ms = clock_ms
        self._latest_ledger: JointLedgerSnapshot | None = None
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._stopped.clear()
        await self.legacy_assembler.start_listeners()
        if self.ledger_queue is not None:
            self._tasks.append(asyncio.create_task(self._run_ledger()))
        self._tasks.append(asyncio.create_task(self._run_publish()))

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.legacy_assembler.stop()

    async def next_joint_frame(self) -> JointStrategyFrame:
        legacy = await self.legacy_assembler.next_frame()
        return self.build_joint_frame(legacy)

    def build_joint_frame(self, legacy: StrategyInputFrame) -> JointStrategyFrame:
        now_ms = self.clock_ms()
        reference_price = _reference_price_from_frame(
            legacy,
            reference_venue=self.liquidity_sourcing.reference_venue,
        )
        if self.ledger_service is not None:
            self.ledger_service.set_reference_price(reference_price)
            ledger = self._ledger_with_market_freshness(
                self.ledger_service.joint_snapshot(
                    reference_price=reference_price,
                    created_at_ms=now_ms,
                ),
                legacy=legacy,
                now_ms=now_ms,
            )
        elif self._latest_ledger is not None:
            ledger = self._ledger_with_market_freshness(
                replace(
                    self._latest_ledger,
                    reference_price=reference_price,
                    created_at_ms=now_ms,
                ),
                legacy=legacy,
                now_ms=now_ms,
            )
        else:
            ledger = None

        return joint_strategy_frame_from_legacy(
            legacy,
            self.binding,
            ledger=ledger,
            liquidity_sourcing=self.liquidity_sourcing,
            now_ms=now_ms,
            max_local_private_age_ms=self.max_local_private_age_ms,
            max_global_private_age_ms=self.max_global_private_age_ms,
        )

    async def _run_publish(self) -> None:
        try:
            while not self._stopped.is_set():
                legacy = await self.legacy_assembler.next_frame()
                if self.bus is not None and self.legacy_output_topic:
                    await self.bus.publish(self.legacy_output_topic, legacy)
                joint = self.build_joint_frame(legacy)
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
        legacy: StrategyInputFrame,
        now_ms: int,
    ) -> JointLedgerSnapshot:
        global_market_age = (
            max(0, now_ms - int(legacy.hedge.t_end_ms))
            if legacy.hedge is not None
            else None
        )
        freshness = LedgerFreshness(
            local_private_age_ms=ledger.freshness.local_private_age_ms,
            global_private_age_ms=ledger.freshness.global_private_age_ms,
            local_market_age_ms=max(0, now_ms - int(legacy.primary.t_end_ms)),
            global_market_age_ms=global_market_age,
            ledger_fresh=ledger.freshness.ledger_fresh,
        )
        return replace(ledger, freshness=freshness, created_at_ms=now_ms)
