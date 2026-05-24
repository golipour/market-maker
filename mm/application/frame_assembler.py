from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable

from mm.bus.base import AbstractEventBus
from mm.core.models.market import MarketEvent
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import HedgeStatus, StrategyInputFrame
from mm.exchanges.shared.provider_adapter import now_ms


class StrategyInputFrameAssembler:
    """Build primary decision frames with an optional aligned hedge event."""

    def __init__(
        self,
        *,
        primary_queue: asyncio.Queue[MarketEvent],
        hedge_queue: asyncio.Queue[MarketEvent] | None = None,
        primary_private_queue: asyncio.Queue[PrivateStateSnapshot] | None = None,
        hedge_private_queue: asyncio.Queue[PrivateStateSnapshot] | None = None,
        bus: AbstractEventBus | None = None,
        output_topic: str | None = None,
        max_hedge_lag_ms: int = 2_000,
        hedge_grace_ms: int = 250,
        hedge_history_size: int = 128,
        clock_ms: Callable[[], int] = now_ms,
    ):
        self.primary_queue = primary_queue
        self.hedge_queue = hedge_queue
        self.primary_private_queue = primary_private_queue
        self.hedge_private_queue = hedge_private_queue
        self.bus = bus
        self.output_topic = output_topic
        self.max_hedge_lag_ms = max(0, int(max_hedge_lag_ms))
        self.hedge_grace_ms = max(0, int(hedge_grace_ms))
        self.clock_ms = clock_ms
        self._hedge_events: deque[MarketEvent] = deque(maxlen=max(1, int(hedge_history_size)))
        self._primary_private: PrivateStateSnapshot | None = None
        self._hedge_private: PrivateStateSnapshot | None = None
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        """Start hedge/private listeners and the primary publish loop."""

        self._stopped.clear()
        await self.start_listeners()
        self._tasks.append(asyncio.create_task(self._run_primary()))

    async def start_listeners(self) -> None:
        """Start hedge and private snapshot listeners without the publish loop."""

        self._stopped.clear()
        self._tasks = []
        if self.hedge_queue is not None:
            self._tasks.append(asyncio.create_task(self._run_hedge()))
        if self.primary_private_queue is not None:
            self._tasks.append(asyncio.create_task(self._run_private(self.primary_private_queue, "primary")))
        if self.hedge_private_queue is not None:
            self._tasks.append(asyncio.create_task(self._run_private(self.hedge_private_queue, "hedge")))

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def next_frame(self) -> StrategyInputFrame:
        primary = await self.primary_queue.get()
        return await self.build_frame(primary)

    async def build_frame(self, primary: MarketEvent) -> StrategyInputFrame:
        self._drain_latest_inputs()
        if self.hedge_queue is not None:
            await self._wait_for_matching_hedge(primary.t_end_ms)
        self._drain_latest_inputs()
        hedge, hedge_status, hedge_lag_ms = self._select_hedge(primary)
        return StrategyInputFrame(
            primary=primary,
            hedge=hedge,
            primary_private=self._primary_private,
            hedge_private=self._hedge_private,
            hedge_status=hedge_status,
            hedge_lag_ms=hedge_lag_ms,
            max_hedge_lag_ms=self.max_hedge_lag_ms,
            created_at_ms=self.clock_ms(),
        )

    async def _run_primary(self) -> None:
        try:
            while not self._stopped.is_set():
                frame = await self.next_frame()
                if self.bus is not None and self.output_topic:
                    await self.bus.publish(self.output_topic, frame)
        except asyncio.CancelledError:
            raise

    async def _run_hedge(self) -> None:
        if self.hedge_queue is None:
            return
        try:
            while not self._stopped.is_set():
                self._hedge_events.append(await self.hedge_queue.get())
        except asyncio.CancelledError:
            raise

    async def _run_private(self, queue: asyncio.Queue[PrivateStateSnapshot], role: str) -> None:
        try:
            while not self._stopped.is_set():
                snapshot = await queue.get()
                if role == "primary":
                    self._primary_private = snapshot
                else:
                    self._hedge_private = snapshot
        except asyncio.CancelledError:
            raise

    def _drain_latest_inputs(self) -> None:
        if self.hedge_queue is not None:
            while True:
                try:
                    self._hedge_events.append(self.hedge_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
        if self.primary_private_queue is not None:
            while True:
                try:
                    self._primary_private = self.primary_private_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        if self.hedge_private_queue is not None:
            while True:
                try:
                    self._hedge_private = self.hedge_private_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def _wait_for_matching_hedge(self, primary_t_end_ms: int) -> None:
        if self.hedge_grace_ms <= 0:
            return
        deadline = self.clock_ms() + self.hedge_grace_ms
        while self.clock_ms() < deadline:
            if any(event.t_end_ms == primary_t_end_ms for event in self._hedge_events):
                return
            await asyncio.sleep(0.005)

    def _select_hedge(self, primary: MarketEvent) -> tuple[MarketEvent | None, HedgeStatus, int | None]:
        if self.hedge_queue is None:
            return None, "disabled", None
        candidates = [event for event in self._hedge_events if event.t_end_ms <= primary.t_end_ms]
        if not candidates:
            return None, "missing", None
        hedge = max(candidates, key=lambda event: event.t_end_ms)
        hedge_lag_ms = primary.t_end_ms - hedge.t_end_ms
        if hedge_lag_ms > self.max_hedge_lag_ms:
            return None, "stale", hedge_lag_ms
        return hedge, "ready", hedge_lag_ms
