from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.models.market import MarketEvent
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import GlobalMarketStatus, StrategyInputFrame
from mm.core.payload_schema import market_event_from_payload
from mm.exchanges.shared.provider_adapter import now_ms


class StrategyInputFrameAssembler:
    """Build local decision frames with an optional aligned global event."""

    def __init__(
        self,
        *,
        local_queue: asyncio.Queue[MarketEvent],
        global_queue: asyncio.Queue[MarketEvent] | None = None,
        local_private_queue: asyncio.Queue[PrivateStateSnapshot] | None = None,
        global_private_queue: asyncio.Queue[PrivateStateSnapshot] | None = None,
        bus: AbstractEventBus | None = None,
        output_topic: str | None = None,
        max_global_lag_ms: int = 2_000,
        global_grace_ms: int = 250,
        global_history_size: int = 128,
        clock_ms: Callable[[], int] = now_ms,
    ):
        self.local_queue = local_queue
        self.global_queue = global_queue
        self.local_private_queue = local_private_queue
        self.global_private_queue = global_private_queue
        self.bus = bus
        self.output_topic = output_topic
        self.max_global_lag_ms = max(0, int(max_global_lag_ms))
        self.global_grace_ms = max(0, int(global_grace_ms))
        self.clock_ms = clock_ms
        self._global_events: deque[MarketEvent] = deque(maxlen=max(1, int(global_history_size)))
        self._local_private: PrivateStateSnapshot | None = None
        self._global_private: PrivateStateSnapshot | None = None
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        """Start global/private listeners and the local publish loop."""

        self._stopped.clear()
        await self.start_listeners()
        self._tasks.append(asyncio.create_task(self._run_local()))

    async def start_listeners(self) -> None:
        """Start global and private snapshot listeners without the publish loop."""

        self._stopped.clear()
        self._tasks = []
        if self.global_queue is not None:
            self._tasks.append(asyncio.create_task(self._run_global()))
        if self.local_private_queue is not None:
            self._tasks.append(asyncio.create_task(self._run_private(self.local_private_queue, "local")))
        if self.global_private_queue is not None:
            self._tasks.append(asyncio.create_task(self._run_private(self.global_private_queue, "global")))

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def next_frame(self) -> StrategyInputFrame:
        local = _require_market_event(await self.local_queue.get())
        return await self.build_frame(local)

    async def build_frame(self, local: MarketEvent) -> StrategyInputFrame:
        self._drain_latest_inputs()
        if self.global_queue is not None:
            await self._wait_for_matching_global(local.t_end_ms)
        self._drain_latest_inputs()
        global_event, global_status, global_lag_ms = self._select_global(local)
        return StrategyInputFrame(local=local, global_=global_event,
            local_private=self._local_private,
            global_private=self._global_private,
            global_status=global_status,
            global_lag_ms=global_lag_ms,
            max_global_lag_ms=self.max_global_lag_ms,
            created_at_ms=self.clock_ms(),
        )

    async def _run_local(self) -> None:
        try:
            while not self._stopped.is_set():
                frame = await self.next_frame()
                if self.bus is not None and self.output_topic:
                    await self.bus.publish(self.output_topic, frame)
        except asyncio.CancelledError:
            raise

    async def _run_global(self) -> None:
        if self.global_queue is None:
            return
        try:
            while not self._stopped.is_set():
                self._global_events.append(_require_market_event(await self.global_queue.get()))
        except asyncio.CancelledError:
            raise

    async def _run_private(self, queue: asyncio.Queue[PrivateStateSnapshot], role: str) -> None:
        try:
            while not self._stopped.is_set():
                snapshot = await queue.get()
                if role == "local":
                    self._local_private = snapshot
                else:
                    self._global_private = snapshot
        except asyncio.CancelledError:
            raise

    def _drain_latest_inputs(self) -> None:
        if self.global_queue is not None:
            while True:
                try:
                    self._global_events.append(_require_market_event(self.global_queue.get_nowait()))
                except asyncio.QueueEmpty:
                    break
        if self.local_private_queue is not None:
            while True:
                try:
                    self._local_private = self.local_private_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        if self.global_private_queue is not None:
            while True:
                try:
                    self._global_private = self.global_private_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def _wait_for_matching_global(self, local_t_end_ms: int) -> None:
        if self.global_grace_ms <= 0:
            return
        deadline = self.clock_ms() + self.global_grace_ms
        while self.clock_ms() < deadline:
            if any(event.t_end_ms == local_t_end_ms for event in self._global_events):
                return
            await asyncio.sleep(0.005)

    def _select_global(self, local: MarketEvent) -> tuple[MarketEvent | None, GlobalMarketStatus, int | None]:
        if self.global_queue is None:
            return None, "disabled", None
        candidates = [event for event in self._global_events if event.t_end_ms <= local.t_end_ms]
        if not candidates:
            return None, "missing", None
        global_event = max(candidates, key=lambda event: event.t_end_ms)
        global_lag_ms = local.t_end_ms - global_event.t_end_ms
        if global_lag_ms > self.max_global_lag_ms:
            return None, "stale", global_lag_ms
        return global_event, "ready", global_lag_ms


def _require_market_event(event: Any) -> MarketEvent:
    if isinstance(event, MarketEvent):
        return event
    parsed = market_event_from_payload(event)
    if parsed is None:
        raise TypeError(f"expected versioned MarketEvent payload, got {type(event)!r}")
    return parsed
