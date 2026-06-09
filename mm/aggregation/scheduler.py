from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable

from mm.core.clock import AbstractClock

IntervalCallback = Callable[[int, int, int], Awaitable[None] | None]


class IntervalScheduler:
    def __init__(self, clock: AbstractClock, *, max_concurrency: int = 8):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._clock = clock
        self._max_concurrency = int(max_concurrency)
        self._callbacks: dict[int, list[IntervalCallback]] = defaultdict(list)
        self._stopped = asyncio.Event()

    def register(self, interval_ms: int, callback: IntervalCallback) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self._callbacks[int(interval_ms)].append(callback)

    def stop(self) -> None:
        self._stopped.set()

    async def run(self) -> None:
        if not self._callbacks:
            return
        next_fire = {
            iv: ((self._clock.now_ms() // iv) + 1) * iv
            for iv in self._callbacks
        }
        while not self._stopped.is_set():
            target = min(next_fire.values())
            await self._clock.sleep_until(target)
            now = self._clock.now_ms()
            due_ticks: list[tuple[int, int]] = []
            for iv, t_fire in list(next_fire.items()):
                if now < t_fire:
                    continue
                due_ticks.append((iv, t_fire))
                next_fire[iv] = t_fire + iv
            if due_ticks:
                await asyncio.gather(*(self._run_tick(iv, t_fire) for iv, t_fire in due_ticks))

    async def _run_tick(self, interval_ms: int, t_fire: int) -> None:
        t_start = t_fire - interval_ms
        callbacks = self._callbacks[interval_ms]
        if self._max_concurrency == 1:
            for callback in callbacks:
                await self._invoke_callback(callback, interval_ms, t_start, t_fire)
            return
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _limited(callback: IntervalCallback) -> None:
            async with semaphore:
                await self._invoke_callback(callback, interval_ms, t_start, t_fire)

        await asyncio.gather(*(_limited(callback) for callback in callbacks))

    async def _invoke_callback(
        self,
        callback: IntervalCallback,
        interval_ms: int,
        t_start: int,
        t_fire: int,
    ) -> None:
        started_ms = self._clock.now_ms()
        result = callback(t_start, t_fire, interval_ms)
        if result is not None:
            await result
        elapsed_ms = max(0, self._clock.now_ms() - started_ms)
        from mm.observability import scheduler_metrics

        scheduler_metrics.record_callback(
            interval_ms=interval_ms,
            callback=_callback_name(callback),
            latency_ms=float(elapsed_ms),
            missed_deadline=elapsed_ms > interval_ms,
        )


def _callback_name(callback: IntervalCallback) -> str:
    return str(getattr(callback, "__qualname__", repr(callback)))
