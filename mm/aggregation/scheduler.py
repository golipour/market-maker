from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable

from mm.core.clock import AbstractClock

IntervalCallback = Callable[[int, int, int], Awaitable[None] | None]


class IntervalScheduler:
    def __init__(self, clock: AbstractClock):
        self._clock = clock
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
            for iv, t_fire in list(next_fire.items()):
                if now < t_fire:
                    continue
                t_start = t_fire - iv
                for callback in self._callbacks[iv]:
                    result = callback(t_start, t_fire, iv)
                    if result is not None:
                        await result
                next_fire[iv] = t_fire + iv

