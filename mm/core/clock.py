from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod


class AbstractClock(ABC):
    @abstractmethod
    def now_ms(self) -> int:
        ...

    @abstractmethod
    async def sleep_until(self, t_ms: int) -> None:
        ...


class WallClock(AbstractClock):
    def now_ms(self) -> int:
        return int(time.time() * 1000)

    async def sleep_until(self, t_ms: int) -> None:
        delay_s = max(0.0, (t_ms - self.now_ms()) / 1000.0)
        await asyncio.sleep(delay_s)


class SimClock(AbstractClock):
    def __init__(self, start_ms: int = 0):
        self._now_ms = int(start_ms)

    def now_ms(self) -> int:
        return self._now_ms

    async def sleep_until(self, t_ms: int) -> None:
        self._now_ms = max(self._now_ms, int(t_ms))
        await asyncio.sleep(0)

