"""Per-strategy supervision with restart backoff."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from mm.observability import log_event
from mm.worker.strategy_slot import StrategySlot


@dataclass(frozen=True, slots=True)
class StrategyBackoffConfig:
    initial_s: float = 1.0
    max_s: float = 60.0


class StrategySupervisor:
    """Restart an isolated strategy slot when its runtime tasks exit unexpectedly."""

    def __init__(
        self,
        *,
        slot: StrategySlot,
        backoff: StrategyBackoffConfig,
        rebuild_slot: Callable[[], StrategySlot] | None = None,
    ) -> None:
        self.slot = slot
        self.backoff = backoff
        self._rebuild_slot = rebuild_slot
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self._delay_s = backoff.initial_s

    async def start(self) -> None:
        self._stopped.clear()
        self._task = asyncio.create_task(self._supervise())

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        await self.slot.stop()

    async def _supervise(self) -> None:
        try:
            while not self._stopped.is_set():
                if self._rebuild_slot is not None:
                    self.slot = self._rebuild_slot()
                try:
                    await self.slot.start()
                    await self._wait_for_slot_exit()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log_event(
                        "strategy_supervisor_slot_error",
                        {
                            "strategy_id": self.slot.strategy_id,
                            "account_id": self.slot.account_id,
                            "error": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                finally:
                    await self.slot.stop()
                if self._stopped.is_set():
                    break
                log_event(
                    "strategy_supervisor_restart_scheduled",
                    {
                        "strategy_id": self.slot.strategy_id,
                        "delay_s": self._delay_s,
                    },
                )
                await asyncio.sleep(self._delay_s)
                self._delay_s = min(self.backoff.max_s, self._delay_s * 2.0)
        except asyncio.CancelledError:
            raise

    async def _wait_for_slot_exit(self) -> None:
        tasks = [task for task in self.slot.monitor_tasks() if isinstance(task, asyncio.Task)]
        if not tasks:
            await self._stopped.wait()
            return
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                raise exc
        raise RuntimeError(f"strategy slot {self.slot.strategy_id!r} exited unexpectedly")
