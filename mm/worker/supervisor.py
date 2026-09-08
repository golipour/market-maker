"""Per-strategy supervision with restart backoff and config reload."""

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
        self._reload_event = asyncio.Event()
        self._reload_done: asyncio.Event | None = None
        self._delay_s = backoff.initial_s

    async def start(self) -> None:
        self._stopped.clear()
        self._task = asyncio.create_task(self._supervise())

    async def stop(self) -> None:
        self._stopped.set()
        self._reload_event.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        await self.slot.stop()
        if self._reload_done is not None:
            self._reload_done.set()
            self._reload_done = None

    async def reload(self) -> None:
        """Rebuild the slot from ``rebuild_slot`` and wait until it is running again."""

        if self._rebuild_slot is None:
            return
        if self._task is None or self._task.done():
            await self.slot.stop()
            self.slot = self._rebuild_slot()
            return
        done = asyncio.Event()
        self._reload_done = done
        self._reload_event.set()
        await self.slot.stop()
        await done.wait()

    async def _supervise(self) -> None:
        try:
            while not self._stopped.is_set():
                if self._rebuild_slot is not None:
                    previous = self.slot
                    self.slot = self._rebuild_slot()
                    if previous is not self.slot:
                        await previous.stop()
                try:
                    await self.slot.start()
                    self._delay_s = self.backoff.initial_s
                    if self._reload_done is not None:
                        self._reload_done.set()
                        self._reload_done = None
                    await self._wait_for_slot_exit_or_reload()
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
                if self._reload_event.is_set():
                    self._reload_event.clear()
                    log_event(
                        "strategy_supervisor_reload",
                        {
                            "strategy_id": self.slot.strategy_id,
                            "account_id": self.slot.account_id,
                        },
                    )
                    continue
                log_event(
                    "strategy_supervisor_restart_scheduled",
                    {
                        "strategy_id": self.slot.strategy_id,
                        "delay_s": self._delay_s,
                    },
                )
                try:
                    await asyncio.wait_for(self._reload_event.wait(), timeout=self._delay_s)
                except asyncio.TimeoutError:
                    pass
                if self._reload_event.is_set():
                    self._reload_event.clear()
                    log_event(
                        "strategy_supervisor_reload",
                        {
                            "strategy_id": self.slot.strategy_id,
                            "account_id": self.slot.account_id,
                        },
                    )
                    continue
                self._delay_s = min(self.backoff.max_s, self._delay_s * 2.0)
        except asyncio.CancelledError:
            raise

    async def _wait_for_slot_exit_or_reload(self) -> None:
        tasks = [task for task in self.slot.monitor_tasks() if isinstance(task, asyncio.Task)]
        reload_waiter = asyncio.create_task(self._reload_event.wait())
        stop_waiter = asyncio.create_task(self._stopped.wait())
        waiters: set[asyncio.Task] = {reload_waiter, stop_waiter, *tasks}
        done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if self._reload_event.is_set() or self._stopped.is_set():
            return
        for task in done:
            if task in (reload_waiter, stop_waiter) or task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                raise exc
        raise RuntimeError(f"strategy slot {self.slot.strategy_id!r} exited unexpectedly")