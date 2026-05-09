from __future__ import annotations

import asyncio

from mm.bus.base import AbstractEventBus
from mm.core.models.execution import Fill
from mm.core.models.private import OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.execution.order_manager import OrderManager


class ExecutionStateReconciler:
    """Mirrors private exchange state into the execution manager."""

    def __init__(self, exchange: str, manager: OrderManager, bus: AbstractEventBus):
        self.exchange = exchange.lower().strip()
        self.manager = manager
        self.bus = bus
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._listen_snapshot()),
            asyncio.create_task(self._listen_open_orders()),
            asyncio.create_task(self._listen_order_updates()),
            asyncio.create_task(self._listen_fills()),
        ]
        await asyncio.sleep(0)
        await self.manager.publish_snapshot()

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _listen_snapshot(self) -> None:
        async for event in self.bus.subscribe(f"private:state:{self.exchange}:snapshot"):
            if isinstance(event, PrivateStateSnapshot):
                snapshot = self.manager.apply_private_snapshot(event)
                await self.bus.publish(f"execution:state:{self.exchange}", snapshot)

    async def _listen_open_orders(self) -> None:
        async for event in self.bus.subscribe(f"private:state:{self.exchange}:open_orders"):
            if isinstance(event, OpenOrder):
                snapshot = self.manager.apply_open_order(event)
                await self.bus.publish(f"execution:state:{self.exchange}", snapshot)

    async def _listen_order_updates(self) -> None:
        async for event in self.bus.subscribe(f"private:state:{self.exchange}:order_updates"):
            if isinstance(event, OrderUpdate):
                snapshot = self.manager.apply_order_update(event)
                await self.bus.publish(f"execution:state:{self.exchange}", snapshot)

    async def _listen_fills(self) -> None:
        async for event in self.bus.subscribe(f"private:state:{self.exchange}:fills"):
            if isinstance(event, Fill):
                snapshot = self.manager.apply_fill(event)
                await self.bus.publish(f"execution:state:{self.exchange}", snapshot)
