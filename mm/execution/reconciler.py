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
            asyncio.create_task(self._listen_cancel_all()),
            asyncio.create_task(self._listen_cancel_actor()),
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

    async def _listen_cancel_all(self) -> None:
        topics = (f"execution:cancel_all:{self.exchange}", "execution:cancel_all:all")
        queues = [self.bus.subscribe_queue(topic) for topic in topics]
        while True:
            tasks = [asyncio.create_task(queue.get()) for queue in queues]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                event = task.result()
                if isinstance(event, dict):
                    await self.manager.cancel_all(reason=str(event.get("reason", "")))

    async def _listen_cancel_actor(self) -> None:
        queue = self.bus.subscribe_queue(f"execution:cancel_actor:{self.exchange}")
        while True:
            event = await queue.get()
            if not isinstance(event, dict):
                continue
            actor_id = event.get("actor_id")
            if not actor_id:
                continue
            cancel_actor = getattr(self.manager, "cancel_actor_orders", None)
            if cancel_actor is not None:
                await cancel_actor(str(actor_id), reason=str(event.get("reason", "")))
