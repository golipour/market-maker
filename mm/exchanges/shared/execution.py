from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TypeVar

from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.models.execution import Order

T = TypeVar("T")
R = TypeVar("R")


async def map_in_thread_pool(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_concurrency: int = 8,
) -> list[R]:
    """Run sync callables in a bounded thread pool, preserving input order."""

    if not items:
        return []
    concurrency = max(1, int(max_concurrency))
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(item: T) -> R:
        async with semaphore:
            return await asyncio.to_thread(fn, item)

    return list(await asyncio.gather(*(_run(item) for item in items)))


class DryRunOrderCommandAdapter(AbstractOrderCommandAdapter):
    """Non-mutating command adapter for local tests and ``mm.tools.print_events`` demos."""

    def __init__(self, exchange: str):
        self.exchange = exchange.lower().strip()
        self._next_id = 1
        self.placed: list[Order] = []
        self.cancelled: list[str] = []

    async def place_orders(self, orders: list[Order]) -> list[str]:
        ids = []
        for order in orders:
            self.placed.append(order)
            ids.append(f"dry-{self.exchange}-{self._next_id}")
            self._next_id += 1
        return ids

    async def cancel_orders(self, order_ids: list[str]) -> None:
        self.cancelled.extend(order_ids)
