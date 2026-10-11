from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import queue
import threading
from typing import TypeVar, cast

from mm.core.interfaces.execution import AbstractOrderCommandAdapter
from mm.core.models.execution import Order

T = TypeVar("T")
R = TypeVar("R")


async def map_in_thread_pool(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_concurrency: int = 32,
) -> list[R]:
    """Run sync callables in a bounded worker pool, preserving input order.

    The implementation avoids ``asyncio`` executor futures because some
    restricted runtimes allow threads but never wake the event loop from
    executor callbacks.
    """

    if not items:
        return []
    concurrency = min(len(items), max(1, int(max_concurrency)))
    work: queue.Queue[tuple[int, T]] = queue.Queue()
    for idx, item in enumerate(items):
        work.put((idx, item))
    results: list[R | None] = [None] * len(items)
    errors: list[BaseException] = []
    stop = threading.Event()

    def _worker() -> None:
        while not stop.is_set():
            try:
                idx, item = work.get_nowait()
            except queue.Empty:
                return
            try:
                if stop.is_set():
                    return
                results[idx] = fn(item)
            except BaseException as exc:  # propagate after active workers settle
                errors.append(exc)
                stop.set()
            finally:
                work.task_done()

    threads = [
        threading.Thread(target=_worker, daemon=True, name="mm-order-worker")
        for _ in range(concurrency)
    ]
    for thread in threads:
        thread.start()

    try:
        while any(thread.is_alive() for thread in threads):
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        stop.set()
        raise

    if errors:
        raise errors[0]
    return [cast(R, result) for result in results]


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
