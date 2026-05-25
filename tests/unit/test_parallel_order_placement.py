from __future__ import annotations

import asyncio
import threading
import time

import pytest

from mm.exchanges.shared.execution import map_in_thread_pool


def test_map_in_thread_pool_preserves_order() -> None:
    async def _run() -> None:
        seen: list[int] = []

        def _record(value: int) -> int:
            seen.append(value)
            return value * 2

        out = await map_in_thread_pool(_record, [1, 2, 3, 4], max_concurrency=2)
        assert out == [2, 4, 6, 8]
        assert sorted(seen) == [1, 2, 3, 4]

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_map_in_thread_pool_empty_input() -> None:
    assert await map_in_thread_pool(lambda x: x, [], max_concurrency=4) == []


@pytest.mark.asyncio
async def test_map_in_thread_pool_stops_starting_work_after_cancellation() -> None:
    started: list[int] = []
    lock = threading.Lock()

    def _slow(value: int) -> int:
        with lock:
            started.append(value)
        time.sleep(0.05)
        return value

    task = asyncio.create_task(map_in_thread_pool(_slow, list(range(10)), max_concurrency=2))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.08)
    assert len(started) <= 2
