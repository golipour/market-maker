from __future__ import annotations

import asyncio

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
        assert seen == [1, 2, 3, 4]

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_map_in_thread_pool_empty_input() -> None:
    assert await map_in_thread_pool(lambda x: x, [], max_concurrency=4) == []
