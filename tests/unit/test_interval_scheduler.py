from __future__ import annotations

import asyncio

import pytest

from mm.aggregation.scheduler import IntervalScheduler
from mm.core.clock import SimClock
from mm.observability.registry import REGISTRY
from mm.observability.schema import (
    MM_SCHEDULER_CALLBACK_LATENCY_MS,
    MM_SCHEDULER_CALLBACK_MISSED_DEADLINE_TOTAL,
)


@pytest.mark.asyncio
async def test_scheduler_runs_callbacks_concurrently_within_tick() -> None:
    clock = SimClock(start_ms=0)
    scheduler = IntervalScheduler(clock, max_concurrency=4)
    started = asyncio.Event()
    release = asyncio.Event()
    overlap = asyncio.Event()
    active = 0
    max_active = 0

    async def slow_callback(_t_start: int, _t_end: int, _iv: int) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        started.set()
        await release.wait()
        active -= 1

    scheduler.register(1000, slow_callback)
    scheduler.register(1000, slow_callback)
    run_task = asyncio.create_task(scheduler.run())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.sleep(0.01)
    assert max_active == 2
    release.set()
    await asyncio.sleep(0)
    scheduler.stop()
    await asyncio.wait_for(run_task, timeout=1.0)
    assert max_active >= 2


@pytest.mark.asyncio
async def test_scheduler_records_missed_deadline_when_callback_exceeds_interval() -> None:
    clock = SimClock(start_ms=0)
    scheduler = IntervalScheduler(clock, max_concurrency=1)

    async def slow_callback(_t_start: int, _t_end: int, interval_ms: int) -> None:
        clock._now_ms += interval_ms + 1

    scheduler.register(100, slow_callback)
    run_task = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0)
    scheduler.stop()
    await asyncio.wait_for(run_task, timeout=1.0)

    rendered = REGISTRY.render()
    assert MM_SCHEDULER_CALLBACK_LATENCY_MS in rendered
    assert MM_SCHEDULER_CALLBACK_MISSED_DEADLINE_TOTAL in rendered


@pytest.mark.asyncio
async def test_scheduler_runs_due_intervals_in_parallel() -> None:
    clock = SimClock(start_ms=1999)
    scheduler = IntervalScheduler(clock, max_concurrency=4)
    seen: list[int] = []

    async def record_interval(_t_start: int, t_end: int, _iv: int) -> None:
        seen.append(t_end)
        await asyncio.sleep(0)

    scheduler.register(1000, record_interval)
    scheduler.register(2000, record_interval)
    run_task = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0)
    scheduler.stop()
    await asyncio.wait_for(run_task, timeout=1.0)
    assert seen == [2000, 2000]
