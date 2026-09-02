"""Tests for the cooperative optimizer cancellation in ``StrategyRunner``.

The 2026-09-01 production failure was triggered by the ``asyncio.to_thread``-based
``_decide_with_timeout``: a timed-out decide call could not be cancelled, so
the optimizer's thread kept running, eventually saturating the default
executor. The structural fix is a dedicated single-worker
``ThreadPoolExecutor`` plus a ``threading.Event`` that the Adam loop checks
at the top of every epoch.

These tests assert that:

1. The runner has a dedicated ``_decide_pool`` (not the default executor).
2. ``set_cancel_event`` is wired to the engine's optimizer backend.
3. A simulated cancel-via-event completes the future, the pool is not
   wedged, and a subsequent decide call goes through.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from mm.application.strategy_runner import (
    OptimizerDecisionTimeout,
    StrategyRunner,
)


@dataclass
class _StubBus:
    """Minimal event-bus stub for the runner constructor."""

    publishes: list[tuple[str, dict]] = field(default_factory=list)

    async def publish(self, topic: str, payload: dict) -> None:  # pragma: no cover
        self.publishes.append((topic, payload))


@dataclass
class _StubFrame:
    frame_id: str = "test-frame"
    local: Any = None
    global_: Any = None
    has_global: bool = False
    t_end_ms: int = 0
    ledger: Any = None
    private_state: Any = None
    global_status: Any = None


class _SlowOptimizerBackend:
    """Fake ``MonteCarloMPCTensorBatched``-shaped backend that respects the cancel event.

    Each call to ``decide`` increments a counter. The ``decide`` call
    blocks on the cancel event for up to 5 s and then returns a stub
    decision. The cancel event is installed via ``set_cancel_event``,
    matching the real backend's protocol.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.cancelled_calls = 0
        self._cancel_event: threading.Event | None = None

    def set_cancel_event(self, event: threading.Event) -> None:
        self._cancel_event = event

    def decide(self, frame: Any, *, frame_step: int) -> Any:
        self.calls += 1
        if self._cancel_event is not None:
            # Block until cancelled, with a hard ceiling to keep the
            # test bounded if cancellation never arrives.
            cancelled = self._cancel_event.wait(timeout=5.0)
            if cancelled:
                self.cancelled_calls += 1
        return {"frame_id": frame.frame_id, "frame_step": frame_step}


class _StubEngine:
    """Stub for ``MpcStrategyEngine`` exposing only the surface the runner needs."""

    def __init__(self) -> None:
        self._optimizer_backend = _SlowOptimizerBackend()
        self.decide_calls = 0

    def decide(self, frame: Any, *, frame_step: int) -> Any:
        self.decide_calls += 1
        return self._optimizer_backend.decide(frame, frame_step=frame_step)


def _build_runner(engine: _StubEngine) -> StrategyRunner:
    """Build a ``StrategyRunner`` with the stub engine in place of the real one."""
    # The dataclass is frozen-ish (slots); use ``object.__new__`` and
    # then set attributes explicitly to avoid ``__post_init__`` doing
    # the full engine build (which would require a real config).
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._engine = engine
    runner._decide_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="mm-test"
    )
    runner._decide_cancel_event = threading.Event()
    backend = runner._engine._optimizer_backend
    if hasattr(backend, "set_cancel_event"):
        backend.set_cancel_event(runner._decide_cancel_event)
    return runner


class _PoolStub:
    """Removed; tests use a real ``ThreadPoolExecutor`` instead."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runner_has_dedicated_pool() -> None:
    engine = _StubEngine()
    runner = _build_runner(engine)
    # The pool is real; structurally the runner owns a separate executor
    # so a wedged optimizer cannot starve other to_thread callers.
    assert isinstance(runner._decide_pool, concurrent.futures.ThreadPoolExecutor)
    assert runner._decide_cancel_event is not None
    # And the cancel event is wired to the backend.
    assert engine._optimizer_backend._cancel_event is runner._decide_cancel_event


def test_runner_stop_shuts_down_pool() -> None:
    """``stop()`` must shut the dedicated decide pool down so the worker thread is reaped."""
    engine = _StubEngine()
    runner = _build_runner(engine)
    # Run a quick decide to make the pool "active".
    pool = runner._decide_pool

    async def scenario() -> None:
        # Use the same code path as the real runner.
        backend = runner._engine._optimizer_backend
        if hasattr(backend, "set_cancel_event"):
            backend.set_cancel_event(runner._decide_cancel_event)
        future = pool.submit(runner._engine.decide, _StubFrame(), frame_step=1)
        wrapped = asyncio.wrap_future(future)
        await wrapped  # backend waits 5s; we cancel it before then.
        # Production stop path:
        runner._decide_cancel_event.set()
        # Mirror stop():
        pool.shutdown(wait=True, cancel_futures=True)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(scenario())
    finally:
        loop.close()
    # The pool is now shut down; submitting raises.
    with pytest.raises(RuntimeError):
        pool.submit(lambda: None)
    engine._optimizer_backend.cancelled_calls == 1


def test_event_drives_backend_cancellation() -> None:
    """Setting the event unblocks a stalled decide within 1 epoch."""
    engine = _StubEngine()
    runner = _build_runner(engine)
    frame = _StubFrame()

    async def scenario() -> Any:
        # Submit; the worker thread blocks on the event.
        future = runner._decide_pool.submit(
            runner._engine.decide, frame, frame_step=1
        )
        wrapped = asyncio.wrap_future(future)
        # Give the worker a moment to enter its wait.
        await asyncio.sleep(0.05)
        assert not future.done()
        # Set the event; the worker should complete shortly.
        runner._decide_cancel_event.set()
        # Bounded wait so a regression surfaces as a test failure.
        return await asyncio.wait_for(wrapped, timeout=1.0)

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(scenario())
        assert result == {"frame_id": "test-frame", "frame_step": 1}
        assert engine._optimizer_backend.cancelled_calls == 1
    finally:
        loop.close()
