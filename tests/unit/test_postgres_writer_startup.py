from __future__ import annotations

import asyncio
import threading

import pytest

from mm.bus.memory import InMemoryBus
from mm.persistence.postgres import PostgresEventWriter


@pytest.mark.asyncio
async def test_postgres_writer_start_does_not_wait_for_readiness_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = InMemoryBus()
    writer = PostgresEventWriter(
        bus=bus,
        topics=("strategy:decisions",),
        database_url="postgresql://example.invalid/db",
        connect_timeout_s=1,
    )
    probe_started = threading.Event()
    release_probe = threading.Event()

    def _slow_probe() -> bool:
        probe_started.set()
        release_probe.wait(timeout=1.0)
        return False

    monkeypatch.setattr(writer, "_ensure_ready", _slow_probe)

    await asyncio.wait_for(writer.start(), timeout=0.1)
    assert probe_started.wait(timeout=1.0)

    release_probe.set()
    await writer.stop()
