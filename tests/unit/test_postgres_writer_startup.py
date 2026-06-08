from __future__ import annotations

import asyncio
import sys
import threading
import types

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.events import envelope_event
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


def test_postgres_writer_reuses_connection_across_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
    )

    writer._write_batch([_market_envelope(1_000)])
    writer._write_batch([_market_envelope(2_000)])

    assert len(fake.connections) == 1
    conn = fake.connections[0]
    assert conn.commit_count == 2
    assert conn.close_count == 0
    assert conn.rollback_count == 0
    assert conn.ddl_count == 1

    writer._close_connection()
    assert conn.close_count == 1


def test_postgres_writer_reconnects_after_failed_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    fake.fail_commit_once = True
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
    )

    writer._write_batch([_market_envelope(1_000)])
    writer._write_batch([_market_envelope(2_000)])

    assert len(fake.connections) == 2
    first, second = fake.connections
    assert first.rollback_count == 1
    assert first.close_count == 1
    assert second.commit_count == 1
    assert second.close_count == 0


def _market_envelope(t_end_ms: int):
    return envelope_event(
        "market:events:nobitex:BTC_IRT:1000",
        {
            "type": "MarketEvent",
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "t_end_ms": t_end_ms,
            "trades": [],
        },
    )


class _FakeCursor:
    def __init__(self, conn: "_FakeConnection") -> None:
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql: str) -> None:
        if "CREATE TABLE" in sql:
            self.conn.ddl_count += 1


class _FakeConnection:
    def __init__(self, module: "_FakePsycopg2") -> None:
        self.module = module
        self.closed = False
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0
        self.ddl_count = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        if self.module.fail_commit_once:
            self.module.fail_commit_once = False
            raise RuntimeError("commit failed")
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


class _FakePsycopg2(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("psycopg2")
        self.connections: list[_FakeConnection] = []
        self.fail_commit_once = False
        self.extras = types.ModuleType("psycopg2.extras")
        self.extras.Json = _FakeJson
        self.extras.execute_values = self._execute_values

    def connect(self, *_args, **_kwargs) -> _FakeConnection:
        conn = _FakeConnection(self)
        self.connections.append(conn)
        return conn

    def _execute_values(self, cur: _FakeCursor, _sql: str, rows: list[tuple]) -> None:
        cur.conn.last_execute_values_rows = rows


class _FakeJson:
    def __init__(self, value, dumps=None) -> None:
        self.value = value
        self.dumps = dumps


def _install_fake_psycopg2(monkeypatch: pytest.MonkeyPatch) -> _FakePsycopg2:
    fake = _FakePsycopg2()
    monkeypatch.setitem(sys.modules, "psycopg2", fake)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", fake.extras)
    return fake
