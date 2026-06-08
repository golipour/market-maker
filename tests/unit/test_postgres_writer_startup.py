from __future__ import annotations

import asyncio
import json
import sys
import threading
import types

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.events import envelope_event
from mm.persistence.compaction import compact_envelope_for_persistence
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
    fake.fail_commit_count = 1
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
        retry_backoff_ms=0,
    )

    writer._write_batch([_market_envelope(1_000)])
    writer._write_batch([_market_envelope(2_000)])

    assert len(fake.connections) == 2
    first, second = fake.connections
    assert first.rollback_count == 1
    assert first.close_count == 1
    assert second.commit_count == 2
    assert second.close_count == 0


def test_postgres_writer_materializes_market_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
    )

    writer._write_batch([_market_envelope(1_000)])

    conn = fake.connections[0]
    market_call = next(
        call
        for call in conn.execute_values_calls
        if "INSERT INTO mm_market_facts" in call["sql"]
    )
    [row] = market_call["rows"]
    assert row[3] == "nobitex"
    assert row[4] == "BTC_IRT"
    assert row[5] == 1000
    assert row[7] == 1_000
    assert row[11] == 99.0
    assert row[12] == 1.25
    assert row[13] == 101.0
    assert row[14] == 1.5
    assert row[15] == 100.0
    assert row[16] == 2.0
    assert row[17] == 100.25
    assert row[18] == 100.5
    assert row[19] == 0.2
    assert row[20] == 0.3
    assert row[21] == 0.5
    assert row[22] == -0.1
    assert row[23] == 2
    assert row[24] == 1
    assert row[25] == 1
    assert row[26] is True


def test_postgres_writer_materializes_action_stage_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
    )
    envelope = compact_envelope_for_persistence(_diagnostics_envelope(), sink="postgres")

    writer._write_batch([envelope])

    conn = fake.connections[0]
    action_call = next(
        call
        for call in conn.execute_values_calls
        if "INSERT INTO mm_action_stage_facts" in call["sql"]
    )
    rows_by_stage = {row[1]: row for row in action_call["rows"]}
    target = rows_by_stage["target_book"]
    rules = rows_by_stage["rules_valid"]
    assert target[4] == "mpc-controller"
    assert target[5] == "nobitex"
    assert target[6] == "BTC_IRT"
    assert target[9] == 2_000
    assert target[10] == 2
    assert target[11] == 1
    assert target[12] == 1
    assert target[13] == 0.5
    assert target[14] == 0.4
    assert target[15] == 50.0
    assert target[16] == 44.0
    assert target[17] == 0
    assert target[18].value == {}
    assert target[19] == 2
    assert target[21]
    assert rules[10] == 1
    assert rules[17] == 1
    assert rules[18].value == {"rules_min_notional": 1}
    assert rules[20] == 1
    assert rules[22]


def test_postgres_writer_spools_batch_after_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    fake.fail_commit_count = 3
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
        retry_attempts=2,
        retry_backoff_ms=0,
        failed_batch_spool_dir=str(tmp_path),
    )
    envelope = _market_envelope(1_000)

    writer._write_batch([envelope])

    assert len(fake.connections) == 3
    assert all(conn.close_count == 1 for conn in fake.connections)
    [spool_file] = list(tmp_path.rglob("*.jsonl"))
    [record] = [
        json.loads(line)
        for line in spool_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert record["writer"] == "postgres"
    assert record["error"] == "RuntimeError"
    assert record["envelope"]["event_id"] == envelope.event_id
    assert record["envelope"]["payload"]["t_end_ms"] == 1_000


def _market_envelope(t_end_ms: int):
    return envelope_event(
        "market:events:nobitex:BTC_IRT:1000",
        {
            "type": "MarketEvent",
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "interval_ms": 1000,
            "t_start_ms": t_end_ms - 1000,
            "t_end_ms": t_end_ms,
            "orderbook": {
                "exchange": "nobitex",
                "symbol": "BTC_IRT",
                "timestamp_ms": t_end_ms,
                "bids": [{"price": 99.0, "qty": 1.25}],
                "asks": [{"price": 101.0, "qty": 1.5}],
                "is_snapshot": True,
            },
            "trades": [
                {
                    "trade_id": "t1",
                    "timestamp_ms": t_end_ms - 500,
                    "price": 100.0,
                    "qty": 0.2,
                    "side": "buy",
                    "exchange": "nobitex",
                    "symbol": "BTC_IRT",
                },
                {
                    "trade_id": "t2",
                    "timestamp_ms": t_end_ms - 250,
                    "price": 101.0,
                    "qty": 0.3,
                    "side": "sell",
                    "exchange": "nobitex",
                    "symbol": "BTC_IRT",
                },
            ],
            "buy_volume": 0.2,
            "sell_volume": 0.3,
            "vwap": 100.5,
            "imbalance": -0.1,
            "microprice": 100.25,
        },
    )


def _diagnostics_envelope():
    orders = [
        {
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "side": "buy",
            "price": 100.0,
            "qty": 0.5,
            "notional": 50.0,
            "order_type": "limit",
            "client_order_id": "buy-1",
        },
        {
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "side": "sell",
            "price": 110.0,
            "qty": 0.4,
            "notional": 44.0,
            "order_type": "limit",
            "client_order_id": "sell-1",
        },
    ]
    rejection = {
        **orders[0],
        "reason": "rules_min_notional",
        "details": {"min_notional": 100.0},
    }
    return envelope_event(
        "strategy:diagnostics:mpc-controller",
        {
            "type": "ControllerDiagnostics",
            "actor_id": "mpc-controller",
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "t_end_ms": 2_000,
            "created_at_ms": 2_000,
            "action_pipeline": {
                "target_book": {
                    "order_count": 2,
                    "buy_qty": 0.5,
                    "sell_qty": 0.4,
                    "buy_notional": 50.0,
                    "sell_notional": 44.0,
                    "orders": orders,
                    "rejection_counts": {},
                    "rejections": [],
                },
                "rules_valid": {
                    "order_count": 1,
                    "buy_qty": 0.0,
                    "sell_qty": 0.4,
                    "buy_notional": 0.0,
                    "sell_notional": 44.0,
                    "orders": orders[1:],
                    "rejection_counts": {"rules_min_notional": 1},
                    "rejections": [rejection],
                },
            },
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
        self.execute_values_calls: list[dict] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        if self.module.fail_commit_count > 0:
            self.module.fail_commit_count -= 1
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
        self.fail_commit_count = 0
        self.extras = types.ModuleType("psycopg2.extras")
        self.extras.Json = _FakeJson
        self.extras.execute_values = self._execute_values

    def connect(self, *_args, **_kwargs) -> _FakeConnection:
        conn = _FakeConnection(self)
        self.connections.append(conn)
        return conn

    def _execute_values(self, cur: _FakeCursor, _sql: str, rows: list[tuple]) -> None:
        cur.conn.execute_values_calls.append({"sql": _sql, "rows": rows})
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
