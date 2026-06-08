from __future__ import annotations

import asyncio
import json
import sys
import threading
import types

import pytest

from mm.bus.memory import InMemoryBus
from mm.core.events import envelope_event
from mm.core.models.execution import Fill, Order
from mm.core.models.private import OpenOrder
from mm.core.models.target_book import JointTargetBook, TargetOrder, VenueTargetBook
from mm.execution.order_manager import ExecutionSnapshot, PendingOrderChange
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


def test_postgres_writer_materializes_model_and_optimizer_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
    )
    envelope = envelope_event(
        "strategy:diagnostics:mpc-controller",
        {
            "type": "ControllerDiagnostics",
            "actor_id": "mpc-controller",
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "t_end_ms": 2_000,
            "frame_step": 12,
            "created_at_ms": 2_100,
            "models": {
                "price": {
                    "kind": "normal",
                    "updated": True,
                    "observation": {"delta": 1.25, "mid": 100.0},
                    "params_before": {"mu": 0.0},
                    "params_after": {"mu": 0.1, "sigma": 2.0},
                    "log_likelihood": -1.25,
                    "nll": 1.25,
                    "standardized_residual": 0.5,
                }
            },
            "optimizer": {
                "config": {"kind": "monte_carlo"},
                "timing": {
                    "controller_total_ms": 12.0,
                    "optimize_ms": 7.5,
                    "clean_ms": 2.0,
                    "model_score_ms": 0.5,
                },
                "objective": {
                    "initial": {
                        "value": 1.0,
                        "loss": -1.0,
                        "components": {
                            "risk_adjusted_return": 2.0,
                            "threshold_shortfall_penalty": -1.0,
                        },
                        "contribution_fractions": {
                            "risk_adjusted_return": 0.67,
                            "threshold_shortfall_penalty": 0.33,
                        },
                    },
                    "final": {
                        "value": 3.0,
                        "loss": -3.0,
                        "components": {
                            "risk_adjusted_return": 4.0,
                            "threshold_shortfall_penalty": -1.0,
                        },
                        "contribution_fractions": {
                            "risk_adjusted_return": 0.8,
                            "threshold_shortfall_penalty": 0.2,
                        },
                    },
                    "best": {
                        "value": 3.5,
                        "loss": -3.5,
                        "expected_total_reward": 10.0,
                        "expected_total_return": 0.03,
                        "expected_next_reward": 1.5,
                        "main_reward_mean": 2.5,
                        "global_reward_mean": 0.1,
                        "components": {
                            "risk_adjusted_return": 4.5,
                            "threshold_shortfall_penalty": -1.0,
                        },
                        "contribution_fractions": {
                            "risk_adjusted_return": 0.82,
                            "threshold_shortfall_penalty": 0.18,
                        },
                    },
                    "improvement": {
                        "best_minus_initial": 2.5,
                        "final_minus_initial": 2.0,
                    },
                },
                "gradients": {"issues": ["nan_clipped"]},
                "rollout": {"sample_count": 64, "horizon": 4},
            },
        },
    )

    writer._write_batch([envelope])

    [model_row] = _model_score_fact_rows(fake)
    assert model_row[1] == "price"
    assert model_row[4] == "mpc-controller"
    assert model_row[5] == "nobitex"
    assert model_row[6] == "BTC_IRT"
    assert model_row[7] == 2_100
    assert model_row[9] == 2_000
    assert model_row[10] == 12
    assert model_row[11] == "normal"
    assert model_row[12] is True
    assert model_row[14] == -1.25
    assert model_row[15] == 1.25
    assert model_row[16] == 0.5
    assert model_row[17] == 2
    assert model_row[18] == 1
    assert model_row[19] == 2

    [optimizer_row] = _optimizer_fact_rows(fake)
    assert optimizer_row[3] == "mpc-controller"
    assert optimizer_row[10] == "monte_carlo"
    assert optimizer_row[11] == 12.0
    assert optimizer_row[12] == 7.5
    assert optimizer_row[13] == 2.0
    assert optimizer_row[14] == 0.5
    assert optimizer_row[20] == 1.0
    assert optimizer_row[21] == 3.0
    assert optimizer_row[22] == 3.5
    assert optimizer_row[26] == 2.5
    assert optimizer_row[27] == 2.0
    assert optimizer_row[28] == 10.0
    assert optimizer_row[29] == 0.03
    assert optimizer_row[33] == 1
    assert optimizer_row[34] == 64

    component_rows = _optimizer_component_fact_rows(fake)
    best_return = next(
        row
        for row in component_rows
        if row[1] == "best" and row[2] == "risk_adjusted_return"
    )
    assert best_return[5] == "mpc-controller"
    assert best_return[10] == 2_000
    assert best_return[11] == 12
    assert best_return[12] == 4.5
    assert best_return[13] == 0.82


def test_postgres_writer_materializes_target_order_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
    )
    envelope = envelope_event(
        "strategy:target_books:mpc-controller",
        JointTargetBook(
            actor_id="mpc-controller",
            mode="liquidity_sourcing",
            frame_id="frame-1",
            decision_id="decision-1",
            target_book_id="target-1",
            local=VenueTargetBook(
                exchange="nobitex",
                symbol="BTC_IRT",
                actor_id="mpc-controller",
                policy="reconcile",
                frame_id="frame-1",
                decision_id="decision-1",
                target_book_id="target-1",
                execution_plan_id="plan-local",
                orders=(
                    TargetOrder(
                        "nobitex",
                        "BTC_IRT",
                        "buy",
                        99.0,
                        0.5,
                        tags=("local", "risk_approved"),
                    ),
                ),
            ),
            global_=VenueTargetBook(
                exchange="wallex",
                symbol="BTC_IRT",
                actor_id="mpc-controller",
                policy="replace",
                frame_id="frame-1",
                decision_id="decision-1",
                target_book_id="target-1",
                execution_plan_id="plan-global",
                orders=(
                    TargetOrder(
                        "wallex",
                        "BTC_IRT",
                        "sell",
                        101.0,
                        0.25,
                        tags=("global",),
                    ),
                ),
            ),
        ),
    )

    writer._write_batch([envelope])

    rows_by_role = {row[6]: row for row in _target_order_fact_rows(fake)}
    local = rows_by_role["local"]
    global_ = rows_by_role["global"]
    assert local[4] == "mpc-controller"
    assert local[5] == "liquidity_sourcing"
    assert local[7] == "nobitex"
    assert local[8] == "BTC_IRT"
    assert local[11] == "reconcile"
    assert local[12] == 0
    assert local[13] == "buy"
    assert local[14] == 99.0
    assert local[15] == 0.5
    assert local[16] == 49.5
    assert local[17] == "limit"
    assert local[18].value == ["local", "risk_approved"]
    assert local[19] == "frame-1"
    assert local[20] == "decision-1"
    assert local[21] == "target-1"
    assert local[22] == "plan-local"
    assert global_[7] == "wallex"
    assert global_[11] == "replace"
    assert global_[13] == "sell"
    assert global_[16] == 25.25
    assert global_[22] == "plan-global"


def test_postgres_writer_materializes_execution_order_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
    )
    envelope = envelope_event(
        "execution:orders:nobitex",
        {
            "type": "PlaceOrdersAccepted",
            "exchange": "nobitex",
            "order_ids": ["ex-1"],
            "orders": [
                {
                    "exchange": "nobitex",
                    "symbol": "BTC_IRT",
                    "side": "buy",
                    "price": 100.0,
                    "qty": 0.5,
                    "order_type": "limit",
                    "client_order_id": "client-1",
                    "exchange_order_id": "ex-1",
                },
            ],
            "timestamp_ms": 2_000,
            "frame_id": "frame-1",
            "decision_id": "decision-1",
            "target_book_id": "target-1",
            "execution_plan_id": "plan-local",
        },
    )

    writer._write_batch([envelope])

    row = _single_execution_fact_row(fake)
    assert row[4] == "PlaceOrdersAccepted"
    assert row[5] == "place"
    assert row[6] == "accepted"
    assert row[7] is True
    assert row[8] == "pending"
    assert row[10] == "nobitex"
    assert row[11] == "BTC_IRT"
    assert row[14] == "client-1"
    assert row[15] == "ex-1"
    assert row[16] == "buy"
    assert row[17] == 100.0
    assert row[18] == 0.5
    assert row[22] == "limit"
    assert row[28] == "frame-1"
    assert row[29] == "decision-1"
    assert row[30] == "target-1"
    assert row[31] == "plan-local"


def test_postgres_writer_materializes_fill_execution_fact(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
    )
    envelope = envelope_event(
        "private:state:nobitex:fills",
        Fill(
            exchange_order_id="ex-1",
            actor_id="actor-1",
            exchange="nobitex",
            symbol="BTC_IRT",
            side="buy",
            price=100.0,
            qty=0.25,
            fee=0.01,
            timestamp_ms=2_500,
        ),
    )

    writer._write_batch([envelope])

    row = _single_execution_fact_row(fake)
    assert row[4] == "Fill"
    assert row[5] == "fill"
    assert row[6] == "filled"
    assert row[7] is True
    assert row[8] == "confirmed"
    assert row[9] == "actor-1"
    assert row[15] == "ex-1"
    assert row[16] == "buy"
    assert row[17] == 100.0
    assert row[18] == 0.25
    assert row[20] == 0.25
    assert row[21] == 0.01
    assert row[23] == "filled"


def test_postgres_writer_materializes_execution_snapshot_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_psycopg2(monkeypatch)
    writer = PostgresEventWriter(
        bus=InMemoryBus(),
        topics=(),
        database_url="postgresql://example/db",
    )
    order = Order("nobitex", "BTC_IRT", "sell", 101.0, 0.4, client_order_id="client-sell")
    envelope = envelope_event(
        "execution:state:nobitex",
        ExecutionSnapshot(
            exchange="nobitex",
            timestamp_ms=3_000,
            live_order_count=1,
            live_orders={
                "live-1": OpenOrder(
                    exchange_order_id="live-1",
                    actor_id="actor-1",
                    order=order,
                    remaining_qty=0.3,
                    filled_qty=0.1,
                    status="open",
                    timestamp_ms=3_000,
                )
            },
            pending_place_count=1,
            pending_cancel_count=0,
            pending_places={
                "pending-1": PendingOrderChange(
                    operation="place",
                    exchange_order_id="pending-1",
                    actor_id="actor-1",
                    accepted_ms=3_000,
                    order=order,
                )
            },
        ),
    )

    writer._write_batch([envelope])

    rows = _execution_fact_rows(fake)
    by_key = {row[1]: row for row in rows}
    assert by_key["snapshot"][5] == "snapshot"
    assert by_key["snapshot"][8] == "aggregate"
    assert by_key["snapshot"][27] == 1
    assert by_key["live:live-1"][8] == "confirmed"
    assert by_key["live:live-1"][15] == "live-1"
    assert by_key["live:live-1"][19] == 0.3
    assert by_key["live:live-1"][20] == 0.1
    assert by_key["live:live-1"][23] == "open"
    assert by_key["pending_place:pending-1"][5] == "place"
    assert by_key["pending_place:pending-1"][8] == "pending_place"


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


def _execution_fact_rows(fake: "_FakePsycopg2") -> list[tuple]:
    conn = fake.connections[0]
    call = next(
        call
        for call in conn.execute_values_calls
        if "INSERT INTO mm_execution_facts" in call["sql"]
    )
    return call["rows"]


def _target_order_fact_rows(fake: "_FakePsycopg2") -> list[tuple]:
    conn = fake.connections[0]
    call = next(
        call
        for call in conn.execute_values_calls
        if "INSERT INTO mm_target_order_facts" in call["sql"]
    )
    return call["rows"]


def _model_score_fact_rows(fake: "_FakePsycopg2") -> list[tuple]:
    conn = fake.connections[0]
    call = next(
        call
        for call in conn.execute_values_calls
        if "INSERT INTO mm_model_score_facts" in call["sql"]
    )
    return call["rows"]


def _optimizer_fact_rows(fake: "_FakePsycopg2") -> list[tuple]:
    conn = fake.connections[0]
    call = next(
        call
        for call in conn.execute_values_calls
        if "INSERT INTO mm_optimizer_facts" in call["sql"]
    )
    return call["rows"]


def _optimizer_component_fact_rows(fake: "_FakePsycopg2") -> list[tuple]:
    conn = fake.connections[0]
    call = next(
        call
        for call in conn.execute_values_calls
        if "INSERT INTO mm_optimizer_component_facts" in call["sql"]
    )
    return call["rows"]


def _single_execution_fact_row(fake: "_FakePsycopg2") -> tuple:
    [row] = _execution_fact_rows(fake)
    return row


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
