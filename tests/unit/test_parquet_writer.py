from __future__ import annotations

import pytest

from mm.core.events import envelope_event
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook, RawTrade
from mm.core.models.private import Balance, PrivateStateSnapshot
from mm.persistence.parquet_writer import ParquetEventWriter
from mm.persistence.replay import read_envelopes


def test_parquet_writer_writes_immutable_parts_under_hive_partition_path(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    writer = ParquetEventWriter(
        bus=None,  # type: ignore[arg-type]
        topics=(),
        root_path=str(tmp_path),
        run_id="test-run",
    )
    first = envelope_event(
        "market:events:nobitex:BTC_IRT:1000",
        {
            "type": "MarketEvent",
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "t_end_ms": 1_000,
        },
    )
    second = envelope_event(
        "market:events:nobitex:BTC_IRT:1000",
        {
            "type": "MarketEvent",
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "t_end_ms": 2_000,
        },
    )

    writer._write_batch([first])
    writer._write_batch([second])

    partition = (
        tmp_path
        / "event_date=1970-01-01"
        / "exchange=nobitex"
        / "symbol=BTC_IRT"
        / "event_type=MarketEvent"
        / "hour=00"
    )
    files = sorted(partition.glob("part-*.parquet"))
    assert len(files) == 2
    row_count = 0
    for path in files:
        table = pq.ParquetFile(path).read()
        row_count += table.num_rows
        assert table.schema.field("exchange").type == pa.string()
        assert table.schema.field("best_bid").type == pa.float64()
        assert table.schema.field("execution_lifecycle").type == pa.string()
        assert "payload_json" not in table.schema.names
        assert "envelope_json" in table.schema.names
    assert row_count == 2
    assert [envelope.timestamp_ms for envelope in read_envelopes(tmp_path)] == [1_000, 2_000]


def test_parquet_writer_compacts_archival_market_trades(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")

    writer = ParquetEventWriter(
        bus=None,  # type: ignore[arg-type]
        topics=(),
        root_path=str(tmp_path),
        run_id="test-run",
    )
    event = MarketEvent(
        exchange="nobitex",
        symbol="BTC_IRT",
        interval_ms=1000,
        t_start_ms=1_000,
        t_end_ms=2_000,
        orderbook=RawOrderbook(
            exchange="nobitex",
            symbol="BTC_IRT",
            timestamp_ms=2_000,
            bids=(PriceLevel(100.0, 1.0),),
            asks=(PriceLevel(101.0, 2.0),),
            is_snapshot=True,
        ),
        trades=(
            RawTrade("t1", 1_100, 100.0, 0.1, "buy", "nobitex", "BTC_IRT"),
            RawTrade("t2", 1_200, 101.0, 0.2, "sell", "nobitex", "BTC_IRT"),
        ),
        buy_volume=0.1,
        sell_volume=0.2,
        vwap=100.66,
        imbalance=-0.1,
        microprice=100.5,
    )

    writer._write_batch([envelope_event("market:events:nobitex:BTC_IRT:1000", event)])

    [row] = _parquet_rows(tmp_path)
    assert row["best_bid"] == 100.0
    assert row["best_bid_qty"] == 1.0
    assert row["best_ask"] == 101.0
    assert row["best_ask_qty"] == 2.0
    assert row["mid"] == 100.5
    assert row["spread"] == 1.0
    assert row["microprice"] == 100.5
    assert row["vwap"] == pytest.approx(100.66)
    assert row["total_trade_volume"] == pytest.approx(0.3)
    assert row["trade_count"] == 2
    assert row["bid_depth_levels"] == 1
    assert row["ask_depth_levels"] == 1
    assert row["orderbook_is_snapshot"] is True

    [envelope] = list(read_envelopes(tmp_path))
    assert envelope.event_type == "MarketEvent"
    assert envelope.payload["trade_count"] == 2
    assert envelope.payload["first_trade_timestamp_ms"] == 1_100
    assert envelope.payload["last_trade_timestamp_ms"] == 1_200
    assert envelope.payload["trades"] == []
    assert envelope.payload["trades_compacted"] is True
    assert envelope.payload["buy_volume"] == pytest.approx(0.1)
    assert envelope.payload["vwap"] == pytest.approx(100.66)


def test_parquet_writer_compacts_zero_balance_snapshot_rows(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")

    writer = ParquetEventWriter(
        bus=None,  # type: ignore[arg-type]
        topics=(),
        root_path=str(tmp_path),
        run_id="test-run",
    )
    snapshot = PrivateStateSnapshot(
        exchange="nobitex",
        timestamp_ms=2_000,
        balances={
            "BTC": Balance("nobitex", "BTC", 1.0, 0.0, 1.0, 2_000),
            "ZERO": Balance("nobitex", "ZERO", 0.0, 0.0, 0.0, 2_000),
        },
    )

    writer._write_batch([envelope_event("private:state:nobitex:snapshot", snapshot)])

    [envelope] = list(read_envelopes(tmp_path))
    assert envelope.event_type == "PrivateStateSnapshot"
    assert envelope.payload["balance_count"] == 2
    assert envelope.payload["nonzero_balance_count"] == 1
    assert envelope.payload["balances_compacted"] is True
    assert sorted(envelope.payload["balances"]) == ["BTC"]


def test_parquet_writer_materializes_typed_strategy_and_execution_columns(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")

    writer = ParquetEventWriter(
        bus=None,  # type: ignore[arg-type]
        topics=(),
        root_path=str(tmp_path),
        run_id="test-run",
    )
    diagnostics = envelope_event(
        "strategy:diagnostics:mpc-controller",
        {
            "type": "ControllerDiagnostics",
            "actor_id": "mpc-controller",
            "exchange": "nobitex",
            "symbol": "BTC_IRT",
            "created_at_ms": 2_100,
            "t_end_ms": 2_000,
            "frame_step": 12,
            "state": {
                "local_inventory": 0.5,
                "global_inventory": 1.5,
                "local_cash": 100.0,
                "global_cash": 200.0,
                "local_wealth": 150.0,
                "global_wealth": 250.0,
            },
            "models": {
                "price": {"updated": True, "nll": 1.25},
                "flow": {"updated": False, "skip_reason": "missing_history"},
            },
            "optimizer": {
                "config": {"kind": "monte_carlo"},
                "timing": {"controller_total_ms": 12.0, "optimize_ms": 7.5},
                "objective": {
                    "initial": {"value": 1.0},
                    "final": {"value": 3.0},
                    "best": {
                        "value": 3.5,
                        "expected_total_reward": 10.0,
                        "expected_total_return": 0.03,
                    },
                    "improvement": {
                        "best_minus_initial": 2.5,
                        "final_minus_initial": 2.0,
                    },
                },
            },
            "risk": {"approved": True},
            "warnings": ["wide_spread"],
        },
    )
    target_book = envelope_event(
        "strategy:target_books:mpc-controller",
        {
            "type": "JointTargetBook",
            "actor_id": "mpc-controller",
            "mode": "liquidity_sourcing",
            "local": {
                "exchange": "nobitex",
                "symbol": "BTC_IRT",
                "orders": [
                    {"side": "buy", "price": 100.0, "qty": 0.5},
                    {"side": "sell", "price": 110.0, "qty": 0.25},
                ],
            },
            "global": {
                "exchange": "wallex",
                "symbol": "BTC_IRT",
                "orders": [{"side": "sell", "price": 120.0, "qty": 0.1}],
            },
        },
    )
    execution = envelope_event(
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
                    "client_order_id": "client-1",
                }
            ],
            "timestamp_ms": 3_000,
        },
    )

    writer._write_batch([diagnostics, target_book, execution])

    rows_by_type = {row["event_type"]: row for row in _parquet_rows(tmp_path)}
    diag = rows_by_type["ControllerDiagnostics"]
    assert diag["frame_step"] == 12
    assert diag["local_inventory"] == 0.5
    assert diag["model_count"] == 2
    assert diag["model_updated_count"] == 1
    assert diag["model_skipped_count"] == 1
    assert diag["optimizer_kind"] == "monte_carlo"
    assert diag["controller_total_ms"] == 12.0
    assert diag["optimize_ms"] == 7.5
    assert diag["best_value"] == 3.5
    assert diag["best_minus_initial"] == 2.5
    assert diag["best_expected_total_reward"] == 10.0
    assert diag["risk_approved"] is True
    assert diag["warnings_count"] == 1

    target = rows_by_type["JointTargetBook"]
    assert target["mode"] == "liquidity_sourcing"
    assert target["local_target_order_count"] == 2
    assert target["global_target_order_count"] == 1
    assert target["target_order_count"] == 3
    assert target["local_target_notional"] == 77.5
    assert target["global_target_notional"] == 12.0

    order = rows_by_type["PlaceOrdersAccepted"]
    assert order["execution_action"] == "place"
    assert order["execution_lifecycle"] == "accepted"
    assert order["execution_success"] is True
    assert order["execution_state_kind"] == "pending"
    assert order["client_order_id"] == "client-1"
    assert order["exchange_order_id"] == "ex-1"
    assert order["order_side"] == "buy"
    assert order["order_price"] == 100.0
    assert order["order_qty"] == 0.5
    assert order["count_value"] == 1


def _parquet_rows(root) -> list[dict]:
    pq = pytest.importorskip("pyarrow.parquet")
    rows: list[dict] = []
    for path in sorted(root.rglob("part-*.parquet")):
        rows.extend(pq.ParquetFile(path).read().to_pylist())
    return rows
