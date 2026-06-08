from __future__ import annotations

import pytest

from mm.core.events import envelope_event
from mm.core.models.market import MarketEvent, PriceLevel, RawOrderbook, RawTrade
from mm.core.models.private import Balance, PrivateStateSnapshot
from mm.persistence.parquet_writer import ParquetEventWriter
from mm.persistence.replay import read_envelopes


def test_parquet_writer_appends_under_hive_partition_path(tmp_path) -> None:
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

    path = tmp_path / "event_date=1970-01-01" / "exchange=nobitex" / "symbol=BTC_IRT" / "00-MarketEvent.parquet"
    table = pq.ParquetFile(path).read()
    assert table.num_rows == 2
    assert table.schema.field("exchange").type == pa.string()
    assert "payload_json" not in table.schema.names
    assert "envelope_json" in table.schema.names


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
