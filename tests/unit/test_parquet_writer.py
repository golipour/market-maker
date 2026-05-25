from __future__ import annotations

import pytest

from mm.core.events import envelope_event
from mm.persistence.parquet_writer import ParquetEventWriter


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
