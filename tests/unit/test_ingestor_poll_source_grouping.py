from __future__ import annotations

import pytest

from mm.runtime.ingestor_config import IngestorFeedConfig, group_feeds_by_poll_source


def test_group_feeds_by_poll_source_merges_intervals() -> None:
    feeds = (
        IngestorFeedConfig("nobitex", "USDT_IRT", 10_000, poll_interval_ms=2000),
        IngestorFeedConfig("nobitex", "USDT_IRT", 20_000, poll_interval_ms=2000),
        IngestorFeedConfig("tabdeal", "USDT_IRT", 10_000, poll_interval_ms=1500),
    )
    grouped = group_feeds_by_poll_source(feeds)
    assert set(grouped) == {"nobitex:USDT_IRT", "tabdeal:USDT_IRT"}
    assert {row.interval_ms for row in grouped["nobitex:USDT_IRT"]} == {10_000, 20_000}


def test_group_feeds_by_poll_source_rejects_conflicting_poll_settings() -> None:
    feeds = (
        IngestorFeedConfig("nobitex", "USDT_IRT", 10_000, poll_interval_ms=2000),
        IngestorFeedConfig("nobitex", "USDT_IRT", 20_000, poll_interval_ms=3000),
    )
    with pytest.raises(ValueError, match="conflicting poll settings"):
        group_feeds_by_poll_source(feeds)


@pytest.mark.asyncio
async def test_build_ingestor_uses_one_engine_per_poll_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from mm.runtime.ingestor import build_ingestor_components
    from mm.runtime.ingestor_config import ingestor_config_from_mapping

    adapter_calls: list[tuple[str, str]] = []

    def _select(exchange: str, symbol: str, **kwargs: object) -> object:
        adapter_calls.append((exchange, symbol))
        return type(
            "Selection",
            (),
            {
                "adapter": type(
                    "Adapter",
                    (),
                    {
                        "connect": lambda self: None,
                        "disconnect": lambda self: None,
                    },
                )(),
                "transport": "fake",
            },
        )()

    monkeypatch.setattr("mm.runtime.ingestor.select_public_market_data_adapter", _select)
    config = ingestor_config_from_mapping(
        {
            "feeds": [
                {"exchange": "nobitex", "symbol": "USDT_IRT", "interval_ms": 10_000},
                {"exchange": "nobitex", "symbol": "USDT_IRT", "interval_ms": 20_000},
            ],
            "bus": {"kind": "redis_streams", "stream_prefix": "test"},
            "observability": {"metrics_enabled": False},
        }
    )
    ingestor = await build_ingestor_components(config, enable_observability=False)
    assert len(ingestor.engines) == 1
    assert "nobitex:USDT_IRT" in ingestor.engines
    assert adapter_calls == [("nobitex", "USDT_IRT")]
