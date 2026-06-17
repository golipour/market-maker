from __future__ import annotations

from mm.registry.market_data import market_data_settings_from_mapping


def test_market_data_ignores_market_data_source_flag() -> None:
    settings = market_data_settings_from_mapping(
        {"market_data": {"market_data_source": "bus", "poll_interval_ms": 2500}}
    )
    assert settings["poll_interval_ms"] == 2500
