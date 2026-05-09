from __future__ import annotations

import pytest

from mm.runtime.config import MarketDataSourceConfig, MarketMakerRuntimeConfig, PrivateRuntimeConfig
from mm.runtime.engine import build_runtime_components
from mm.runtime.print_events import build_print_queues


def test_runtime_config_uses_hedge_as_only_secondary_source() -> None:
    config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
        hedge=MarketDataSourceConfig(role="hedge", exchange="wallex", symbol="BTC_IRT"),
        interval_ms=1000,
    )

    contract = config.as_contract()

    assert contract["strategy_input_frame"]["hedge_is_anchor"] is True
    assert contract["hedge"]["role"] == "hedge"
    assert config.frame_topic == "strategy:frames:nobitex:BTC_IRT:1000"
    assert config.binding.comparable is True


def test_runtime_config_rejects_anchor_like_secondary_roles() -> None:
    with pytest.raises(ValueError, match="hedge source role"):
        MarketMakerRuntimeConfig(
            primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT"),
            hedge=MarketDataSourceConfig(role="primary", exchange="wallex", symbol="BTC_IRT"),
        )


@pytest.mark.asyncio
async def test_runtime_builder_wires_primary_and_hedge_topics_without_starting_network() -> None:
    config = MarketMakerRuntimeConfig(
        primary=MarketDataSourceConfig(role="primary", exchange="nobitex", symbol="BTC_IRT", mode="rest"),
        hedge=MarketDataSourceConfig(role="hedge", exchange="wallex", symbol="BTC_IRT", mode="websocket"),
        interval_ms=1000,
        primary_private=PrivateRuntimeConfig(api_token=None),
    )

    runtime = await build_runtime_components(config)
    queues = build_print_queues(runtime, include_raw=True)
    labels = {label for label, _ in queues}

    assert "strategy:frames:nobitex:BTC_IRT:1000" in labels
    assert "market:events:nobitex:BTC_IRT:1000" in labels
    assert "market:events:wallex:BTC_IRT:1000" in labels
    assert "market:raw:nobitex:BTC_IRT:trades" in labels
    assert "market:raw:wallex:BTC_IRT:orderbook" in labels
    assert runtime.public_market_selections["primary"].transport == "rest"
    assert runtime.public_market_selections["hedge"].transport == "websocket"
