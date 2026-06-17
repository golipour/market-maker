from __future__ import annotations

from typing import Any

from mm.exchanges.kucoin.market_data import KucoinMarketDataAdapter
from mm.exchanges.kucoin.order_adapter import KucoinRestOrderCommandAdapter
from mm.exchanges.kucoin.private_state import (
    KucoinRestPrivateStateAdapter,
    KucoinWebSocketPrivateStateAdapter,
)
from mm.exchanges.registry import ExchangeRegistration
from mm.exchanges.shared.provider_adapter import TransportMode


def build_market_data(
    symbol: str,
    *,
    transport: TransportMode,
    depth_limit: int,
    trades_limit: int,
    poll_interval_ms: int,
):
    return KucoinMarketDataAdapter(
        symbol,
        mode=transport,
        depth_limit=depth_limit,
        trades_limit=trades_limit,
        poll_interval_ms=poll_interval_ms,
    )


def build_private_state(
    symbol: str,
    *,
    selected_mode: str,
    api_token: Any,
    poll_interval_ms: int,
):
    if selected_mode == "websocket":
        return KucoinWebSocketPrivateStateAdapter(
            symbol,
            api_token=api_token,
            poll_interval_ms=poll_interval_ms,
        )
    return KucoinRestPrivateStateAdapter(
        symbol,
        api_token=api_token,
        poll_interval_ms=poll_interval_ms,
    )


def build_order_command(
    symbol: str,
    *,
    api_token: Any,
    max_concurrency: int,
):
    return KucoinRestOrderCommandAdapter(
        symbol,
        api_token=api_token,
        max_concurrency=max_concurrency,
    )


REGISTRATION = ExchangeRegistration(
    name="kucoin",
    build_market_data=build_market_data,
    build_private_state=build_private_state,
    build_order_command=build_order_command,
    uses_trade_stats_limits=True,
)
