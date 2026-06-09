from __future__ import annotations

from typing import Any

from mm.exchanges.registry import ExchangeRegistration
from mm.exchanges.shared.provider_adapter import TransportMode
from mm.exchanges.tabdeal import (
    TabdealMarketDataAdapter,
    TabdealRestPrivateStateAdapter,
    TabdealWebSocketPrivateStateAdapter,
)
from mm.exchanges.tabdeal.order_adapter import TabdealRestOrderCommandAdapter


def build_market_data(
    symbol: str,
    *,
    transport: TransportMode,
    depth_limit: int,
    trades_limit: int,
    poll_interval_ms: int,
):
    return TabdealMarketDataAdapter(
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
        return TabdealWebSocketPrivateStateAdapter(
            symbol,
            api_token=api_token,
            poll_interval_ms=poll_interval_ms,
        )
    return TabdealRestPrivateStateAdapter(
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
    return TabdealRestOrderCommandAdapter(
        symbol,
        api_token=api_token,
        max_concurrency=max_concurrency,
    )


REGISTRATION = ExchangeRegistration(
    name="tabdeal",
    build_market_data=build_market_data,
    build_private_state=build_private_state,
    build_order_command=build_order_command,
    uses_trade_stats_limits=True,
)
