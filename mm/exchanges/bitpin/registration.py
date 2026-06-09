from __future__ import annotations

from typing import Any

from mm.exchanges.bitpin import (
    BitpinMarketDataAdapter,
    BitpinRestPrivateStateAdapter,
    BitpinWebSocketPrivateStateAdapter,
)
from mm.exchanges.bitpin.order_adapter import BitpinRestOrderCommandAdapter
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
    return BitpinMarketDataAdapter(
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
        return BitpinWebSocketPrivateStateAdapter(
            symbol,
            api_token=api_token,
            poll_interval_ms=poll_interval_ms,
        )
    return BitpinRestPrivateStateAdapter(
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
    return BitpinRestOrderCommandAdapter(
        symbol,
        api_token=api_token,
        max_concurrency=max_concurrency,
    )


REGISTRATION = ExchangeRegistration(
    name="bitpin",
    build_market_data=build_market_data,
    build_private_state=build_private_state,
    build_order_command=build_order_command,
    uses_trade_stats_limits=True,
)
