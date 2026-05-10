"""
MarketDataMux: compose REST vs websocket vs buffered polling **market data** providers.

Centralizes the branching that used to live inline in ``OrderFlowStepper.__init__`` so
runtime wiring has one place for main + hedge order books and main trades (Phase 4).
"""

from __future__ import annotations

from dataclasses import dataclass

from mm.layers.core.ports.market_data import OrderbookDataProvider, TradeDataProvider
from mm.layers.runtime.market_data_providers import (
    BufferedPollingOrderbookDataProvider,
    BufferedPollingTradeDataProvider,
    RestOrderbookDataProvider,
    WebSocketBufferedMarketDataProvider,
)


@dataclass(frozen=True, slots=True)
class MarketDataBundle:
    """Concrete providers selected for one stepper (main trade + main book + optional hedge book)."""

    trade_provider: TradeDataProvider
    orderbook_provider: OrderbookDataProvider
    hedge_orderbook_provider: OrderbookDataProvider | None


class MarketDataMux:
    """Factory for ``MarketDataBundle`` from the same flags ``OrderFlowStepper`` accepts."""

    @staticmethod
    def build_providers(
        *,
        exchange: str,
        symbol: str,
        trades_limit: int,
        depth_limit: int,
        trade_provider: TradeDataProvider | None,
        orderbook_provider: OrderbookDataProvider | None,
        use_buffered_orderbook_provider: bool,
        use_websocket_market_data: bool,
        trades_poll_interval_ms: int,
        trade_buffer_ms: int,
        orderbook_poll_interval_ms: int,
        orderbook_buffer_ms: int,
        websocket_reconnect_ms: int,
        private_api_token: str | None,
        hedge_exchange: str | None,
        hedge_symbol: str | None,
        use_hedge_websocket_market_data: bool,
        hedge_depth_limit: int | None,
        hedge_orderbook_poll_interval_ms: int | None,
        hedge_orderbook_buffer_ms: int | None,
        hedge_websocket_reconnect_ms: int | None,
    ) -> MarketDataBundle:
        if use_websocket_market_data and trade_provider is None and orderbook_provider is None:
            ws = WebSocketBufferedMarketDataProvider(
                exchange=exchange,
                symbol=symbol,
                trades_limit=trades_limit,
                depth_limit=depth_limit,
                max_buffer_ms=max(trade_buffer_ms, orderbook_buffer_ms),
                reconnect_ms=websocket_reconnect_ms,
                private_api_token=private_api_token,
            )
            orderbook_out: OrderbookDataProvider = ws
            trade_out: TradeDataProvider = ws
        else:
            if trade_provider is not None:
                trade_out = trade_provider
            else:
                trade_out = BufferedPollingTradeDataProvider(
                    exchange=exchange,
                    symbol=symbol,
                    trades_limit=trades_limit,
                    poll_interval_ms=trades_poll_interval_ms,
                    max_buffer_ms=trade_buffer_ms,
                )
            if orderbook_provider is not None:
                orderbook_out = orderbook_provider
            elif use_buffered_orderbook_provider:
                orderbook_out = BufferedPollingOrderbookDataProvider(
                    exchange=exchange,
                    symbol=symbol,
                    depth_limit=depth_limit,
                    poll_interval_ms=orderbook_poll_interval_ms,
                    max_buffer_ms=orderbook_buffer_ms,
                )
            else:
                orderbook_out = RestOrderbookDataProvider(
                    exchange=exchange,
                    symbol=symbol,
                    depth_limit=depth_limit,
                )

        hedge_ex = str(hedge_exchange).strip() if hedge_exchange else ""
        hedge_sym = str(hedge_symbol).strip() if hedge_symbol else str(symbol).strip()
        hedge_out: OrderbookDataProvider | None = None
        if hedge_ex:
            _hedge_depth_limit = (
                int(hedge_depth_limit)
                if hedge_depth_limit is not None
                else int(depth_limit)
            )
            _hedge_orderbook_poll_interval_ms = (
                int(hedge_orderbook_poll_interval_ms)
                if hedge_orderbook_poll_interval_ms is not None
                else int(orderbook_poll_interval_ms)
            )
            _hedge_orderbook_buffer_ms = (
                int(hedge_orderbook_buffer_ms)
                if hedge_orderbook_buffer_ms is not None
                else int(orderbook_buffer_ms)
            )
            _hedge_websocket_reconnect_ms = (
                int(hedge_websocket_reconnect_ms)
                if hedge_websocket_reconnect_ms is not None
                else int(websocket_reconnect_ms)
            )
            if use_hedge_websocket_market_data:
                hedge_out = WebSocketBufferedMarketDataProvider(
                    exchange=hedge_ex,
                    symbol=hedge_sym,
                    trades_limit=1,
                    depth_limit=_hedge_depth_limit,
                    max_buffer_ms=_hedge_orderbook_buffer_ms,
                    reconnect_ms=_hedge_websocket_reconnect_ms,
                    private_api_token=None,
                )
            else:
                hedge_out = BufferedPollingOrderbookDataProvider(
                    exchange=hedge_ex,
                    symbol=hedge_sym,
                    depth_limit=_hedge_depth_limit,
                    poll_interval_ms=_hedge_orderbook_poll_interval_ms,
                    max_buffer_ms=_hedge_orderbook_buffer_ms,
                )

        return MarketDataBundle(
            trade_provider=trade_out,
            orderbook_provider=orderbook_out,
            hedge_orderbook_provider=hedge_out,
        )
