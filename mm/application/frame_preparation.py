"""Prepare strategy input frames for decision-making (own-order adjustment)."""

from __future__ import annotations

from collections.abc import Mapping

from mm.application.orderbook_view import subtract_own_orders
from mm.core.models.execution import Order
from mm.core.models.market import MarketEvent
from mm.core.models.strategy import StrategyInputFrame


def clean_strategy_frame(
    frame: StrategyInputFrame,
    open_orders: Mapping[str, Order],
) -> StrategyInputFrame:
    """Return a frame whose primary/hedge books exclude the bot's live orders."""

    clean_primary = _replace_event_orderbook(
        frame.primary,
        subtract_own_orders(frame.primary.orderbook, open_orders),
    )
    clean_hedge = None
    if frame.hedge is not None:
        clean_hedge = _replace_event_orderbook(
            frame.hedge,
            subtract_own_orders(frame.hedge.orderbook, open_orders),
        )
    return StrategyInputFrame(
        primary=clean_primary,
        hedge=clean_hedge,
        primary_private=frame.primary_private,
        hedge_private=frame.hedge_private,
        hedge_status=frame.hedge_status,
        hedge_lag_ms=frame.hedge_lag_ms,
        max_hedge_lag_ms=frame.max_hedge_lag_ms,
        created_at_ms=frame.created_at_ms,
    )


def _replace_event_orderbook(event: MarketEvent, orderbook) -> MarketEvent:
    return MarketEvent(
        exchange=event.exchange,
        symbol=event.symbol,
        interval_ms=event.interval_ms,
        t_start_ms=event.t_start_ms,
        t_end_ms=event.t_end_ms,
        orderbook=orderbook,
        trades=event.trades,
        buy_volume=event.buy_volume,
        sell_volume=event.sell_volume,
        vwap=event.vwap,
        imbalance=event.imbalance,
        microprice=event.microprice,
    )
