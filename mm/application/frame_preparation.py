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
    """Return a frame whose local/global books exclude the bot's live orders."""

    clean_local = _replace_event_orderbook(
        frame.local,
        subtract_own_orders(frame.local.orderbook, open_orders),
    )
    clean_global = None
    if frame.global_ is not None:
        clean_global = _replace_event_orderbook(
            frame.global_,
            subtract_own_orders(frame.global_.orderbook, open_orders),
        )
    return StrategyInputFrame(
        local=clean_local,
        global_=clean_global,
        local_private=frame.local_private,
        global_private=frame.global_private,
        global_status=frame.global_status,
        global_lag_ms=frame.global_lag_ms,
        max_global_lag_ms=frame.max_global_lag_ms,
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
