"""Step tuple for torch market-flow / price estimators (no trade_stats Orderbook)."""

from __future__ import annotations

from typing import Any, NamedTuple


class Step(NamedTuple):
    """One advance of the order-flow stepper; tuple-compatible with legacy layout."""

    t_end_ms: int
    ob: Any
    buy_lots: int
    sell_lots: int
    inventory: int
    cash: int | float
    private_trade_fee_tick_lot: float
    private_fills_bid: tuple[tuple[int, int], ...]
    private_fills_ask: tuple[tuple[int, int], ...]
    main_buy_vwap: float
    main_sell_vwap: float
    global_buy_vwap: float | None
    global_sell_vwap: float | None
    global_inventory: int
    hedge_cash: int | float
