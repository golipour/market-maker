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
    main_buy_ref: float
    main_sell_ref: float
    hedge_buy_ref: float | None
    hedge_sell_ref: float | None
    hedge_inventory: int
    hedge_cash: int | float
