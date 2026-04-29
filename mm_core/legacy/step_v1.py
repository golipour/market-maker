from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StepV1:
    """
    Typed view over the current legacy `Step` tuple.

    Current tuple shape (15 fields):
      (t_end_ms, ob, buy_lots, sell_lots, inventory, cash,
       private_trade_fee_tick_lot, private_fills_bid, private_fills_ask,
       main_buy_ref_ticks, main_sell_ref_ticks,
       anchor_buy_ref_ticks, anchor_sell_ref_ticks,
       anchor_inventory, anchor_cash)
    """

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
    anchor_buy_ref: float | None
    anchor_sell_ref: float | None
    anchor_inventory: int
    anchor_cash: int | float


def parse_step(step: tuple) -> StepV1:
    """
    Parse legacy step tuple into `StepV1`.

    This function is the single place that "knows" the tuple layout.
    """
    (
        t_end_ms,
        ob,
        buy_lots,
        sell_lots,
        inventory,
        cash,
        private_trade_fee_tick_lot,
        private_fills_bid,
        private_fills_ask,
        main_buy_ref,
        main_sell_ref,
        anchor_buy_ref,
        anchor_sell_ref,
        anchor_inventory,
        anchor_cash,
    ) = step
    return StepV1(
        t_end_ms=int(t_end_ms),
        ob=ob,
        buy_lots=int(buy_lots),
        sell_lots=int(sell_lots),
        inventory=int(inventory),
        cash=cash,
        private_trade_fee_tick_lot=float(private_trade_fee_tick_lot),
        private_fills_bid=tuple(private_fills_bid),
        private_fills_ask=tuple(private_fills_ask),
        main_buy_ref=float(main_buy_ref),
        main_sell_ref=float(main_sell_ref),
        anchor_buy_ref=float(anchor_buy_ref) if anchor_buy_ref is not None else None,
        anchor_sell_ref=float(anchor_sell_ref) if anchor_sell_ref is not None else None,
        anchor_inventory=int(anchor_inventory),
        anchor_cash=anchor_cash,
    )

