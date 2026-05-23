from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from mm.core.models.execution import Order
from mm.core.models.market import msgspec


if msgspec is not None:

    class Balance(msgspec.Struct, frozen=True):
        exchange: str
        asset: str
        free: float
        locked: float
        total: float
        timestamp_ms: int


    class OpenOrder(msgspec.Struct, frozen=True):
        exchange_order_id: str
        actor_id: Optional[str]
        order: Order
        remaining_qty: float
        filled_qty: float
        status: str
        timestamp_ms: int


    class OrderUpdate(msgspec.Struct, frozen=True):
        exchange: str
        symbol: str
        exchange_order_id: str
        actor_id: Optional[str]
        status: str
        remaining_qty: float
        filled_qty: float
        timestamp_ms: int

else:

    @dataclass(frozen=True, slots=True)
    class Balance:
        exchange: str
        asset: str
        free: float
        locked: float
        total: float
        timestamp_ms: int


    @dataclass(frozen=True, slots=True)
    class OpenOrder:
        exchange_order_id: str
        actor_id: Optional[str]
        order: Order
        remaining_qty: float
        filled_qty: float
        status: str
        timestamp_ms: int


    @dataclass(frozen=True, slots=True)
    class OrderUpdate:
        exchange: str
        symbol: str
        exchange_order_id: str
        actor_id: Optional[str]
        status: str
        remaining_qty: float
        filled_qty: float
        timestamp_ms: int


@dataclass(frozen=True, slots=True)
class PrivateStateSnapshot:
    exchange: str
    timestamp_ms: int
    balances: dict[str, Balance] = field(default_factory=dict)
    open_orders: dict[str, OpenOrder] = field(default_factory=dict)


PrivateEvent = Balance | OpenOrder | OrderUpdate
