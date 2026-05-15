from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from mm.core.models.market import msgspec


if msgspec is not None:

    class Order(msgspec.Struct, frozen=True):
        exchange: str
        symbol: str
        side: str
        price: float
        qty: float
        order_type: str = "limit"
        client_order_id: Optional[str] = None


    class Action(msgspec.Struct, frozen=True):
        actor_id: str
        orders_to_create: Tuple[Order, ...] = ()
        orders_to_cancel: Tuple[str, ...] = ()


    class Fill(msgspec.Struct, frozen=True):
        exchange_order_id: str
        actor_id: str
        exchange: str
        symbol: str
        side: str
        price: float
        qty: float
        fee: float
        timestamp_ms: int

else:

    @dataclass(frozen=True, slots=True)
    class Order:
        exchange: str
        symbol: str
        side: str
        price: float
        qty: float
        order_type: str = "limit"
        client_order_id: Optional[str] = None


    @dataclass(frozen=True, slots=True)
    class Action:
        actor_id: str
        orders_to_create: Tuple[Order, ...] = ()
        orders_to_cancel: Tuple[str, ...] = ()


    @dataclass(frozen=True, slots=True)
    class Fill:
        exchange_order_id: str
        actor_id: str
        exchange: str
        symbol: str
        side: str
        price: float
        qty: float
        fee: float
        timestamp_ms: int


@dataclass(slots=True)
class TradingState:
    """Wallet and live-order snapshot passed to the risk engine each frame."""

    actor_id: str
    cash: float = 0.0
    inventory: float = 0.0
    open_orders: dict[str, Order] = field(default_factory=dict)
    model_data: dict[str, object] = field(default_factory=dict)

