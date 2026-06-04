from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mm.exchanges.shared.provider_adapter import TransportMode

PrivateStateMode = Literal["rest", "websocket"]
RequestedTransportMode = Literal["rest", "websocket"]


@dataclass(frozen=True, slots=True)
class PublicMarketDataCapabilities:
    trades_rest: bool
    trades_websocket: bool
    orderbook_rest: bool
    orderbook_websocket: bool
    preferred_transport: TransportMode = "rest"

    @property
    def rest(self) -> bool:
        return self.trades_rest and self.orderbook_rest

    @property
    def websocket(self) -> bool:
        return self.trades_websocket and self.orderbook_websocket

    def supports(self, mode: TransportMode) -> bool:
        if mode == "rest":
            return self.rest
        if mode == "websocket":
            return self.websocket
        return False

    def support_details(self, mode: TransportMode) -> dict[str, bool]:
        if mode == "rest":
            return {
                "trades": self.trades_rest,
                "orderbook": self.orderbook_rest,
            }
        if mode == "websocket":
            return {
                "trades": self.trades_websocket,
                "orderbook": self.orderbook_websocket,
            }
        return {"trades": False, "orderbook": False}


@dataclass(frozen=True, slots=True)
class PrivateStateCapabilities:
    balances_rest: bool
    open_orders_rest: bool
    order_updates_websocket: bool
    fills_websocket: bool
    preferred_mode: PrivateStateMode = "rest"

    @property
    def rest(self) -> bool:
        return self.balances_rest and self.open_orders_rest

    @property
    def websocket(self) -> bool:
        return self.order_updates_websocket and self.fills_websocket

    def supports(self, mode: PrivateStateMode) -> bool:
        if mode == "rest":
            return self.rest
        if mode == "websocket":
            return self.websocket
        return False

    def support_details(self, mode: PrivateStateMode) -> dict[str, bool]:
        if mode == "rest":
            return {
                "balances": self.balances_rest,
                "open_orders": self.open_orders_rest,
            }
        if mode == "websocket":
            return {
                "order_updates": self.order_updates_websocket,
                "fills": self.fills_websocket,
            }
        return {"balances": False, "open_orders": False, "order_updates": False, "fills": False}


@dataclass(frozen=True, slots=True)
class OrderCommandCapabilities:
    place_rest: bool
    cancel_rest: bool
    amend_rest: bool = False
    market_order_rest: bool = False


@dataclass(frozen=True, slots=True)
class ExchangeCapabilities:
    exchange: str
    public_market_data: PublicMarketDataCapabilities
    private_state: PrivateStateCapabilities
    order_commands: OrderCommandCapabilities


EXCHANGE_CAPABILITIES: dict[str, ExchangeCapabilities] = {
    "nobitex": ExchangeCapabilities(
        exchange="nobitex",
        public_market_data=PublicMarketDataCapabilities(
            trades_rest=True,
            trades_websocket=True,
            orderbook_rest=True,
            orderbook_websocket=True,
            preferred_transport="rest",
        ),
        private_state=PrivateStateCapabilities(
            balances_rest=True,
            open_orders_rest=True,
            order_updates_websocket=False,
            fills_websocket=False,
            preferred_mode="rest",
        ),
        order_commands=OrderCommandCapabilities(
            place_rest=True, cancel_rest=True, market_order_rest=True
        ),
    ),
    "wallex": ExchangeCapabilities(
        exchange="wallex",
        public_market_data=PublicMarketDataCapabilities(
            trades_rest=True,
            trades_websocket=True,
            orderbook_rest=True,
            orderbook_websocket=True,
            preferred_transport="rest",
        ),
        private_state=PrivateStateCapabilities(
            balances_rest=True,
            open_orders_rest=True,
            order_updates_websocket=False,
            fills_websocket=False,
            preferred_mode="rest",
        ),
        order_commands=OrderCommandCapabilities(
            place_rest=True, cancel_rest=True, market_order_rest=True
        ),
    ),
    "tabdeal": ExchangeCapabilities(
        exchange="tabdeal",
        public_market_data=PublicMarketDataCapabilities(
            trades_rest=True,
            trades_websocket=True,
            orderbook_rest=True,
            orderbook_websocket=True,
            preferred_transport="rest",
        ),
        private_state=PrivateStateCapabilities(
            balances_rest=True,
            open_orders_rest=True,
            order_updates_websocket=True,
            fills_websocket=True,
            preferred_mode="rest",
        ),
        order_commands=OrderCommandCapabilities(
            place_rest=True, cancel_rest=True, market_order_rest=True
        ),
    ),
}


def supports_market_orders(exchange: str) -> bool:
    return get_exchange_capabilities(exchange).order_commands.market_order_rest


def get_exchange_capabilities(exchange: str) -> ExchangeCapabilities:
    key = exchange.lower().strip()
    try:
        return EXCHANGE_CAPABILITIES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported exchange: {exchange}") from exc


def validate_public_market_transport(exchange: str, mode: TransportMode) -> None:
    capabilities = get_exchange_capabilities(exchange)
    if not capabilities.public_market_data.supports(mode):
        details = capabilities.public_market_data.support_details(mode)
        missing = ", ".join(name for name, supported in details.items() if not supported)
        raise ValueError(
            f"{exchange} does not support complete public market data over {mode}"
            f" (missing: {missing or 'unknown'})"
        )


def resolve_public_market_transport(exchange: str, requested: str) -> TransportMode:
    mode = requested.lower().strip()
    capabilities = get_exchange_capabilities(exchange).public_market_data
    if mode not in ("rest", "websocket"):
        raise ValueError(f"unsupported public market data mode: {requested}")
    if mode == "websocket" and capabilities.supports("websocket"):
        return "websocket"
    if capabilities.supports("rest"):
        return "rest"
    raise ValueError(f"{exchange} has no complete public market data transport")


def validate_private_state_mode(exchange: str, mode: PrivateStateMode) -> None:
    capabilities = get_exchange_capabilities(exchange)
    if not capabilities.private_state.supports(mode):
        details = capabilities.private_state.support_details(mode)
        missing = ", ".join(name for name, supported in details.items() if not supported)
        raise ValueError(
            f"{exchange} does not support complete private state over {mode}"
            f" (missing: {missing or 'unknown'})"
        )


def resolve_private_state_mode(exchange: str, requested: str) -> PrivateStateMode:
    mode = requested.lower().strip()
    capabilities = get_exchange_capabilities(exchange).private_state
    if mode not in ("rest", "websocket"):
        raise ValueError(f"unsupported private state mode: {requested}")
    if mode == "websocket" and capabilities.supports("websocket"):
        return "websocket"
    if capabilities.supports("rest"):
        return "rest"
    raise ValueError(f"{exchange} has no complete private state mode")
