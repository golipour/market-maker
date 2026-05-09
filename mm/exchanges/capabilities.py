from __future__ import annotations

from dataclasses import dataclass

from mm.exchanges.shared.provider_adapter import TransportMode


@dataclass(frozen=True, slots=True)
class PublicMarketDataCapabilities:
    trades_rest: bool
    trades_websocket: bool
    orderbook_rest: bool
    orderbook_websocket: bool
    preferred_transport: TransportMode = "websocket"

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


@dataclass(frozen=True, slots=True)
class OrderCommandCapabilities:
    place_rest: bool
    cancel_rest: bool
    amend_rest: bool = False


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
            preferred_transport="websocket",
        ),
        private_state=PrivateStateCapabilities(
            balances_rest=True,
            open_orders_rest=True,
            order_updates_websocket=False,
            fills_websocket=False,
        ),
        order_commands=OrderCommandCapabilities(place_rest=True, cancel_rest=True),
    ),
    "wallex": ExchangeCapabilities(
        exchange="wallex",
        public_market_data=PublicMarketDataCapabilities(
            trades_rest=True,
            trades_websocket=True,
            orderbook_rest=True,
            orderbook_websocket=True,
            preferred_transport="websocket",
        ),
        private_state=PrivateStateCapabilities(
            balances_rest=True,
            open_orders_rest=True,
            order_updates_websocket=False,
            fills_websocket=False,
        ),
        order_commands=OrderCommandCapabilities(place_rest=True, cancel_rest=True),
    ),
}


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
    if mode in ("rest", "websocket"):
        validate_public_market_transport(exchange, mode)  # type: ignore[arg-type]
        return mode  # type: ignore[return-value]
    if mode != "auto":
        raise ValueError(f"unsupported public market data mode: {requested}")

    if capabilities.supports(capabilities.preferred_transport):
        return capabilities.preferred_transport
    for candidate in ("websocket", "rest"):
        if capabilities.supports(candidate):
            return candidate
    raise ValueError(f"{exchange} has no complete public market data transport")
