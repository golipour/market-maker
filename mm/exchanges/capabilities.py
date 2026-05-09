from __future__ import annotations

from dataclasses import dataclass

from mm.exchanges.shared.provider_adapter import TransportMode


@dataclass(frozen=True, slots=True)
class PublicMarketDataCapabilities:
    rest: bool
    websocket: bool

    def supports(self, mode: TransportMode) -> bool:
        if mode == "rest":
            return self.rest
        if mode == "websocket":
            return self.websocket
        return False


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
        public_market_data=PublicMarketDataCapabilities(rest=True, websocket=True),
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
        public_market_data=PublicMarketDataCapabilities(rest=True, websocket=True),
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
        raise ValueError(f"{exchange} does not support public market data over {mode}")
