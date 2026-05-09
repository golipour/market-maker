from __future__ import annotations

import pytest

from mm.exchanges.capabilities import (
    EXCHANGE_CAPABILITIES,
    ExchangeCapabilities,
    OrderCommandCapabilities,
    PrivateStateCapabilities,
    PublicMarketDataCapabilities,
    resolve_public_market_transport,
    validate_public_market_transport,
)
from mm.exchanges.factory import select_public_market_data_adapter
from mm.exchanges.nobitex.rest import NobitexRestMarketDataAdapter
from mm.exchanges.nobitex.ws import NobitexWsMarketDataAdapter


def test_auto_public_market_transport_prefers_complete_websocket() -> None:
    assert resolve_public_market_transport("nobitex", "auto") == "websocket"


def test_factory_returns_selected_single_transport_adapter() -> None:
    rest = select_public_market_data_adapter("nobitex", "BTC_IRT", mode="rest")
    websocket = select_public_market_data_adapter("nobitex", "BTC_IRT", mode="websocket")

    assert rest.transport == "rest"
    assert isinstance(rest.adapter, NobitexRestMarketDataAdapter)
    assert websocket.transport == "websocket"
    assert isinstance(websocket.adapter, NobitexWsMarketDataAdapter)


def test_public_transport_requires_trades_and_orderbook_on_same_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        EXCHANGE_CAPABILITIES,
        "partial",
        ExchangeCapabilities(
            exchange="partial",
            public_market_data=PublicMarketDataCapabilities(
                trades_rest=True,
                trades_websocket=True,
                orderbook_rest=True,
                orderbook_websocket=False,
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
    )

    with pytest.raises(ValueError, match="missing: orderbook"):
        validate_public_market_transport("partial", "websocket")

    assert resolve_public_market_transport("partial", "auto") == "rest"
