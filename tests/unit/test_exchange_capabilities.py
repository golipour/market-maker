from __future__ import annotations

import pytest

from mm.exchanges.capabilities import (
    EXCHANGE_CAPABILITIES,
    ExchangeCapabilities,
    OrderCommandCapabilities,
    PrivateStateCapabilities,
    PublicMarketDataCapabilities,
    resolve_private_state_mode,
    resolve_public_market_transport,
    validate_public_market_transport,
)
from mm.exchanges.factory import select_public_market_data_adapter
from mm.exchanges.bitpin.rest import BitpinRestMarketDataAdapter
from mm.exchanges.bitpin.ws import BitpinWsMarketDataAdapter
from mm.exchanges.nobitex.rest import NobitexRestMarketDataAdapter
from mm.exchanges.nobitex.ws import NobitexWsMarketDataAdapter
from mm.exchanges.tabdeal.rest import TabdealRestMarketDataAdapter
from mm.exchanges.tabdeal.ws import TabdealWsMarketDataAdapter


def test_rest_is_default_transport_semantics() -> None:
    assert resolve_public_market_transport("nobitex", "rest") == "rest"
    assert resolve_private_state_mode("nobitex", "rest") == "rest"


def test_factory_returns_selected_single_transport_adapter() -> None:
    rest = select_public_market_data_adapter("nobitex", "BTC_IRT", mode="rest")
    websocket = select_public_market_data_adapter("nobitex", "BTC_IRT", mode="websocket")

    assert rest.transport == "rest"
    assert isinstance(rest.adapter, NobitexRestMarketDataAdapter)
    assert websocket.transport == "websocket"
    assert isinstance(websocket.adapter, NobitexWsMarketDataAdapter)

    tabdeal_rest = select_public_market_data_adapter("tabdeal", "USDT_IRT", mode="rest")
    tabdeal_ws = select_public_market_data_adapter("tabdeal", "USDT_IRT", mode="websocket")

    assert isinstance(tabdeal_rest.adapter, TabdealRestMarketDataAdapter)
    assert isinstance(tabdeal_ws.adapter, TabdealWsMarketDataAdapter)

    bitpin_rest = select_public_market_data_adapter("bitpin", "USDT_IRT", mode="rest")
    bitpin_ws = select_public_market_data_adapter("bitpin", "USDT_IRT", mode="websocket")

    assert isinstance(bitpin_rest.adapter, BitpinRestMarketDataAdapter)
    assert bitpin_ws.transport == "websocket"
    assert isinstance(bitpin_ws.adapter, BitpinWsMarketDataAdapter)


def test_websocket_request_falls_back_to_rest_for_unsupported_parts(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert resolve_public_market_transport("partial", "websocket") == "rest"
    assert resolve_private_state_mode("partial", "websocket") == "rest"
