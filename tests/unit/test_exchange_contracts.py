from __future__ import annotations

import pytest

from mm.core.interfaces.market_data import PublicMarketDataAdapter
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.exchanges.capabilities import get_exchange_capabilities, validate_public_market_transport
from mm.exchanges.factory import build_public_market_data_adapter
from mm.exchanges.nobitex.rest import NobitexRestMarketDataAdapter
from mm.exchanges.wallex.ws import WallexWsMarketDataAdapter


def test_exchange_capabilities_keep_public_private_execution_separate():
    capabilities = get_exchange_capabilities("nobitex")

    assert capabilities.public_market_data.rest is True
    assert capabilities.public_market_data.websocket is True
    assert capabilities.private_state.balances_rest is True
    assert capabilities.private_state.open_orders_rest is True
    assert capabilities.order_commands.place_rest is True


def test_public_market_factory_selects_one_transport_adapter():
    rest_adapter = build_public_market_data_adapter(
        "nobitex",
        "BTC_IRT",
        mode="rest",
        depth_limit=2,
        trades_limit=2,
    )
    ws_adapter = build_public_market_data_adapter("wallex", "BTC_IRT", mode="websocket")

    assert isinstance(rest_adapter, PublicMarketDataAdapter)
    assert isinstance(rest_adapter, NobitexRestMarketDataAdapter)
    assert isinstance(ws_adapter, WallexWsMarketDataAdapter)


def test_unsupported_exchange_fails_before_runtime_wiring():
    with pytest.raises(ValueError, match="unsupported exchange"):
        validate_public_market_transport("missing", "rest")


def test_private_state_models_use_float_domain_values():
    balance = Balance(
        exchange="nobitex",
        asset="BTC",
        free=1.25,
        locked=0.5,
        total=1.75,
        timestamp_ms=1000,
    )
    snapshot = PrivateStateSnapshot(exchange="nobitex", timestamp_ms=1000, balances={"BTC": balance})

    assert snapshot.balances["BTC"].free == 1.25
    assert OpenOrder.__name__ == "OpenOrder"
    assert OrderUpdate.__name__ == "OrderUpdate"
