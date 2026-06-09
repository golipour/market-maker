from __future__ import annotations

import pytest

from mm.exchanges.capabilities import EXCHANGE_CAPABILITIES
from mm.exchanges.factory import (
    select_order_command_adapter,
    select_private_state_adapter,
    select_public_market_data_adapter,
    select_venue_order_limit_source,
    supported_exchanges,
)
from mm.exchanges.registry import get_exchange_registration, registered_exchanges
from mm.exchanges.shared.venue_limits_trade_stats import TradeStatsVenueOrderLimitSource
from mm.execution.order_validation import NoOpVenueOrderLimitSource


def test_registered_exchanges_match_capabilities() -> None:
    assert set(registered_exchanges()) == set(EXCHANGE_CAPABILITIES)
    assert supported_exchanges() == registered_exchanges()


@pytest.mark.parametrize("exchange", registered_exchanges())
def test_registry_builds_all_adapter_kinds(exchange: str) -> None:
    registration = get_exchange_registration(exchange)
    public = select_public_market_data_adapter(exchange, "USDT_IRT", mode="rest")
    private = select_private_state_adapter(
        exchange,
        "USDT_IRT",
        mode="rest",
        api_token="token",
    )
    orders = select_order_command_adapter(exchange, "USDT_IRT", api_token="token")

    assert public.exchange == exchange
    assert private.exchange == exchange
    assert orders.exchange == exchange
    assert public.adapter is not None
    assert private.adapter is not None
    assert orders.adapter is not None
    assert registration.name == exchange


def test_unknown_exchange_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unsupported exchange"):
        get_exchange_registration("unknown-exchange")


def test_trade_stats_limit_source_uses_registration_flag() -> None:
    for exchange in registered_exchanges():
        source = select_venue_order_limit_source(exchange)
        registration = get_exchange_registration(exchange)
        if registration.uses_trade_stats_limits:
            assert isinstance(source, TradeStatsVenueOrderLimitSource)
        else:
            assert isinstance(source, NoOpVenueOrderLimitSource)
