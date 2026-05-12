"""Nobitex IRT quote scaling — thin facade over ``instrument_wire`` (kept for call-site brevity)."""

from __future__ import annotations

from mm.core.models.market import PriceLevel, RawOrderbook, RawTrade
from mm.exchanges.instrument_wire import RIAL_PER_TOMAN_NOBITEX, build_instrument_wire

NOBITEX_RIAL_PER_TOMAN = RIAL_PER_TOMAN_NOBITEX


def nobitex_irt_quoted_pair(symbol: str) -> bool:
    """True when the domain pair's quote asset is ``IRT`` on Nobitex (Rial wire, Toman domain)."""

    try:
        return build_instrument_wire("nobitex", symbol).quote_asset == "IRT"
    except ValueError:
        return False


def scale_public_quote_price_from_api(symbol: str, price: float) -> float:
    return build_instrument_wire("nobitex", symbol).scale_public_price_from_api(float(price))


def scale_limit_price_to_nobitex_api(symbol: str, price: float) -> float:
    return build_instrument_wire("nobitex", symbol).scale_limit_price_to_api(float(price))


def scale_nobitex_orderbook_quotes(symbol: str, book: RawOrderbook) -> RawOrderbook:
    return build_instrument_wire("nobitex", symbol).scale_orderbook_from_api(book)


def scale_nobitex_trade_quote(symbol: str, trade: RawTrade) -> RawTrade:
    return build_instrument_wire("nobitex", symbol).scale_trade_from_api(trade)
