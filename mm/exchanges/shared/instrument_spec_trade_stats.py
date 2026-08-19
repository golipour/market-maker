"""Build ``InstrumentSpec`` venue fields from cached ``exchange_info`` (Nobitex, Wallex, …)."""

from __future__ import annotations

from mm.core.models.instrument import InstrumentSpec, build_instrument_spec
from mm.exchanges.instrument_wire import build_instrument_wire
from mm.exchanges.shared.venue_limits_trade_stats import _limits_from_exchange_info
from mm.integrations.trade_stats import get_exchange_info, get_lot_size, get_tick_size

_TRADE_STATS_EXCHANGES = frozenset(
    {"nobitex", "wallex", "bitpin", "kucoin", "binance", "tabdeal", "bitwana"}
)


def _quote_from_api(exchange: str, symbol: str, api_quote: float | None) -> float | None:
    """Map API quote-denominated limits to domain quote units (e.g. Nobitex Rial → IRT Toman)."""

    if api_quote is None or float(api_quote) <= 0.0:
        return None
    wire = build_instrument_wire(exchange, symbol)
    return float(wire.scale_public_price_from_api(float(api_quote)))


def instrument_spec_from_trade_stats(exchange: str, symbol: str) -> InstrumentSpec:
    """Populate tick/lot/min from ``exchange_info`` in **domain** float units.

    Quote fields (``price_step``, ``min_notional``) use ``instrument_wire`` scaling so they
    match ``RawOrderbook`` / ``Order`` prices (Nobitex ``USDT_IRT``: API Rial × 0.1 → IRT).
    Base fields (``qty_step``, ``min_qty``) are already in base-asset units from the API.
    """

    exchange_key = exchange.lower().strip()
    if exchange_key not in _TRADE_STATS_EXCHANGES:
        return build_instrument_spec(exchange_key, symbol)

    price_step_api = float(get_tick_size(symbol, exchange_key))
    qty_step = float(get_lot_size(symbol, exchange_key))
    limits = _limits_from_exchange_info(
        get_exchange_info(symbol, exchange_key),
        exchange=exchange_key,
        symbol=symbol,
    )

    price_step = _quote_from_api(exchange_key, symbol, price_step_api)
    min_notional = _quote_from_api(
        exchange_key,
        symbol,
        limits.min_notional_quote if limits.min_notional_quote > 0.0 else None,
    )

    return build_instrument_spec(
        exchange_key,
        symbol,
        price_step=price_step,
        qty_step=qty_step if qty_step > 0.0 else None,
        min_qty=limits.min_base_qty if limits.min_base_qty > 0.0 else None,
        min_notional=min_notional,
    )
