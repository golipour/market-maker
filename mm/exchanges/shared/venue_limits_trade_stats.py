"""Derive ``VenueOrderLimits`` from ``trade_stats`` cached ``exchange_info`` (Nobitex, Wallex, …)."""

from __future__ import annotations

from typing import Any

from mm.core.interfaces.order_validation import AbstractVenueOrderLimitSource
from mm.core.models.venue_order_limits import VenueOrderLimits
from mm.exchanges.instrument_wire import build_instrument_wire
from mm.integrations.trade_stats import get_exchange_info, get_lot_size, get_tick_size


def _quote_from_api(exchange: str, symbol: str, api_quote: float | None) -> float:
    if api_quote is None or float(api_quote) <= 0.0:
        return 0.0
    wire = build_instrument_wire(exchange, symbol)
    return float(wire.scale_public_price_from_api(float(api_quote)))


def _limits_from_exchange_info(
    info: dict[str, Any],
    *,
    exchange: str = "",
    symbol: str = "",
    price_step: float | None = None,
    qty_step: float | None = None,
) -> VenueOrderLimits:
    min_base = 0.0
    min_notional = 0.0
    for f in info.get("filters") or []:
        if not isinstance(f, dict):
            continue
        ft = str(f.get("filterType", "") or "")
        if ft == "LOT_SIZE":
            mq = f.get("minQty")
            if mq is not None and str(mq).strip():
                try:
                    min_base = max(min_base, float(mq))
                except Exception:
                    pass
        elif ft == "MIN_NOTIONAL":
            mn = f.get("minNotional")
            if mn is not None and str(mn).strip():
                try:
                    v = float(mn)
                    if v > 0.0:
                        min_notional = max(min_notional, v)
                except Exception:
                    pass
    exchange_key = str(exchange or "").lower().strip()
    symbol_key = str(symbol or "").strip()
    if exchange_key and symbol_key:
        min_notional = _quote_from_api(exchange_key, symbol_key, min_notional)
        if price_step is not None:
            price_step = _quote_from_api(exchange_key, symbol_key, price_step)
    return VenueOrderLimits(
        min_base_qty=min_base,
        min_notional_quote=min_notional,
        price_step=float(price_step or 0.0),
        qty_step=float(qty_step or 0.0),
    )


class TradeStatsVenueOrderLimitSource(AbstractVenueOrderLimitSource):
    """Loads limits from the same ``exchange_info`` pipeline as ``trade_stats`` depth helpers."""

    def __init__(self, exchange: str) -> None:
        self._exchange = exchange.lower().strip()
        self._cache: dict[str, VenueOrderLimits] = {}

    def limits_for(self, symbol: str) -> VenueOrderLimits:
        key = str(symbol or "").strip()
        if key not in self._cache:
            info = get_exchange_info(key, self._exchange)
            self._cache[key] = _limits_from_exchange_info(
                info,
                exchange=self._exchange,
                symbol=key,
                price_step=float(get_tick_size(key, self._exchange)),
                qty_step=float(get_lot_size(key, self._exchange)),
            )
        return self._cache[key]
