"""Derive ``VenueOrderLimits`` from ``trade_stats`` cached ``exchange_info`` (Nobitex, Wallex, …)."""

from __future__ import annotations

from typing import Any

from mm.core.interfaces.order_validation import AbstractVenueOrderLimitSource
from mm.core.models.venue_order_limits import VenueOrderLimits
from mm.integrations.trade_stats import get_exchange_info


def _limits_from_exchange_info(info: dict[str, Any]) -> VenueOrderLimits:
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
    return VenueOrderLimits(min_base_qty=min_base, min_notional_quote=min_notional)


class TradeStatsVenueOrderLimitSource(AbstractVenueOrderLimitSource):
    """Loads limits from the same ``exchange_info`` pipeline as ``trade_stats`` depth helpers."""

    def __init__(self, exchange: str) -> None:
        self._exchange = exchange.lower().strip()
        self._cache: dict[str, VenueOrderLimits] = {}

    def limits_for(self, symbol: str) -> VenueOrderLimits:
        key = str(symbol or "").strip()
        if key not in self._cache:
            info = get_exchange_info(key, self._exchange)
            self._cache[key] = _limits_from_exchange_info(info)
        return self._cache[key]
