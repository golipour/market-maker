from __future__ import annotations

from mm.core.interfaces.order_validation import order_passes_limits
from mm.core.models.execution import Order
from mm.core.models.venue_order_limits import VenueOrderLimits
from mm.exchanges.shared.venue_limits_trade_stats import _limits_from_exchange_info


def test_order_passes_limits_base() -> None:
    lim = VenueOrderLimits(min_base_qty=0.01, min_notional_quote=100.0)
    o = Order(exchange="nobitex", symbol="USDT_IRT", side="buy", price=50_000.0, qty=0.001)
    ok, _ = order_passes_limits(o, lim)
    assert not ok


def test_order_passes_limits_notional() -> None:
    lim = VenueOrderLimits(min_base_qty=0.0, min_notional_quote=1_000_000.0)
    o = Order(exchange="nobitex", symbol="USDT_IRT", side="buy", price=50_000.0, qty=0.01)
    ok, _ = order_passes_limits(o, lim)
    assert not ok


def test_limits_from_exchange_info_filters() -> None:
    info = {
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": "0.02"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "500000"},
        ]
    }
    lim = _limits_from_exchange_info(info)
    assert lim.min_base_qty == 0.02
    assert lim.min_notional_quote == 500000.0
