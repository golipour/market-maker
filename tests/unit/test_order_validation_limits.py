from __future__ import annotations

from mm.core.interfaces.order_validation import order_passes_limits
from mm.core.models.execution import Order
from mm.core.models.venue_order_limits import VenueOrderLimits
from mm.exchanges.shared.venue_limits_trade_stats import _limits_from_exchange_info
from mm.integrations.trade_stats import WallexAdapter


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


def test_wallex_exchange_info_exposes_quantity_step_and_minimums(monkeypatch) -> None:
    def fake_get(self, path, params=None):
        _ = self, path, params
        return {
            "success": True,
            "result": {
                "symbols": {
                    "USDTTMN": {
                        "tickSize": 0,
                        "stepSize": 2,
                        "minQty": 0.01,
                        "minNotional": 50000,
                    }
                }
            },
        }

    monkeypatch.setattr(WallexAdapter, "_get", fake_get)

    info = WallexAdapter().exchange_info("USDT_IRT")
    lim = _limits_from_exchange_info(
        info,
        exchange="wallex",
        symbol="USDT_IRT",
        price_step=1.0,
        qty_step=0.01,
    )

    assert lim.price_step == 1.0
    assert lim.qty_step == 0.01
    assert lim.min_base_qty == 0.01
    assert lim.min_notional_quote == 50000.0
