from __future__ import annotations

from mm.exchanges.shared.instrument_spec_trade_stats import instrument_spec_from_trade_stats


def test_instrument_spec_from_exchange_info_filters(monkeypatch) -> None:
    monkeypatch.setattr(
        "mm.exchanges.shared.instrument_spec_trade_stats.get_tick_size",
        lambda symbol, exchange: 1.0,
    )
    monkeypatch.setattr(
        "mm.exchanges.shared.instrument_spec_trade_stats.get_lot_size",
        lambda symbol, exchange: 0.01,
    )
    monkeypatch.setattr(
        "mm.exchanges.shared.instrument_spec_trade_stats.get_exchange_info",
        lambda symbol, exchange: {
            "filters": [
                {"filterType": "LOT_SIZE", "minQty": "0.02"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "100"},
            ]
        },
    )

    spec = instrument_spec_from_trade_stats("nobitex", "USDT_IRT")

    assert spec.price_step == 0.1
    assert spec.qty_step == 0.01
    assert spec.min_qty == 0.02
    assert spec.min_notional == 10.0

    spec_identity = instrument_spec_from_trade_stats("nobitex", "BTC_USDT")
    assert spec_identity.price_step == 1.0
    assert spec_identity.min_notional == 100.0


def test_nobitex_irt_scales_api_quote_fields_to_domain(monkeypatch) -> None:
    monkeypatch.setattr(
        "mm.exchanges.shared.instrument_spec_trade_stats.get_tick_size",
        lambda symbol, exchange: 10.0,
    )
    monkeypatch.setattr(
        "mm.exchanges.shared.instrument_spec_trade_stats.get_lot_size",
        lambda symbol, exchange: 0.01,
    )
    monkeypatch.setattr(
        "mm.exchanges.shared.instrument_spec_trade_stats.get_exchange_info",
        lambda symbol, exchange: {
            "filters": [
                {"filterType": "LOT_SIZE", "minQty": "0.01"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "500000"},
            ]
        },
    )

    spec = instrument_spec_from_trade_stats("nobitex", "USDT_IRT")

    assert spec.price_step == 1.0
    assert spec.min_notional == 50_000.0
