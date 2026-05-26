from __future__ import annotations

from mm.integrations import trade_stats
from mm.exchanges.shared.instrument_spec_trade_stats import instrument_spec_from_trade_stats


class _ExchangeInfoAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[str] = []

    def exchange_info(self, symbol: str) -> dict:
        self.calls.append(symbol)
        return {"exchange": self.name, "symbol": symbol}


def test_get_exchange_info_uses_exchange_specific_cache(monkeypatch) -> None:
    nobitex = _ExchangeInfoAdapter("nobitex")
    wallex = _ExchangeInfoAdapter("wallex")
    monkeypatch.setattr(trade_stats, "EXCHANGES", {"nobitex": nobitex, "wallex": wallex})
    monkeypatch.setattr(trade_stats, "_exchange_info_cache", {})

    assert trade_stats.get_exchange_info("USDT_IRT", "nobitex") == {
        "exchange": "nobitex",
        "symbol": "USDT_IRT",
    }
    assert trade_stats.get_exchange_info("USDT_IRT", "wallex") == {
        "exchange": "wallex",
        "symbol": "USDT_IRT",
    }
    assert trade_stats.get_exchange_info("USDT_IRT", " NOBITEX ") == {
        "exchange": "nobitex",
        "symbol": "USDT_IRT",
    }
    assert nobitex.calls == ["USDT_IRT"]
    assert wallex.calls == ["USDT_IRT"]


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
