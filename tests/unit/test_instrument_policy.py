from __future__ import annotations

from mm.core.models.instrument import SymbolBinding, build_instrument_spec, split_symbol
from mm.core.models.market import PriceLevel, RawOrderbook
from mm.runtime.print_events import summarize_event


def test_symbol_split_supports_domain_and_compact_symbols() -> None:
    assert split_symbol("BTC_IRT") == ("BTC", "IRT")
    assert split_symbol("BTCUSDT") == ("BTC", "USDT")
    assert split_symbol("eth-usdc") == ("ETH", "USDC")


def test_same_symbol_on_second_exchange_is_comparable_without_tick_conversion() -> None:
    primary = build_instrument_spec("nobitex", "BTC_IRT", price_step=1.0)
    secondary = build_instrument_spec("wallex", "BTC_IRT", price_step=10.0)
    binding = SymbolBinding(primary=primary, secondary=secondary)

    contract = binding.as_contract()

    assert binding.comparable is True
    assert contract["comparison"]["price_scale"] == "1:1"
    assert contract["comparison"]["qty_scale"] == "1:1"
    assert "order-boundary rounding" in contract["comparison"]["tick_size_effect"]


def test_printed_orderbook_events_expose_float_unit_policy() -> None:
    summary = summarize_event(
        RawOrderbook(
            exchange="nobitex",
            symbol="BTC_IRT",
            timestamp_ms=1000,
            bids=(PriceLevel(100.5, 0.25),),
            asks=(PriceLevel(101.0, 0.5),),
            is_snapshot=True,
        )
    )

    assert summary["instrument"]["base_asset"] == "BTC"
    assert summary["instrument"]["quote_asset"] == "IRT"
    assert summary["best_bid"] == 100.5
    assert summary["best_bid_qty"] == 0.25
    assert summary["unit_policy"] == "price=quote_per_base, qty=base"
