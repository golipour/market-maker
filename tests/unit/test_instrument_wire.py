from __future__ import annotations

import pytest

from mm.exchanges.instrument_wire import (
    RIAL_PER_TOMAN_NOBITEX,
    build_instrument_wire,
    merge_wallet_balances_raw,
)


def test_default_identity_non_irt_pairs():
    w = build_instrument_wire("nobitex", "BTC_USDT")
    assert w.api_market_symbol == "BTCUSDT"
    assert w.private_order_currencies() == ("btc", "usdt")
    assert w.public_price_scale_from_api == 1.0
    assert w.scale_public_price_from_api(50_000.0) == 50_000.0


def test_nobitex_irt_pair_wire():
    w = build_instrument_wire("nobitex", "USDT_IRT")
    assert w.api_market_symbol == "USDTIRT"
    assert w.private_order_currencies() == ("usdt", "rls")
    assert w.public_price_scale_from_api == pytest.approx(1.0 / RIAL_PER_TOMAN_NOBITEX)
    assert w.scale_limit_price_to_api(50_000.0) == 500_000.0


def test_wallex_irt_domain_maps_to_tmn_market_symbol():
    w = build_instrument_wire("wallex", "USDT_IRT")
    assert w.api_market_symbol == "USDTTMN"
    assert w.private_order_currencies() == ("usdt", "irt")
    assert w.public_price_scale_from_api == 1.0


def test_merge_wallet_balances_wallex_folds_tmn_alias():
    raw = {"TMN": {"free": 100.0, "locked": 0.0, "total": 100.0}, "USDT": {"free": 1.0, "locked": 0.0, "total": 1.0}}
    out = merge_wallet_balances_raw("wallex", raw)
    assert out["IRT"]["free"] == 100.0
    assert out["USDT"]["free"] == 1.0


def test_merge_wallet_balances_unknown_exchange_uses_identity_canonical():
    raw = {"RLS": {"free": 10.0, "locked": 0.0, "total": 10.0}}
    out = merge_wallet_balances_raw("othercex", raw)
    assert out["IRT"]["free"] == 10.0
