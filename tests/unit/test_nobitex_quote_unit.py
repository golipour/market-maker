from __future__ import annotations

from mm.exchanges.nobitex import private_rest
from mm.exchanges.nobitex.quote_unit import (
    nobitex_irt_quoted_pair,
    scale_limit_price_to_nobitex_api,
    scale_public_quote_price_from_api,
)


def test_nobitex_irt_pair_detection():
    assert nobitex_irt_quoted_pair("BTC_IRT")
    assert nobitex_irt_quoted_pair("USDT_IRT")
    assert not nobitex_irt_quoted_pair("BTC_USDT")
    assert not nobitex_irt_quoted_pair("")


def test_scale_public_quote_price_from_api():
    assert scale_public_quote_price_from_api("BTC_IRT", 1_000_000.0) == 100_000.0
    assert scale_public_quote_price_from_api("BTC_USDT", 1_000_000.0) == 1_000_000.0


def test_scale_limit_price_to_nobitex_api():
    assert scale_limit_price_to_nobitex_api("BTC_IRT", 100_000.0) == 1_000_000.0
    assert scale_limit_price_to_nobitex_api("BTC_USDT", 100_000.0) == 100_000.0


def test_merge_nobitex_wallet_balances_rls_and_irt_to_toman_irt():
    raw = {
        "RLS": {"free": 1000.0, "locked": 500.0, "total": 1500.0},
        "IRT": {"free": 200.0, "locked": 0.0, "total": 200.0},
        "USDT": {"free": 10.0, "locked": 0.0, "total": 10.0},
    }
    out = private_rest.merge_nobitex_wallet_balances_to_internal_toman(raw)
    # (1000+200)/10 = 120 IRT free; 500/10 = 50 locked
    assert out["IRT"]["free"] == 120.0
    assert out["IRT"]["locked"] == 50.0
    assert out["USDT"]["free"] == 10.0


def test_place_limit_order_scales_price_for_irt_pair(monkeypatch):
    captured: dict = {}

    def fake_post(path: str, api_token: str, data: dict | None = None):
        captured["data"] = dict(data or {})
        return {"status": "ok", "order": {"id": 1}}

    monkeypatch.setattr(private_rest, "_post_auth", fake_post)
    private_rest.place_limit_order("BTC_IRT", "buy", "0.1", 100_000.0, "token")
    assert captured["data"]["price"] == "1000000.0"

    private_rest.place_limit_order("BTC_USDT", "buy", "0.1", 50_000.0, "token")
    assert captured["data"]["price"] == "50000.0"
