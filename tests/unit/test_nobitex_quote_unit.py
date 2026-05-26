from __future__ import annotations

import base64

import requests

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


def test_nobitex_token_header_does_not_double_prefix():
    assert private_rest._token_authorization_header("abc") == "Token abc"
    assert private_rest._token_authorization_header("Token abc") == "Token abc"
    assert private_rest._token_authorization_header("Bearer abc") == "Bearer abc"


def test_nobitex_api_key_signs_open_order_query(monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    auth = private_rest.NobitexApiKeyAuth("public-key", private_pem)
    captured: dict = {}
    real_session = requests.sessions.Session()

    class FakeResponse:
        ok = True

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "ok", "orders": []}

    class FakeSession:
        def prepare_request(self, request):
            return real_session.prepare_request(request)

        def send(self, prepared, timeout):
            captured["path_url"] = prepared.path_url
            captured["headers"] = dict(prepared.headers)
            captured["body"] = prepared.body
            captured["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr(private_rest.requests, "Session", lambda: FakeSession())
    monkeypatch.setattr(private_rest.time, "time", lambda: 1_700_000_000)

    assert private_rest.list_open_orders("USDT_IRT", auth) == []

    assert captured["path_url"] == (
        "/market/orders/list?status=open&srcCurrency=usdt&dstCurrency=rls&details=2"
    )
    assert captured["headers"]["Nobitex-Key"] == "public-key"
    assert captured["headers"]["Nobitex-Timestamp"] == "1700000000"
    signed_message = f"1700000000GET{captured['path_url']}".encode("utf-8")
    public_key.verify(
        base64.b64decode(captured["headers"]["Nobitex-Signature"]),
        signed_message,
    )
