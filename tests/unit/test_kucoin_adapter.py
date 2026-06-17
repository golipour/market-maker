from __future__ import annotations

import base64
import hashlib
import hmac
import json
from urllib.parse import urlparse

import pytest

from mm.core.models.execution import Order
from mm.exchanges.factory import (
    select_order_command_adapter,
    select_private_state_adapter,
)
from mm.exchanges.kucoin import private_rest
from mm.exchanges.kucoin.order_adapter import KucoinRestOrderCommandAdapter
from mm.exchanges.kucoin.private_state import (
    KucoinRestPrivateStateAdapter,
    KucoinWebSocketPrivateStateAdapter,
)
from mm.exchanges.kucoin.rest import KucoinRestMarketDataAdapter
from mm.runtime.config import PrivateRuntimeConfig


def _auth() -> private_rest.KucoinApiKeyAuth:
    return private_rest.KucoinApiKeyAuth("key", "secret", "passphrase")


def test_kucoin_runtime_credentials_use_api_key_secret_and_passphrase():
    cfg = PrivateRuntimeConfig(
        api_key="key",
        api_private_key="secret",
        api_passphrase="passphrase",
    )

    auth = cfg.resolved_api_credentials("kucoin")

    assert isinstance(auth, private_rest.KucoinApiKeyAuth)
    assert auth.api_key == "key"
    assert auth.api_secret == "secret"
    assert auth.api_passphrase == "passphrase"


def test_kucoin_runtime_credentials_allow_env_vars(monkeypatch):
    monkeypatch.setenv("KUCOIN_API_KEY", "key")
    monkeypatch.setenv("KUCOIN_API_PRIVATE_KEY", "secret")
    monkeypatch.setenv("KUCOIN_API_PASSPHRASE", "passphrase")
    cfg = PrivateRuntimeConfig(api_key_env="KUCOIN_API_KEY")

    auth = cfg.resolved_api_credentials("kucoin")

    assert isinstance(auth, private_rest.KucoinApiKeyAuth)
    assert auth.api_key == "key"
    assert auth.api_secret == "secret"
    assert auth.api_passphrase == "passphrase"


def test_kucoin_private_state_factory_modes():
    auth = _auth()

    rest = select_private_state_adapter("kucoin", "BTC_USDT", mode="rest", api_token=auth)
    ws = select_private_state_adapter("kucoin", "BTC_USDT", mode="websocket", api_token=auth)

    assert isinstance(rest.adapter, KucoinRestPrivateStateAdapter)
    assert isinstance(ws.adapter, KucoinWebSocketPrivateStateAdapter)


def test_kucoin_order_command_factory():
    auth = _auth()

    selected = select_order_command_adapter("kucoin", "BTC_USDT", api_token=auth)

    assert isinstance(selected.adapter, KucoinRestOrderCommandAdapter)


def test_kucoin_api_symbol_uses_hyphen_separator():
    assert private_rest.api_symbol("BTC_USDT") == "BTC-USDT"


def test_kucoin_signed_request_uses_hmac_sha256_base64_signature(monkeypatch):
    auth = _auth()
    captured: dict = {}

    class FakeResponse:
        ok = True
        content = b'{"code":"200000","data":{"items":[]}}'

        def json(self):
            return {"code": "200000", "data": {"items": []}}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = dict(params or {})
        captured["headers"] = dict(headers or {})
        return FakeResponse()

    monkeypatch.setattr(private_rest.time, "time", lambda: 1_700_000_000.123)
    monkeypatch.setattr(private_rest.requests, "get", fake_get)

    assert private_rest.list_open_orders("BTC_USDT", auth) == []

    headers = captured["headers"]
    assert headers["KC-API-KEY"] == "key"
    assert headers["KC-API-KEY-VERSION"] == "2"
    expected_passphrase = base64.b64encode(
        hmac.new(b"secret", b"passphrase", hashlib.sha256).digest()
    ).decode("utf-8")
    assert headers["KC-API-PASSPHRASE"] == expected_passphrase
    endpoint = "/api/v1/orders?status=active&symbol=BTC-USDT"
    str_to_sign = "1700000000123GET" + endpoint
    expected_sign = base64.b64encode(
        hmac.new(b"secret", str_to_sign.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")
    assert headers["KC-API-SIGN"] == expected_sign


def test_kucoin_get_balances_parses_trade_accounts(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        ok = True
        content = b"{}"

        def json(self):
            return {
                "code": "200000",
                "data": [
                    {"currency": "USDT", "available": "1.5", "holds": "0.2", "balance": "1.7"},
                    {"currency": "BTC", "available": "0.01", "holds": "0", "balance": "0.01"},
                ],
            }

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["path"] = urlparse(url).path
        captured["params"] = dict(params or {})
        return FakeResponse()

    monkeypatch.setattr(private_rest.requests, "get", fake_get)

    balances = private_rest.get_balances(_auth())

    assert captured["path"] == "/api/v1/accounts"
    assert captured["params"] == {"type": "trade"}
    assert balances["USDT"] == {"free": 1.5, "locked": 0.2, "total": 1.7}
    assert balances["BTC"] == {"free": 0.01, "locked": 0.0, "total": 0.01}


def test_kucoin_place_limit_order_posts_json_body(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        ok = True
        content = b"{}"

        def json(self):
            return {"code": "200000", "data": {"orderId": "oid-1"}}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = dict(headers or {})
        return FakeResponse()

    monkeypatch.setattr(private_rest.requests, "post", fake_post)

    out = private_rest.place_limit_order("BTC_USDT", "buy", "0.1", "50000", _auth(), client_order_id="cid-1")

    assert out["orderId"] == "oid-1"
    body = json.loads(captured["data"])
    assert body == {
        "symbol": "BTC-USDT",
        "side": "buy",
        "type": "limit",
        "size": "0.1",
        "price": "50000",
        "clientOid": "cid-1",
    }


def test_kucoin_cancel_order_uses_delete_path(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        ok = True
        content = b"{}"

        def json(self):
            return {"code": "200000", "data": {"cancelledOrderIds": ["oid-1"]}}

    def fake_delete(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        return FakeResponse()

    monkeypatch.setattr(private_rest.requests, "delete", fake_delete)

    out = private_rest.cancel_order("BTC_USDT", "oid-1", _auth())

    assert out["cancelledOrderIds"] == ["oid-1"]
    assert captured["url"].endswith("/api/v1/orders/oid-1")


@pytest.mark.asyncio
async def test_kucoin_rest_market_data_parses_wrapped_payload(monkeypatch):
    adapter = KucoinRestMarketDataAdapter("BTC_USDT", depth_limit=20, trades_limit=10)

    async def fake_get_json(path, params=None):
        if path == "/api/v1/market/histories":
            return {
                "code": "200000",
                "data": [
                    {"time": 1_700_000_000_000, "price": "50000", "size": "0.01", "side": "buy"},
                ],
            }
        if path == "/api/v1/market/orderbook/level2_20":
            return {
                "code": "200000",
                "data": {
                    "time": 1_700_000_000_000,
                    "bids": [["49999", "1"]],
                    "asks": [["50001", "2"]],
                },
            }
        raise AssertionError(path)

    adapter._client.get_json = fake_get_json  # type: ignore[method-assign]

    trades = await adapter.fetch_trades()
    book = await adapter.fetch_orderbook()

    assert len(trades) == 1
    assert trades[0].price == 50000.0
    assert trades[0].side == "buy"
    assert len(book.bids) == 1
    assert len(book.asks) == 1


def test_kucoin_normalize_open_order_maps_remain_size():
    adapter = KucoinRestPrivateStateAdapter("BTC_USDT", api_token=_auth())
    row = {
        "id": "oid-1",
        "clientOid": "actor-1",
        "side": "buy",
        "price": "50000",
        "size": "1",
        "dealSize": "0.2",
        "remainSize": "0.8",
        "type": "limit",
        "status": "active",
        "createdAt": 1_700_000_000_000,
    }

    order = adapter.normalize_open_order(row, timestamp_ms=1)

    assert order is not None
    assert order.exchange_order_id == "oid-1"
    assert order.remaining_qty == 0.8
    assert order.filled_qty == 0.2



@pytest.mark.asyncio
async def test_kucoin_order_adapter_place_orders(monkeypatch):
    auth = _auth()
    adapter = KucoinRestOrderCommandAdapter("BTC_USDT", api_token=auth)
    monkeypatch.setattr(
        private_rest,
        "place_limit_order",
        lambda *args, **kwargs: {"orderId": "oid-1"},
    )
    ids = await adapter.place_orders([
        Order(
            exchange="kucoin",
            symbol="BTC_USDT",
            side="buy",
            price=50000.0,
            qty=0.1,
            client_order_id="cid-1",
        )
    ])
    assert ids == ["oid-1"]
