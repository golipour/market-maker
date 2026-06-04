from __future__ import annotations

import hashlib
import hmac
from urllib.parse import parse_qs, urlparse

import requests

from mm.core.models.execution import Order
from mm.exchanges.factory import (
    select_order_command_adapter,
    select_private_state_adapter,
)
from mm.exchanges.tabdeal import private_rest
from mm.exchanges.tabdeal.order_adapter import TabdealRestOrderCommandAdapter
from mm.exchanges.tabdeal.private_state import (
    TabdealRestPrivateStateAdapter,
    TabdealWebSocketPrivateStateAdapter,
)
from mm.runtime.config import PrivateRuntimeConfig


def test_tabdeal_runtime_credentials_use_api_key_and_private_key_as_secret():
    cfg = PrivateRuntimeConfig(api_key="key", api_private_key="secret")

    auth = cfg.resolved_api_credentials("tabdeal")

    assert isinstance(auth, private_rest.TabdealApiKeyAuth)
    assert auth.api_key == "key"
    assert auth.api_secret == "secret"


def test_tabdeal_private_state_factory_modes():
    auth = private_rest.TabdealApiKeyAuth("key", "secret")

    rest = select_private_state_adapter("tabdeal", "USDT_IRT", mode="rest", api_token=auth)
    ws = select_private_state_adapter("tabdeal", "USDT_IRT", mode="websocket", api_token=auth)

    assert isinstance(rest.adapter, TabdealRestPrivateStateAdapter)
    assert isinstance(ws.adapter, TabdealWebSocketPrivateStateAdapter)


def test_tabdeal_order_command_factory():
    auth = private_rest.TabdealApiKeyAuth("key", "secret")

    selected = select_order_command_adapter("tabdeal", "USDT_IRT", api_token=auth)

    assert isinstance(selected.adapter, TabdealRestOrderCommandAdapter)


def test_tabdeal_signed_request_uses_hmac_sha256_query_signature(monkeypatch):
    auth = private_rest.TabdealApiKeyAuth("key", "secret")
    captured: dict = {}

    class FakeResponse:
        ok = True
        content = b"[]"

        def json(self):
            return []

    def fake_request(method, url, params=None, headers=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["params"] = dict(params or {})
        captured["headers"] = dict(headers or {})
        captured["timeout"] = timeout
        request = requests.Request(method, url, params=params)
        captured["path_url"] = requests.Session().prepare_request(request).path_url
        return FakeResponse()

    monkeypatch.setattr(private_rest.time, "time", lambda: 1_700_000_000.123)
    monkeypatch.setattr(private_rest.requests, "request", fake_request)

    assert private_rest.list_open_orders("USDT_IRT", auth) == []

    params = captured["params"]
    signature = params.pop("signature")
    query = "symbol=USDTIRT&timestamp=1700000000123"
    expected = hmac.new(b"secret", query.encode("utf-8"), hashlib.sha256).hexdigest()
    assert params == {"symbol": "USDTIRT", "timestamp": 1700000000123}
    assert signature == expected
    assert captured["headers"]["X-MBX-APIKEY"] == "key"


def test_tabdeal_place_limit_order_payload(monkeypatch):
    captured: dict = {}

    def fake_request(method, url, params=None, headers=None, timeout=None):
        captured["method"] = method
        captured["path"] = urlparse(url).path
        captured["params"] = dict(params or {})

        class FakeResponse:
            ok = True
            content = b"{}"

            def json(self):
                return {"orderId": 42, "status": "NEW"}

        return FakeResponse()

    monkeypatch.setattr(private_rest.time, "time", lambda: 1_700_000_000.0)
    monkeypatch.setattr(private_rest.requests, "request", fake_request)

    out = private_rest.place_limit_order(
        "USDT_IRT",
        "buy",
        "1.5",
        "100000",
        private_rest.TabdealApiKeyAuth("key", "secret"),
        client_order_id="client-1",
    )

    assert out["orderId"] == 42
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/order"
    assert captured["params"]["symbol"] == "USDTIRT"
    assert captured["params"]["side"] == "BUY"
    assert captured["params"]["type"] == "LIMIT"
    assert captured["params"]["quantity"] == "1.5"
    assert captured["params"]["price"] == "100000.0"
    assert captured["params"]["newClientOrderId"] == "client-1"
