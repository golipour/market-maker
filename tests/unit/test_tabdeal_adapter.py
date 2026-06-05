from __future__ import annotations

import hashlib
import hmac
from urllib.parse import parse_qs, urlparse

import requests
import pytest

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


def test_tabdeal_runtime_credentials_allow_api_token_env_as_api_key(monkeypatch):
    monkeypatch.setenv("TABDEAL_API_TOKEN", "key")
    monkeypatch.setenv("TABDEAL_API_PRIVATE_KEY", "secret")
    cfg = PrivateRuntimeConfig(api_token_env="TABDEAL_API_TOKEN")

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


def test_tabdeal_get_balances_parses_account_balances(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        ok = True
        content = b"{}"

        def json(self):
            return {
                "balances": [
                    {"asset": "USDT", "free": "1.5", "freeze": "0.2"},
                    {"asset": "IRT", "free": "1000", "freeze": "50"},
                ]
            }

    def fake_request(method, url, params=None, headers=None, timeout=None):
        captured["path"] = urlparse(url).path
        return FakeResponse()

    monkeypatch.setattr(private_rest.requests, "request", fake_request)

    balances = private_rest.get_balances(private_rest.TabdealApiKeyAuth("key", "secret"))

    assert captured["path"] == "/r/api/v1/account"
    assert balances["USDT"] == {"free": 1.5, "locked": 0.2, "total": 1.7}
    assert balances["IRT"] == {"free": 1000.0, "locked": 50.0, "total": 1050.0}


def test_tabdeal_get_balances_aggregates_duplicate_asset_rows(monkeypatch):
    class FakeResponse:
        ok = True
        content = b"{}"

        def json(self):
            return {
                "balances": [
                    {"asset": "USDT", "free": "893.85902722", "freeze": "0"},
                    {"asset": "USDT", "free": "0", "freeze": "0"},
                    {"asset": "IRT", "free": "0.28593611", "freeze": "0"},
                    {"asset": "IRT", "free": "0", "freeze": "0"},
                ]
            }

    monkeypatch.setattr(private_rest.requests, "request", lambda *args, **kwargs: FakeResponse())

    balances = private_rest.get_balances(private_rest.TabdealApiKeyAuth("key", "secret"))

    assert balances["USDT"] == {"free": 893.85902722, "locked": 0.0, "total": 893.85902722}
    assert balances["IRT"] == {"free": 0.28593611, "locked": 0.0, "total": 0.28593611}


def test_tabdeal_get_balances_parses_wrapped_funding_balances(monkeypatch):
    calls: list[str] = []

    class FakeResponse:
        ok = True
        content = b"{}"

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_request(method, url, params=None, headers=None, timeout=None):
        path = urlparse(url).path
        calls.append(path)
        if path == "/r/api/v1/account":
            return FakeResponse({"balances": []})
        return FakeResponse({"data": [{"asset": "USDT", "free": "2", "locked": "0.3", "total": "2.3"}]})

    monkeypatch.setattr(private_rest.requests, "request", fake_request)

    balances = private_rest.get_balances(private_rest.TabdealApiKeyAuth("key", "secret"))

    assert calls == ["/r/api/v1/account", "/r/api/v1/asset/get-funding-asset"]
    assert balances["USDT"] == {"free": 2.0, "locked": 0.3, "total": 2.3}


def test_tabdeal_get_balances_missing_credentials_raises():
    with pytest.raises(RuntimeError, match="credentials missing"):
        private_rest.get_balances(None)


def test_tabdeal_get_balances_empty_responses_raise_useful_error(monkeypatch):
    class FakeResponse:
        ok = True
        content = b"{}"

        def json(self):
            return {}

    monkeypatch.setattr(private_rest.requests, "request", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="empty balance response"):
        private_rest.get_balances(private_rest.TabdealApiKeyAuth("key", "secret"))


def test_tabdeal_5xx_private_rest_errors_are_transient(monkeypatch):
    class FakeResponse:
        ok = False
        status_code = 503
        content = b"<html>nginx</html>"
        text = "<html>nginx</html>"

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(private_rest.requests, "request", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(private_rest.TabdealTransientError, match="503"):
        private_rest.list_filled_orders("USDT_IRT", private_rest.TabdealApiKeyAuth("key", "secret"))


@pytest.mark.asyncio
async def test_tabdeal_private_adapter_suppresses_transient_fill_poll_errors():
    adapter = TabdealRestPrivateStateAdapter(
        "USDT_IRT",
        api_token=private_rest.TabdealApiKeyAuth("key", "secret"),
    )

    await adapter._publish_error("fill_poll", private_rest.TabdealTransientError("Tabdeal HTTP 503"))

    assert adapter._error_queue.empty()


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
