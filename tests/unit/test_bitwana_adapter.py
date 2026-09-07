from __future__ import annotations

import pytest

from mm.exchanges.bitwana import private_rest
from mm.exchanges.bitwana.order_adapter import BitwanaRestOrderCommandAdapter
from mm.exchanges.bitwana.private_state import BitwanaRestPrivateStateAdapter
from mm.exchanges.bitwana.rest import BitwanaRestMarketDataAdapter
from mm.exchanges.factory import (
    select_order_command_adapter,
    select_private_state_adapter,
    select_public_market_data_adapter,
)
from mm.runtime.config import PrivateRuntimeConfig


def test_bitwana_runtime_credentials_use_access_and_refresh_tokens():
    cfg = PrivateRuntimeConfig(
        auth_mode="signed_key",
        api_key="access-token",
        api_secret="refresh-token",
    )

    auth = cfg.resolved_api_credentials("bitwana")

    assert isinstance(auth, private_rest.BitwanaJwtAuth)
    assert auth.access_token == "access-token"
    assert auth.refresh_token == "refresh-token"
    assert auth.authorization_header() == "access-token"


def test_bitwana_authorization_header_is_raw_jwt() -> None:
    auth = private_rest.BitwanaJwtAuth("access-token", "refresh-token")
    assert auth.authorization_header() == "access-token"
    assert private_rest.BitwanaJwtAuth("Bearer access-token").authorization_header() == "access-token"


def test_bitwana_runtime_credentials_from_env(monkeypatch):
    monkeypatch.setenv("BITWANA_API_TOKEN", "access-token")
    monkeypatch.setenv("BITWANA_REFRESH_TOKEN", "refresh-token")
    cfg = PrivateRuntimeConfig(api_token_env="BITWANA_API_TOKEN")

    auth = cfg.resolved_api_credentials("bitwana")

    assert isinstance(auth, private_rest.BitwanaJwtAuth)
    assert auth.access_token == "access-token"
    assert auth.refresh_token == "refresh-token"


def test_bitwana_factories_return_rest_adapters():
    auth = private_rest.BitwanaJwtAuth("access", "refresh")

    public = select_public_market_data_adapter("bitwana", "USDT_IRT", mode="websocket")
    private = select_private_state_adapter("bitwana", "USDT_IRT", mode="rest", api_token=auth)
    orders = select_order_command_adapter("bitwana", "USDT_IRT", api_token=auth)

    assert public.transport == "rest"
    assert isinstance(public.adapter, BitwanaRestMarketDataAdapter)
    assert isinstance(private.adapter, BitwanaRestPrivateStateAdapter)
    assert isinstance(orders.adapter, BitwanaRestOrderCommandAdapter)


@pytest.mark.asyncio
async def test_bitwana_rest_market_data_normalizes_level2_and_recent_trades():
    class FakeClient:
        async def get_json(self, path, params=None):
            if path.endswith("/level2"):
                return {
                    "statusCode": 200,
                    "content": {
                        "bids": [{"price": 100, "size": 2}],
                        "asks": [{"price": 101, "size": 3}],
                    },
                }
            if "/trades/recent/market/" in path:
                return {
                    "statusCode": 200,
                    "content": [
                        {
                            "amount": 1.25,
                            "unitPrice": 100.5,
                            "time": 1_700_000_000,
                            "takerSide": "S",
                        }
                    ],
                }
            raise AssertionError(path)

    adapter = BitwanaRestMarketDataAdapter("USDT_IRT", depth_limit=5)
    adapter._client = FakeClient()  # type: ignore[assignment]

    book = await adapter.fetch_orderbook()
    trades = await adapter.fetch_trades()

    assert book.exchange == "bitwana"
    assert book.bids[0].price == 100.0
    assert book.asks[0].qty == 3.0
    assert trades[0].side == "sell"
    assert trades[0].qty == 1.25
    assert trades[0].timestamp_ms == 1_700_000_000_000


def test_bitwana_api_symbol_keeps_underscore():
    assert private_rest.api_symbol("USDT_IRT") == "USDT_IRT"


def test_bitwana_open_orders_unwraps_page_and_skips_closed(monkeypatch):
    calls: list[str] = []

    class FakeResponse:
        status_code = 200
        ok = True
        content = b"{}"

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_request(method, url, params=None, json=None, headers=None, timeout=None):
        calls.append(url)
        assert headers is not None
        assert headers.get("Authorization") == "access"
        return FakeResponse(
            {
                "statusCode": 200,
                "content": {
                    "content": [
                        {
                            "id": 1,
                            "marketName": "USDT_IRT",
                            "status": "OPEN",
                            "side": "BUY",
                            "size": 2,
                            "remainedSize": 1.5,
                            "unitPrice": 100,
                        },
                        {
                            "id": 2,
                            "marketName": "USDT_IRT",
                            "status": "DONE",
                            "side": "SELL",
                            "size": 1,
                            "remainedSize": 0,
                            "unitPrice": 101,
                        },
                    ]
                },
            }
        )

    monkeypatch.setattr(private_rest.requests, "request", fake_request)
    auth = private_rest.BitwanaJwtAuth("access", "refresh")
    rows = private_rest.list_open_orders("USDT_IRT", auth)
    assert calls and calls[0].endswith("/api/1/orders/open/self")
    assert [row["id"] for row in rows] == [1]
    assert rows[0]["remainingAmount"] == 1.5
    assert rows[0]["price"] == 100


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.ok = 200 <= int(status_code) < 400
        self._payload = payload
        self.content = b"{}" if payload is not None else b""
        self.text = ""

    def json(self):
        return self._payload


def test_bitwana_unwrap_accepts_accepted_cancel_status() -> None:
    assert private_rest._unwrap({"statusCode": 202, "content": None}) is None
    assert private_rest._unwrap({"statusCode": "202", "content": {"id": 9}}) == {"id": 9}
    assert private_rest._unwrap({"statusCode": 204, "content": {}}) == {}


def test_bitwana_unwrap_still_rejects_error_status() -> None:
    with pytest.raises(RuntimeError, match="Bitwana HTTP 400"):
        private_rest._unwrap({"statusCode": 400, "content": None, "message": "bad"})


def test_bitwana_cancel_treats_202_body_as_success(monkeypatch) -> None:
    def fake_request(method, url, params=None, json=None, headers=None, timeout=None):
        assert method == "PUT"
        assert url.endswith("/api/1/orders/cancel/974646013")
        return _FakeResponse({"statusCode": 202, "message": "bitwana error", "content": None})

    monkeypatch.setattr(private_rest.requests, "request", fake_request)
    auth = private_rest.BitwanaJwtAuth("access", "refresh")
    result = private_rest.cancel_order("974646013", auth)
    assert result["success"] is True
    assert result["id"] == 974646013


@pytest.mark.asyncio
async def test_bitwana_adapter_cancel_does_not_fail_on_202(monkeypatch) -> None:
    def fake_request(method, url, params=None, json=None, headers=None, timeout=None):
        return _FakeResponse({"statusCode": 202, "content": None})

    monkeypatch.setattr(private_rest.requests, "request", fake_request)
    adapter = BitwanaRestOrderCommandAdapter(
        "USDT_IRT",
        api_token=private_rest.BitwanaJwtAuth("access", "refresh"),
    )
    await adapter.cancel_orders(["974646013", "974646016"])


def test_bitwana_cancel_treats_missing_order_as_success(monkeypatch) -> None:
    class NotFound:
        status_code = 404
        ok = False
        content = b"{}"
        text = "missing"

        def json(self):
            return {"statusCode": 404, "message": "not found"}

    def fake_request(method, url, params=None, json=None, headers=None, timeout=None):
        return NotFound()

    monkeypatch.setattr(private_rest.requests, "request", fake_request)
    auth = private_rest.BitwanaJwtAuth("access", "refresh")
    result = private_rest.cancel_order(974646017, auth)
    assert result["success"] is True
    assert result["already_gone"] is True
