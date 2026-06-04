from __future__ import annotations

import requests

import pytest

from mm.core.models.execution import Order
from mm.exchanges.bitpin import private_rest
from mm.exchanges.bitpin.rest import BitpinRestMarketDataAdapter
from mm.exchanges.bitpin.order_adapter import BitpinRestOrderCommandAdapter
from mm.exchanges.bitpin.private_state import (
    BitpinRestPrivateStateAdapter,
    BitpinWebSocketPrivateStateAdapter,
)
from mm.exchanges.factory import (
    select_order_command_adapter,
    select_private_state_adapter,
)
from mm.runtime.config import PrivateRuntimeConfig


def test_bitpin_runtime_credentials_use_api_key_and_private_key_as_secret():
    cfg = PrivateRuntimeConfig(api_key="key", api_private_key="secret")

    auth = cfg.resolved_api_credentials("bitpin")

    assert isinstance(auth, private_rest.BitpinApiKeyAuth)
    assert auth.api_key == "key"
    assert auth.api_secret == "secret"


def test_bitpin_factories_return_rest_adapters():
    auth = private_rest.BitpinTokenAuth("access")

    private = select_private_state_adapter("bitpin", "USDT_IRT", mode="rest", api_token=auth)
    orders = select_order_command_adapter("bitpin", "USDT_IRT", api_token=auth)

    assert isinstance(private.adapter, BitpinRestPrivateStateAdapter)
    assert isinstance(orders.adapter, BitpinRestOrderCommandAdapter)


def test_bitpin_private_state_factory_returns_websocket_adapter():
    auth = private_rest.BitpinApiKeyAuth("key", "secret")

    private = select_private_state_adapter("bitpin", "USDT_IRT", mode="websocket", api_token=auth)

    assert isinstance(private.adapter, BitpinWebSocketPrivateStateAdapter)


@pytest.mark.asyncio
async def test_bitpin_rest_market_data_uses_v2_orderbook_endpoint():
    calls: list[tuple[str, dict | None]] = []

    class FakeClient:
        async def get_json(self, path, params=None):
            calls.append((path, params))
            if path == "/v1/mkt/markets/":
                return {"results": [{"id": 5, "code": "USDT_IRT"}], "next": None}
            if path == "/v2/mth/actives/5/" and params == {"type": "buy"}:
                return {"orders": [{"price": "50000", "remain": "2"}]}
            if path == "/v2/mth/actives/5/" and params == {"type": "sell"}:
                return {"orders": [{"price": "51000", "remain": "3"}]}
            raise AssertionError(f"unexpected request: {path} {params}")

    adapter = BitpinRestMarketDataAdapter("USDT_IRT", depth_limit=5)
    adapter._client = FakeClient()  # type: ignore[assignment]

    book = await adapter.fetch_orderbook()

    assert calls == [
        ("/v1/mkt/markets/", {"page": 1}),
        ("/v2/mth/actives/5/", {"type": "buy"}),
        ("/v2/mth/actives/5/", {"type": "sell"}),
    ]
    assert book.bids[0].price == pytest.approx(50000.0)
    assert book.bids[0].qty == pytest.approx(2.0)
    assert book.asks[0].price == pytest.approx(51000.0)
    assert book.asks[0].qty == pytest.approx(3.0)


def test_bitpin_login_and_authorized_request(monkeypatch):
    private_rest._ACCESS_TOKEN_CACHE.clear()
    private_rest._MARKET_ID_CACHE.clear()
    captured: list[dict] = []

    class FakeResponse:
        ok = True
        content = b"{}"
        text = ""

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.append({"method": "POST", "url": url, "json": json, "headers": headers, "timeout": timeout})
        if url.endswith("/usr/api/login/"):
            return FakeResponse({"access": "access-token", "refresh": "refresh-token"})
        if url.endswith("/odr/orders/bulk"):
            return FakeResponse({"orders": [{"id": 42, "state": "active"}]})
        return FakeResponse({"id": 42, "state": "initial"})

    def fake_get(url, params=None, headers=None, timeout=None):
        captured.append({"method": "GET", "url": url, "params": params, "headers": headers, "timeout": timeout})
        if url.endswith("/mkt/markets/"):
            return FakeResponse({"results": [{"id": 5, "code": "USDT_IRT"}], "next": None})
        return FakeResponse({"results": [{"currency": {"code": "USDT"}, "balance": "1.5", "blocked": "0.2"}]})

    monkeypatch.setattr(private_rest.requests, "post", fake_post)
    monkeypatch.setattr(private_rest.requests, "get", fake_get)

    auth = private_rest.BitpinApiKeyAuth("key", "secret")
    balances = private_rest.get_balances(auth)
    order = private_rest.place_bulk_orders(
        [
            {
                "symbol": "USDT_IRT",
                "base_amount": "2",
                "price": "100000",
                "side": "buy",
                "order_type": "limit",
                "client_order_id": "cid",
            }
        ],
        auth,
    )

    assert balances["USDT"]["free"] == 1.5
    assert order["orders"][0]["id"] == 42
    assert captured[0]["json"] == {"api_key": "key", "api_secret": "secret"}
    assert captured[1]["headers"]["Authorization"] == "Bearer access-token"
    assert captured[-1]["url"].endswith("/odr/orders/bulk")
    assert captured[-1]["json"] == [{
        "symbol": "USDT_IRT",
        "base_amount": "2",
        "price": "100000",
        "side": "buy",
        "type": "limit",
        "identifier": "cid",
    }]


def test_bitpin_websocket_info(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        ok = True
        content = b"{}"
        text = ""

        def json(self):
            return {"user_identifier": "V33EQ5LV", "ws_token": "jwt-token"}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(private_rest.requests, "post", fake_post)

    info = private_rest.get_websocket_info(private_rest.BitpinApiKeyAuth("key", "secret"))

    assert info == private_rest.BitpinWebSocketInfo("V33EQ5LV", "jwt-token")
    assert captured["url"].endswith("/api/v1/usr/ws-info/")
    assert captured["json"] == {"api_key": "key", "secret_key": "secret"}


def test_bitpin_bulk_cancel_uses_ids_payload(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        ok = True
        content = b"{}"
        text = ""

        def json(self):
            return {"canceled_orders": ["1", "2"], "not_canceled_orders": []}

    def fake_delete(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(private_rest.requests, "delete", fake_delete)

    out = private_rest.cancel_bulk_orders(["1", "2"], private_rest.BitpinTokenAuth("access"))

    assert out["canceled_orders"] == ["1", "2"]
    assert captured["url"].endswith("/api/v1/odr/orders/bulk/")
    assert captured["json"] == {"ids": ["1", "2"]}
    assert captured["headers"]["Authorization"] == "Bearer access"


def test_bitpin_list_filled_orders_uses_documented_fills_endpoint(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        ok = True
        content = b"{}"
        text = ""

        def json(self):
            return [
                {
                    "id": 65375307,
                    "symbol": "USDT_IRT",
                    "base_amount": "63.0",
                    "quote_amount": "123417",
                    "price": "1959",
                    "created_at": "2024-07-29T14:07:34.534824+03:30",
                    "commission": "0",
                    "side": "sell",
                    "commission_currency": "IRT",
                    "order_id": 535179385,
                    "identifier": "cid",
                }
            ]

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(private_rest.requests, "get", fake_get)

    rows = private_rest.list_filled_orders("USDT_IRT", private_rest.BitpinTokenAuth("access"))

    assert rows[0]["order_id"] == 535179385
    assert captured["url"].endswith("/api/v1/odr/fills/")
    assert captured["params"] == {"symbol": "USDT_IRT", "limit": 100}
    assert captured["headers"]["Authorization"] == "Bearer access"


@pytest.mark.asyncio
async def test_bitpin_order_adapter_uses_bulk_place_and_chunks(monkeypatch):
    calls: list[list[dict]] = []

    def fake_place_bulk_orders(orders, auth):
        calls.append(list(orders))
        offset = sum(len(call) for call in calls[:-1])
        return {
            "orders": [
                {"id": offset + idx + 1, "state": "active"}
                for idx, _ in enumerate(orders)
            ]
        }

    monkeypatch.setattr(private_rest, "place_bulk_orders", fake_place_bulk_orders)
    adapter = BitpinRestOrderCommandAdapter("USDT_IRT", api_token=private_rest.BitpinTokenAuth("access"))
    orders = [
        Order(exchange="bitpin", symbol="USDT_IRT", side="buy", price=100000 + i, qty=1 + i)
        for i in range(11)
    ]

    ids = await adapter.place_orders(orders)

    assert ids == [str(i) for i in range(1, 12)]
    assert [len(call) for call in calls] == [10, 1]
    assert calls[0][0] == {
        "symbol": "USDT_IRT",
        "base_amount": "1",
        "price": "100000",
        "side": "buy",
        "order_type": "limit",
        "client_order_id": None,
    }


@pytest.mark.asyncio
async def test_bitpin_order_adapter_bulk_cancel_raises_on_not_canceled(monkeypatch):
    def fake_cancel_bulk_orders(order_ids, auth, *, identifiers=None):
        return {"canceled_orders": ["1"], "not_canceled_orders": ["2"]}

    monkeypatch.setattr(private_rest, "cancel_bulk_orders", fake_cancel_bulk_orders)
    adapter = BitpinRestOrderCommandAdapter("USDT_IRT", api_token=private_rest.BitpinTokenAuth("access"))

    with pytest.raises(RuntimeError, match="cancel bulk orders failed"):
        await adapter.cancel_orders(["1", "2"])


def test_bitpin_open_order_normalization():
    adapter = BitpinRestPrivateStateAdapter("USDT_IRT", api_token=private_rest.BitpinTokenAuth("access"))

    open_order = adapter.normalize_open_order(
        {
            "id": 10,
            "type": "sell",
            "price_limit": "100100",
            "amount1": "3",
            "exchanged1": "1",
            "remain_amount": "2",
            "state": "active",
            "identifier": "cid",
        },
        timestamp_ms=123,
    )

    assert open_order is not None
    assert open_order.exchange_order_id == "10"
    assert open_order.order.side == "sell"
    assert open_order.order.price == 100100.0
    assert open_order.remaining_qty == 2.0
    assert open_order.filled_qty == 1.0


def test_bitpin_fill_normalization_uses_order_id_and_base_amount():
    adapter = BitpinRestPrivateStateAdapter("PIXFI_IRT", api_token=private_rest.BitpinTokenAuth("access"))

    fill = adapter.normalize_fill(
        {
            "id": 65375307,
            "symbol": "PIXFI_IRT",
            "base_amount": "63.0",
            "quote_amount": "123417",
            "price": "1959",
            "created_at": "2024-07-29T14:07:34.534824+03:30",
            "commission": "10",
            "side": "sell",
            "commission_currency": "IRT",
            "order_id": 535179385,
            "identifier": "cid",
        },
        timestamp_ms=123,
    )

    assert fill is not None
    assert fill.exchange_order_id == "535179385"
    assert fill.actor_id == "cid"
    assert fill.side == "sell"
    assert fill.price == pytest.approx(1959.0)
    assert fill.qty == pytest.approx(63.0)
    assert fill.fee == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_bitpin_private_websocket_user_events():
    adapter = BitpinWebSocketPrivateStateAdapter(
        "PAXG_IRT",
        api_token=private_rest.BitpinApiKeyAuth("key", "secret"),
    )

    await adapter._handle_user_message(
        {
            "push": {
                "pub": {
                    "data": {
                        "id": 123,
                        "symbol": "PAXG_IRT",
                        "type": "market",
                        "side": "sell",
                        "price": "464414599",
                        "base_amount": "0.0012",
                        "dealed_base_amount": "0.0004",
                        "identifier": "cid",
                        "state": "active",
                        "event": "user_order_update",
                        "event_time": "2025-10-19T12:03:47.620634Z",
                    }
                }
            }
        }
    )
    await adapter._handle_user_message(
        {
            "push": {
                "pub": {
                    "data": {
                        "id": 1234,
                        "symbol": "PAXG_IRT",
                        "base_amount": "0.0004",
                        "price": "464314617",
                        "commission": "0",
                        "side": "sell",
                        "order_id": 123,
                        "identifier": "cid",
                        "event": "user_match_update",
                        "event_time": "2025-10-19T12:03:47.651296Z",
                    }
                }
            }
        }
    )

    update = await adapter._order_update_queue.get()
    fill = await adapter._fill_queue.get()

    assert update.exchange_order_id == "123"
    assert update.remaining_qty == pytest.approx(0.0008)
    assert update.filled_qty == pytest.approx(0.0004)
    assert fill.exchange_order_id == "123"
    assert fill.qty == pytest.approx(0.0004)
    assert fill.price == 464314617.0
