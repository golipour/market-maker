from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm.exchanges.bitwana.rest import BitwanaRestMarketDataAdapter
from mm.exchanges.bitpin.rest import BitpinRestMarketDataAdapter
from mm.exchanges.bitpin.ws import BitpinWsMarketDataAdapter
from mm.exchanges.nobitex.rest import NobitexRestMarketDataAdapter
from mm.exchanges.nobitex.ws import NobitexWsMarketDataAdapter
from mm.exchanges.tabdeal.rest import TabdealRestMarketDataAdapter
from mm.exchanges.tabdeal.ws import TabdealWsMarketDataAdapter
from mm.exchanges.wallex.rest import WallexRestMarketDataAdapter
from mm.exchanges.wallex.ws import WallexWsMarketDataAdapter

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "exchange_payloads"


class FakeRestClient:
    def __init__(self, payloads: dict[str, object]):
        self.payloads = payloads
        self.calls: list[tuple[str, dict | None]] = []

    async def get_json(self, path: str, params: dict | None = None):
        self.calls.append((path, params))
        return self.payloads[path]


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.asyncio
async def test_nobitex_rest_payloads_normalize_to_raw_events():
    adapter = NobitexRestMarketDataAdapter("BTC_IRT", depth_limit=2, trades_limit=2)
    adapter._client = FakeRestClient(
        {
            "/v3/orderbook/BTCIRT": load_fixture("nobitex_rest_orderbook.json"),
            "/v2/trades/BTCIRT": load_fixture("nobitex_rest_trades.json"),
        }
    )

    book = await adapter.fetch_orderbook()
    trades = await adapter.fetch_trades()

    assert book.exchange == "nobitex"
    assert book.symbol == "BTC_IRT"
    assert [level.price for level in book.bids] == [100000.0, 99900.0]
    assert [level.qty for level in book.asks] == [0.4, 2.0]
    assert [trade.trade_id for trade in trades] == ["n1", "n2"]
    assert [trade.side for trade in trades] == ["sell", "buy"]
    assert trades[0].price == 100000.0
    assert trades[1].price == 100100.0


@pytest.mark.asyncio
async def test_wallex_rest_payloads_normalize_to_raw_events():
    adapter = WallexRestMarketDataAdapter("BTC_IRT", depth_limit=2, trades_limit=2)
    adapter._client = FakeRestClient(
        {
            "/v1/depth": load_fixture("wallex_rest_orderbook.json"),
            "/v1/trades": load_fixture("wallex_rest_trades.json"),
        }
    )

    book = await adapter.fetch_orderbook()
    trades = await adapter.fetch_trades()

    assert book.exchange == "wallex"
    assert [level.price for level in book.bids] == [1000000.0, 999000.0]
    assert [level.qty for level in book.asks] == [0.4, 2.0]
    assert [trade.trade_id for trade in trades] == ["w1", "w2"]
    assert [trade.side for trade in trades] == ["sell", "buy"]
    assert adapter._client.calls[0] == ("/v1/depth", {"symbol": "BTCTMN"})


@pytest.mark.asyncio
async def test_tabdeal_rest_payloads_normalize_to_raw_events():
    adapter = TabdealRestMarketDataAdapter("USDT_IRT", depth_limit=2, trades_limit=2)
    adapter._client = FakeRestClient(
        {
            "/r/api/v1/depth": load_fixture("tabdeal_rest_orderbook.json"),
            "/r/api/v1/trades": load_fixture("tabdeal_rest_trades.json"),
        }
    )

    book = await adapter.fetch_orderbook()
    trades = await adapter.fetch_trades()

    assert book.exchange == "tabdeal"
    assert book.symbol == "USDT_IRT"
    assert [level.price for level in book.bids] == [100000.0, 99900.0]
    assert [level.qty for level in book.asks] == [0.4, 2.5]
    assert [trade.trade_id for trade in trades] == ["10", "11"]
    assert [trade.side for trade in trades] == ["sell", "buy"]
    assert adapter._client.calls[0] == ("/r/api/v1/depth", {"symbol": "USDTIRT", "limit": 2})


@pytest.mark.asyncio
async def test_bitpin_rest_payloads_normalize_to_raw_events():
    adapter = BitpinRestMarketDataAdapter("USDT_IRT", depth_limit=2, trades_limit=2)
    adapter._client = FakeRestClient(
        {
            "/v1/mkt/markets/": load_fixture("bitpin_markets.json"),
            "/v2/mth/actives/5/": load_fixture("bitpin_rest_sell_orderbook.json"),
            "/v1/mth/matches/5/": load_fixture("bitpin_rest_trades.json"),
        }
    )

    async def fake_get_json(path: str, params: dict | None = None):
        adapter._client.calls.append((path, params))
        if path == "/v2/mth/actives/5/" and params == {"type": "buy"}:
            return load_fixture("bitpin_rest_buy_orderbook.json")
        return adapter._client.payloads[path]

    adapter._client.get_json = fake_get_json

    book = await adapter.fetch_orderbook()
    trades = await adapter.fetch_trades()

    assert book.exchange == "bitpin"
    assert book.symbol == "USDT_IRT"
    assert [level.price for level in book.bids] == [100000.0, 99900.0]
    assert [level.qty for level in book.asks] == [0.4, 2.5]
    assert [trade.trade_id for trade in trades] == ["10", "11"]
    assert [trade.side for trade in trades] == ["sell", "buy"]
    assert adapter._client.calls[0] == ("/v1/mkt/markets/", {"page": 1})


@pytest.mark.asyncio
async def test_bitwana_rest_payloads_normalize_to_raw_events():
    adapter = BitwanaRestMarketDataAdapter("USDT_IRT", depth_limit=2, trades_limit=2)
    adapter._client = FakeRestClient(
        {
            "/api/1/orders/orderBook/market/USDT_IRT/level2": load_fixture(
                "bitwana_rest_orderbook.json"
            ),
            "/api/1/trades/recent/market/USDT_IRT": load_fixture("bitwana_rest_trades.json"),
        }
    )

    book = await adapter.fetch_orderbook()
    trades = await adapter.fetch_trades()

    assert book.exchange == "bitwana"
    assert book.symbol == "USDT_IRT"
    assert [level.price for level in book.bids] == [186692.0, 186600.0]
    assert [level.qty for level in book.asks] == [21.49, 4.0]
    assert [trade.side for trade in trades] == ["sell", "buy"]
    assert trades[0].price == 186681.0
    assert trades[0].timestamp_ms == 1787047216000


@pytest.mark.asyncio
async def test_nobitex_websocket_payloads_normalize_to_raw_events():
    adapter = NobitexWsMarketDataAdapter("BTC_IRT", depth_limit=2)

    await adapter.handle_message(load_fixture("nobitex_ws_orderbook.json"))
    await adapter.handle_message(load_fixture("nobitex_ws_trade.json"))

    book = await adapter._book_queue.get()
    trade = await adapter._trade_queue.get()

    assert book.bids[0].price == 100000.0
    assert book.asks[0].qty == 0.4
    assert trade.trade_id == "nw1"
    assert trade.side == "buy"
    assert trade.price == 100100.0


@pytest.mark.asyncio
async def test_wallex_websocket_payloads_normalize_to_raw_events():
    adapter = WallexWsMarketDataAdapter("BTC_IRT", depth_limit=2)

    await adapter.handle_message(load_fixture("wallex_ws_buy_depth.json"))
    assert adapter._book_queue.empty()
    await adapter.handle_message(load_fixture("wallex_ws_sell_depth.json"))
    await adapter.handle_message(load_fixture("wallex_ws_trade.json"))

    book = await adapter._book_queue.get()
    trade = await adapter._trade_queue.get()

    assert book.bids[0].price == 1000000.0
    assert book.asks[0].price == 1001000.0
    assert trade.trade_id == "ww1"
    assert trade.timestamp_ms == 1710000002000
    assert trade.side == "buy"


@pytest.mark.asyncio
async def test_tabdeal_websocket_payloads_normalize_to_raw_events():
    adapter = TabdealWsMarketDataAdapter("USDT_IRT", depth_limit=2)

    await adapter.handle_message(load_fixture("tabdeal_ws_orderbook.json"))
    await adapter.handle_message(load_fixture("tabdeal_ws_trade.json"))

    book = await adapter._book_queue.get()
    trade = await adapter._trade_queue.get()

    assert book.bids[0].price == 100000.0
    assert book.asks[0].qty == 0.4
    assert trade.trade_id == "12"
    assert trade.timestamp_ms == 1710000003000
    assert trade.side == "buy"


@pytest.mark.asyncio
async def test_bitpin_websocket_payloads_normalize_to_raw_events():
    adapter = BitpinWsMarketDataAdapter("USDT_IRT", depth_limit=2)

    await adapter.handle_message(load_fixture("bitpin_ws_orderbook.json"))
    await adapter.handle_message(load_fixture("bitpin_ws_trade.json"))

    book = await adapter._book_queue.get()
    trade = await adapter._trade_queue.get()

    assert book.bids[0].price == 100000.0
    assert book.asks[0].qty == 0.4
    assert trade.trade_id == "73802972_73807087"
    assert trade.price == 100000.0
    assert trade.qty == 0.5
    assert trade.side == "sell"
