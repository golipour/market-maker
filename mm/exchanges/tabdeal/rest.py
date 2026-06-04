from __future__ import annotations

from typing import Any

from mm.core.models.market import RawOrderbook, RawTrade
from mm.exchanges.instrument_wire import build_instrument_wire
from mm.exchanges.shared.provider_adapter import (
    RestClient,
    RestPollingMarketDataAdapter,
    now_ms,
    parse_timestamp_ms,
    rows_to_levels,
)


class TabdealRestMarketDataAdapter(RestPollingMarketDataAdapter):
    BASE_URL = "https://api1.tabdeal.org"

    def __init__(
        self,
        symbol: str,
        *,
        trades_limit: int = 100,
        depth_limit: int = 50,
        poll_interval_ms: int = 1000,
    ):
        super().__init__("tabdeal", symbol, poll_interval_ms=poll_interval_ms)
        self.trades_limit = max(1, int(trades_limit))
        self.depth_limit = max(1, int(depth_limit))
        self._client = RestClient(self.BASE_URL)
        self._wire = build_instrument_wire("tabdeal", symbol)

    async def fetch_trades(self) -> list[RawTrade]:
        payload = await self._client.get_json(
            "/r/api/v1/trades",
            {"symbol": self._api_symbol(), "limit": self.trades_limit},
        )
        rows = payload if isinstance(payload, list) else payload.get("trades", []) if isinstance(payload, dict) else []
        trades = [self._wire.scale_trade_from_api(self._trade_from_item(item)) for item in rows[: self.trades_limit] if isinstance(item, dict)]
        trades.sort(key=lambda trade: trade.timestamp_ms)
        return trades

    async def fetch_orderbook(self) -> RawOrderbook:
        payload = await self._client.get_json(
            "/r/api/v1/depth",
            {"symbol": self._api_symbol(), "limit": self.depth_limit},
        )
        if not isinstance(payload, dict):
            raise ValueError("tabdeal orderbook error: bad response")
        book = RawOrderbook(
            exchange=self.exchange,
            symbol=self.symbol,
            timestamp_ms=parse_timestamp_ms(payload.get("lastUpdateId") or payload.get("E") or now_ms()),
            bids=rows_to_levels(payload.get("bids"), limit=self.depth_limit, reverse=True),
            asks=rows_to_levels(payload.get("asks"), limit=self.depth_limit),
            is_snapshot=True,
        )
        return self._wire.scale_orderbook_from_api(book)

    def _api_symbol(self) -> str:
        return self._wire.api_market_symbol

    def _trade_from_item(self, item: dict[str, Any]) -> RawTrade:
        is_buyer_maker = bool(item.get("isBuyerMaker", item.get("m", False)))
        return RawTrade(
            trade_id=str(item.get("id") or item.get("tradeId")) if (item.get("id") or item.get("tradeId")) is not None else None,
            timestamp_ms=parse_timestamp_ms(item.get("time") or item.get("T")),
            price=float(item.get("price", item.get("p", 0)) or 0),
            qty=float(item.get("qty", item.get("q", 0)) or 0),
            side="sell" if is_buyer_maker else "buy",
            exchange=self.exchange,
            symbol=self.symbol,
        )
