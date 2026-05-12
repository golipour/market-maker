from __future__ import annotations

from typing import Any

from mm.core.models.market import RawOrderbook, RawTrade
from mm.exchanges.nobitex.quote_unit import scale_nobitex_orderbook_quotes, scale_nobitex_trade_quote
from mm.exchanges.shared.provider_adapter import (
    RestClient,
    RestPollingMarketDataAdapter,
    now_ms,
    parse_timestamp_ms,
    rows_to_levels,
)


class NobitexRestMarketDataAdapter(RestPollingMarketDataAdapter):
    BASE_URL = "https://apiv2.nobitex.ir"

    def __init__(
        self,
        symbol: str,
        *,
        trades_limit: int = 100,
        depth_limit: int = 50,
        poll_interval_ms: int = 1000,
    ):
        super().__init__("nobitex", symbol, poll_interval_ms=poll_interval_ms)
        self.trades_limit = max(1, int(trades_limit))
        self.depth_limit = max(1, int(depth_limit))
        self._client = RestClient(self.BASE_URL)

    async def fetch_trades(self) -> list[RawTrade]:
        payload = await self._client.get_json(f"/v2/trades/{self._api_symbol()}")
        if payload.get("status") != "ok":
            return []
        trades = []
        for item in (payload.get("trades") or [])[: self.trades_limit]:
            trades.append(scale_nobitex_trade_quote(self.symbol, self._trade_from_item(item)))
        trades.sort(key=lambda trade: trade.timestamp_ms)
        return trades

    async def fetch_orderbook(self) -> RawOrderbook:
        payload = await self._client.get_json(f"/v3/orderbook/{self._api_symbol()}")
        if payload.get("status") != "ok":
            raise ValueError(payload.get("message", "nobitex orderbook error"))
        book = RawOrderbook(
            exchange=self.exchange,
            symbol=self.symbol,
            timestamp_ms=now_ms(),
            bids=rows_to_levels(payload.get("bids"), limit=self.depth_limit, reverse=True),
            asks=rows_to_levels(payload.get("asks"), limit=self.depth_limit),
            is_snapshot=True,
        )
        return scale_nobitex_orderbook_quotes(self.symbol, book)

    def _api_symbol(self) -> str:
        return self.symbol.replace("_", "").upper()

    def _trade_from_item(self, item: dict[str, Any]) -> RawTrade:
        taker_side = str(item.get("type", "")).lower()
        return RawTrade(
            trade_id=str(item.get("id")) if item.get("id") is not None else None,
            timestamp_ms=parse_timestamp_ms(item.get("time")),
            price=float(item.get("price", 0) or 0),
            qty=float(item.get("volume", 0) or 0),
            side="buy" if taker_side == "buy" else "sell",
            exchange=self.exchange,
            symbol=self.symbol,
        )
