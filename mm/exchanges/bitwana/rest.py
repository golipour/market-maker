from __future__ import annotations

import os
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


class BitwanaRestMarketDataAdapter(RestPollingMarketDataAdapter):
    BASE_URL = os.environ.get("BITWANA_API_BASE_URL", "https://bit.yshm.ir").rstrip("/")

    def __init__(
        self,
        symbol: str,
        *,
        trades_limit: int = 100,
        depth_limit: int = 50,
        poll_interval_ms: int = 1000,
    ):
        super().__init__("bitwana", symbol, poll_interval_ms=poll_interval_ms)
        self.trades_limit = max(1, int(trades_limit))
        self.depth_limit = max(1, int(depth_limit))
        self._client = RestClient(self.BASE_URL)
        self._wire = build_instrument_wire("bitwana", symbol)

    def _market(self) -> str:
        return self._wire.api_market_symbol

    async def fetch_trades(self) -> list[RawTrade]:
        payload = await self._client.get_json(f"/api/1/trades/recent/market/{self._market()}")
        rows = _content_rows(payload)[: self.trades_limit]
        trades = [
            self._wire.scale_trade_from_api(self._trade_from_item(item))
            for item in rows
            if isinstance(item, dict)
        ]
        trades.sort(key=lambda trade: trade.timestamp_ms)
        return trades

    async def fetch_orderbook(self) -> RawOrderbook:
        payload = await self._client.get_json(
            f"/api/1/orders/orderBook/market/{self._market()}/level2"
        )
        result = _content_dict(payload)
        book = RawOrderbook(
            exchange=self.exchange,
            symbol=self.symbol,
            timestamp_ms=now_ms(),
            bids=rows_to_levels(
                _level_rows(result.get("bids")),
                limit=self.depth_limit,
                reverse=True,
            ),
            asks=rows_to_levels(_level_rows(result.get("asks")), limit=self.depth_limit),
            is_snapshot=True,
        )
        return self._wire.scale_orderbook_from_api(book)

    def _trade_from_item(self, item: dict[str, Any]) -> RawTrade:
        taker_side = str(item.get("takerSide") or "").upper()
        side = "sell" if taker_side.startswith("S") else "buy"
        timestamp = item.get("time", item.get("created", item.get("updated")))
        return RawTrade(
            trade_id=str(item.get("id")) if item.get("id") is not None else None,
            timestamp_ms=parse_timestamp_ms(timestamp),
            price=float(item.get("unitPrice", 0) or 0),
            qty=float(item.get("amount", 0) or 0),
            side=side,
            exchange=self.exchange,
            symbol=self.symbol,
        )


def _content_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("content"), dict):
        return payload["content"]
    return payload if isinstance(payload, dict) else {}


def _content_rows(payload: Any) -> list[Any]:
    if isinstance(payload, dict) and isinstance(payload.get("content"), list):
        return payload["content"]
    if isinstance(payload, list):
        return payload
    return []


def _level_rows(rows: Any) -> list[list[str]]:
    out: list[list[str]] = []
    if not isinstance(rows, list):
        return out
    for item in rows:
        if isinstance(item, dict):
            out.append([str(item.get("price", "0")), str(item.get("size", item.get("quantity", "0")))])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append([str(item[0]), str(item[1])])
    return out
