from __future__ import annotations

from typing import Any

from mm.core.models.market import RawOrderbook, RawTrade
from mm.exchanges.instrument_wire import build_instrument_wire
from mm.exchanges.kucoin.private_rest import api_symbol
from mm.exchanges.shared.provider_adapter import (
    RestClient,
    RestPollingMarketDataAdapter,
    now_ms,
    parse_timestamp_ms,
    rows_to_levels,
)


def _unwrap_kucoin(payload: Any, *, path: str) -> Any:
    if not isinstance(payload, dict):
        raise ValueError(f"kucoin {path}: bad response")
    if str(payload.get("code", "")) != "200000":
        raise ValueError(payload.get("msg", "KuCoin API error"))
    return payload.get("data")


def _depth_tier(depth_limit: int) -> int:
    return 100 if int(depth_limit) > 20 else 20


class KucoinRestMarketDataAdapter(RestPollingMarketDataAdapter):
    BASE_URL = "https://api.kucoin.com"

    def __init__(
        self,
        symbol: str,
        *,
        trades_limit: int = 100,
        depth_limit: int = 50,
        poll_interval_ms: int = 1000,
    ):
        super().__init__("kucoin", symbol, poll_interval_ms=poll_interval_ms)
        self.trades_limit = max(1, int(trades_limit))
        self.depth_limit = max(1, int(depth_limit))
        self._client = RestClient(self.BASE_URL)
        self._wire = build_instrument_wire("kucoin", symbol)

    async def fetch_trades(self) -> list[RawTrade]:
        payload = await self._client.get_json(
            "/api/v1/market/histories",
            {"symbol": api_symbol(self.symbol)},
        )
        data = _unwrap_kucoin(payload, path="/api/v1/market/histories")
        rows = data if isinstance(data, list) else []
        trades = [
            self._wire.scale_trade_from_api(self._trade_from_item(item))
            for item in rows[: self.trades_limit]
            if isinstance(item, dict)
        ]
        trades.sort(key=lambda trade: trade.timestamp_ms)
        return trades

    async def fetch_orderbook(self) -> RawOrderbook:
        tier = _depth_tier(self.depth_limit)
        payload = await self._client.get_json(
            f"/api/v1/market/orderbook/level2_{tier}",
            {"symbol": api_symbol(self.symbol)},
        )
        data = _unwrap_kucoin(payload, path=f"/api/v1/market/orderbook/level2_{tier}")
        if not isinstance(data, dict):
            raise ValueError("kucoin orderbook error: bad response")
        book = RawOrderbook(
            exchange=self.exchange,
            symbol=self.symbol,
            timestamp_ms=parse_timestamp_ms(data.get("time") or now_ms()),
            bids=rows_to_levels(data.get("bids"), limit=self.depth_limit, reverse=True),
            asks=rows_to_levels(data.get("asks"), limit=self.depth_limit),
            is_snapshot=True,
        )
        return self._wire.scale_orderbook_from_api(book)

    def _trade_from_item(self, item: dict[str, Any]) -> RawTrade:
        side = str(item.get("side", "")).lower()
        return RawTrade(
            trade_id=str(item.get("sequence") or item.get("tradeId") or "") or None,
            timestamp_ms=parse_timestamp_ms(item.get("time")),
            price=float(item.get("price", 0) or 0),
            qty=float(item.get("size", 0) or 0),
            side="sell" if side == "sell" else "buy",
            exchange=self.exchange,
            symbol=self.symbol,
        )
