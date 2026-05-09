from __future__ import annotations

from typing import Any

from mm.core.models.market import RawOrderbook, RawTrade
from mm.exchanges.shared.provider_adapter import (
    RestClient,
    RestPollingMarketDataAdapter,
    now_ms,
    parse_timestamp_ms,
    rows_to_levels,
)


class WallexRestMarketDataAdapter(RestPollingMarketDataAdapter):
    BASE_URL = "https://api.wallex.ir"

    def __init__(
        self,
        symbol: str,
        *,
        trades_limit: int = 100,
        depth_limit: int = 50,
        poll_interval_ms: int = 1000,
    ):
        super().__init__("wallex", symbol, poll_interval_ms=poll_interval_ms)
        self.trades_limit = max(1, int(trades_limit))
        self.depth_limit = max(1, int(depth_limit))
        self._client = RestClient(self.BASE_URL)

    async def fetch_trades(self) -> list[RawTrade]:
        payload = await self._client.get_json("/v1/trades", {"symbol": self._api_symbol()})
        if not payload.get("success"):
            return []
        result = payload.get("result") or {}
        trades = [
            self._trade_from_item(item)
            for item in (result.get("latestTrades") or [])[: self.trades_limit]
        ]
        trades.sort(key=lambda trade: trade.timestamp_ms)
        return trades

    async def fetch_orderbook(self) -> RawOrderbook:
        payload = await self._client.get_json("/v1/depth", {"symbol": self._api_symbol()})
        if not payload.get("success"):
            raise ValueError(payload.get("message", "wallex orderbook error"))
        result = payload.get("result") or {}
        return RawOrderbook(
            exchange=self.exchange,
            symbol=self.symbol,
            timestamp_ms=now_ms(),
            bids=rows_to_levels(self._depth_rows(result.get("bid")), limit=self.depth_limit, reverse=True),
            asks=rows_to_levels(self._depth_rows(result.get("ask")), limit=self.depth_limit),
            is_snapshot=True,
        )

    def _api_symbol(self) -> str:
        symbol = self.symbol.replace("_", "").upper()
        if symbol.endswith("IRT"):
            return f"{symbol[:-3]}TMN"
        return symbol

    @staticmethod
    def _depth_rows(rows: Any) -> list[list[str]]:
        out = []
        if isinstance(rows, list):
            for item in rows:
                if isinstance(item, dict):
                    out.append([str(item.get("price", "0")), str(item.get("quantity", "0"))])
        return out

    def _trade_from_item(self, item: dict[str, Any]) -> RawTrade:
        is_taker_buy = bool(item.get("isBuyOrder", False))
        return RawTrade(
            trade_id=str(item.get("id")) if item.get("id") is not None else None,
            timestamp_ms=parse_timestamp_ms(item.get("timestamp")),
            price=float(item.get("price", 0) or 0),
            qty=float(item.get("quantity", 0) or 0),
            side="buy" if is_taker_buy else "sell",
            exchange=self.exchange,
            symbol=self.symbol,
        )

