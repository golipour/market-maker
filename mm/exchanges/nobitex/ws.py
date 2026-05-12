from __future__ import annotations

import json
from typing import Any

from mm.core.models.market import RawOrderbook, RawTrade
from mm.exchanges.nobitex.quote_unit import scale_nobitex_orderbook_quotes, scale_nobitex_trade_quote
from mm.exchanges.shared.provider_adapter import (
    WebSocketMarketDataAdapter,
    parse_timestamp_ms,
    rows_to_levels,
)


class NobitexWsMarketDataAdapter(WebSocketMarketDataAdapter):
    def __init__(self, symbol: str, *, depth_limit: int = 50):
        super().__init__("nobitex", symbol)
        self.depth_limit = max(1, int(depth_limit))

    def websocket_url(self) -> str:
        return "wss://ws.nobitex.ir/connection/websocket"

    async def subscribe(self, ws: Any) -> None:
        symbol = self.symbol.replace("_", "").upper()
        await ws.send(json.dumps({"connect": {}, "id": 1}))
        await ws.send(json.dumps({"id": 2, "subscribe": {"channel": f"public:orderbook-{symbol}"}}))
        await ws.send(json.dumps({"id": 3, "subscribe": {"channel": f"public:trades-{symbol}"}}))

    async def handle_message(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        push = payload.get("push")
        if not isinstance(push, dict):
            return
        channel = str(push.get("channel", ""))
        data = self._extract_data(push)
        if not isinstance(data, dict):
            return
        if channel.startswith("public:orderbook-"):
            book = RawOrderbook(
                exchange=self.exchange,
                symbol=self.symbol,
                timestamp_ms=parse_timestamp_ms(data.get("lastUpdate")),
                bids=rows_to_levels(data.get("bids"), limit=self.depth_limit, reverse=True),
                asks=rows_to_levels(data.get("asks"), limit=self.depth_limit),
                is_snapshot=True,
            )
            await self.emit_orderbook(scale_nobitex_orderbook_quotes(self.symbol, book))
        elif channel.startswith("public:trades-"):
            taker_side = str(data.get("type", "")).lower()
            trade = RawTrade(
                trade_id=str(data.get("id")) if data.get("id") is not None else None,
                timestamp_ms=parse_timestamp_ms(data.get("time")),
                price=float(data.get("price", 0) or 0),
                qty=float(data.get("volume", 0) or 0),
                side="buy" if taker_side == "buy" else "sell",
                exchange=self.exchange,
                symbol=self.symbol,
            )
            await self.emit_trade(scale_nobitex_trade_quote(self.symbol, trade))

    @staticmethod
    def _extract_data(push: dict[str, Any]) -> Any:
        pub = push.get("pub") or {}
        data = pub.get("data")
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        if isinstance(data, dict):
            return data
        return push.get("data")

