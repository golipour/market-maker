from __future__ import annotations

import json
from typing import Any

from mm.core.models.market import RawOrderbook, RawTrade
from mm.exchanges.instrument_wire import build_instrument_wire
from mm.exchanges.shared.provider_adapter import (
    WebSocketMarketDataAdapter,
    parse_timestamp_ms,
    rows_to_levels,
)


class TabdealWsMarketDataAdapter(WebSocketMarketDataAdapter):
    def __init__(self, symbol: str, *, depth_limit: int = 50):
        super().__init__("tabdeal", symbol)
        self.depth_limit = max(1, int(depth_limit))
        self._wire = build_instrument_wire("tabdeal", symbol)

    def websocket_url(self) -> str:
        return "wss://api1.tabdeal.org/stream/"

    async def subscribe(self, ws: Any) -> None:
        symbol = self._wire.api_market_symbol.lower()
        await ws.send(json.dumps({"method": "SUBSCRIBE", "params": [f"{symbol}@depth@2000ms"], "id": 1}))
        await ws.send(json.dumps({"method": "SUBSCRIBE", "params": [f"{symbol}@trade"], "id": 2}))

    async def handle_message(self, payload: Any) -> None:
        if not isinstance(payload, dict) or "result" in payload:
            return
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        if not isinstance(data, dict):
            return
        event_type = str(data.get("e") or "")
        stream = str(payload.get("stream") or "")
        if event_type == "depthUpdate" or "@depth" in stream:
            book = RawOrderbook(
                exchange=self.exchange,
                symbol=self.symbol,
                timestamp_ms=parse_timestamp_ms(data.get("E")),
                bids=rows_to_levels(data.get("b"), limit=self.depth_limit, reverse=True),
                asks=rows_to_levels(data.get("a"), limit=self.depth_limit),
                is_snapshot=True,
            )
            await self.emit_orderbook(self._wire.scale_orderbook_from_api(book))
        elif event_type in {"trade", "aggTrade"} or stream.endswith("@trade"):
            trade = RawTrade(
                trade_id=str(data.get("t") or data.get("a")) if (data.get("t") or data.get("a")) is not None else None,
                timestamp_ms=parse_timestamp_ms(data.get("T") or data.get("E")),
                price=float(data.get("p", 0) or 0),
                qty=float(data.get("q", 0) or 0),
                side="sell" if bool(data.get("m", False)) else "buy",
                exchange=self.exchange,
                symbol=self.symbol,
            )
            await self.emit_trade(self._wire.scale_trade_from_api(trade))
