from __future__ import annotations

import json
from typing import Any

from mm.core.models.market import RawOrderbook, RawTrade
from mm.exchanges.shared.provider_adapter import (
    WebSocketMarketDataAdapter,
    now_ms,
    parse_timestamp_ms,
    rows_to_levels,
    update_depth_side,
)


class WallexWsMarketDataAdapter(WebSocketMarketDataAdapter):
    def __init__(self, symbol: str, *, depth_limit: int = 50):
        super().__init__("wallex", symbol)
        self.depth_limit = max(1, int(depth_limit))
        self._buy_depth: list[list[str]] = []
        self._sell_depth: list[list[str]] = []

    def websocket_url(self) -> str:
        return "wss://api.wallex.ir/ws"

    async def subscribe(self, ws: Any) -> None:
        market = self._api_symbol()
        await ws.send(json.dumps(["subscribe", {"channel": f"{market}@buyDepth"}]))
        await ws.send(json.dumps(["subscribe", {"channel": f"{market}@sellDepth"}]))
        await ws.send(json.dumps(["subscribe", {"channel": f"{market}@trade"}]))

    async def handle_message(self, payload: Any) -> None:
        if not isinstance(payload, list) or len(payload) < 2:
            return
        channel = str(payload[0])
        data = payload[1]
        if channel.endswith("@trade") and isinstance(data, dict):
            await self.emit_trade(
                RawTrade(
                    trade_id=str(data.get("id")) if data.get("id") is not None else None,
                    timestamp_ms=parse_timestamp_ms(data.get("timestamp")),
                    price=float(data.get("price", 0) or 0),
                    qty=float(data.get("quantity", 0) or 0),
                    side="buy" if bool(data.get("isBuyOrder", False)) else "sell",
                    exchange=self.exchange,
                    symbol=self.symbol,
                )
            )
            return
        if channel.endswith("@buyDepth"):
            self._buy_depth = update_depth_side(
                self._buy_depth,
                data,
                descending=True,
                limit=self.depth_limit,
                row_parser=self._wallex_depth_row,
            )
        elif channel.endswith("@sellDepth"):
            self._sell_depth = update_depth_side(
                self._sell_depth,
                data,
                descending=False,
                limit=self.depth_limit,
                row_parser=self._wallex_depth_row,
            )
        else:
            return
        if self._buy_depth and self._sell_depth:
            await self.emit_orderbook(
                RawOrderbook(
                    exchange=self.exchange,
                    symbol=self.symbol,
                    timestamp_ms=now_ms(),
                    bids=rows_to_levels(self._buy_depth, limit=self.depth_limit, reverse=True),
                    asks=rows_to_levels(self._sell_depth, limit=self.depth_limit),
                    is_snapshot=True,
                )
            )

    def _api_symbol(self) -> str:
        symbol = self.symbol.replace("_", "").upper()
        if symbol.endswith("IRT"):
            return f"{symbol[:-3]}TMN"
        return symbol

    @staticmethod
    def _wallex_depth_row(row: Any) -> tuple[str, str] | None:
        if not isinstance(row, dict):
            return None
        price = row.get("price")
        qty = row.get("quantity")
        if price is None or qty is None:
            return None
        return str(price), str(qty)

