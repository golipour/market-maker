from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import websockets

from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.core.models.market import RawOrderbook, RawTrade
from mm.exchanges.instrument_wire import build_instrument_wire
from mm.exchanges.kucoin import private_rest
from mm.exchanges.shared.provider_adapter import is_permanent_market_data_error, parse_timestamp_ms, rows_to_levels


def _kucoin_ws_trade_time_ms(d: dict) -> int:
    raw = d.get("E") or d.get("time") or d.get("ts") or d.get("T")
    return parse_timestamp_ms(raw)


def _kucoin_ping_pong_json(payload: dict) -> str | None:
    if payload.get("type") == "ping":
        return json.dumps({"type": "pong", "id": payload.get("id")})
    return None


class KucoinWsMarketDataAdapter(AbstractMarketDataAdapter):
    """KuCoin classic bullet-public websocket with x-push-spot fallback."""

    def __init__(self, symbol: str, *, depth_limit: int = 50):
        self.exchange = "kucoin"
        self.symbol = symbol
        self.depth_limit = max(1, int(depth_limit))
        self._wire = build_instrument_wire("kucoin", symbol)
        self._api_symbol = private_rest.api_symbol(symbol)
        self._connected = False
        self._use_classic = False
        self._trade_queue: asyncio.Queue[RawTrade] = asyncio.Queue()
        self._book_queue: asyncio.Queue[RawOrderbook] = asyncio.Queue()
        self._error_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def connect(self) -> None:
        self._connected = True
        self._task = asyncio.create_task(self._run_forever())

    async def disconnect(self) -> None:
        self._connected = False
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def trade_stream(self) -> AsyncIterator[RawTrade]:
        while self._connected:
            yield await self._trade_queue.get()

    async def orderbook_stream(self) -> AsyncIterator[RawOrderbook]:
        while self._connected:
            yield await self._book_queue.get()

    async def error_stream(self) -> AsyncIterator[dict[str, str]]:
        while self._connected:
            yield await self._error_queue.get()

    async def _run_forever(self) -> None:
        reconnect_s = 1.0
        while self._connected:
            url = await asyncio.to_thread(private_rest.bullet_public_websocket_url)
            if url:
                self._use_classic = True
            else:
                self._use_classic = False
                url = "wss://x-push-spot.kucoin.com"
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    reconnect_s = 1.0
                    await self._subscribe(ws)
                    async for message in ws:
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError as exc:
                            await self._publish_error("websocket_decode", exc)
                            continue
                        pong = _kucoin_ping_pong_json(payload)
                        if pong is not None:
                            await ws.send(pong)
                            continue
                        await self._handle_message(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("websocket", exc)
                if self._connected:
                    if is_permanent_market_data_error(exc):
                        await asyncio.sleep(300.0)
                    else:
                        await asyncio.sleep(reconnect_s)
                        reconnect_s = min(30.0, reconnect_s * 2.0)

    async def _subscribe(self, ws: Any) -> None:
        base = int(asyncio.get_event_loop().time() * 1000)
        if self._use_classic:
            depth_tier = "50" if self.depth_limit >= 50 else "5"
            await ws.send(json.dumps({
                "id": str(base),
                "type": "subscribe",
                "topic": f"/market/match:{self._api_symbol}",
                "privateChannel": False,
                "response": True,
            }))
            await ws.send(json.dumps({
                "id": str(base + 1),
                "type": "subscribe",
                "topic": f"/spotMarket/level2Depth{depth_tier}:{self._api_symbol}",
                "privateChannel": False,
                "response": True,
            }))
            return
        await ws.send(json.dumps({
            "id": str(base),
            "action": "SUBSCRIBE",
            "channel": "obu",
            "tradeType": "SPOT",
            "symbol": self._api_symbol,
            "depth": "50",
        }))
        await ws.send(json.dumps({
            "id": str(base + 1),
            "action": "SUBSCRIBE",
            "channel": "trade",
            "tradeType": "SPOT",
            "symbol": self._api_symbol,
        }))

    async def _handle_message(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        ptype = payload.get("type")
        if ptype in ("ack", "welcome", "receipt", "error"):
            return
        if ptype == "message":
            topic = str(payload.get("topic", ""))
            data = payload.get("data")
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return
            if not isinstance(data, dict):
                return
            if "/market/match:" in topic:
                await self._emit_trade_from_classic(data)
                return
            if "level2Depth" in topic or "/market/level2" in topic or "/spotMarket/level2" in topic:
                await self._emit_book_from_levels(
                    data.get("bids"),
                    data.get("asks"),
                    data.get("timestamp") or data.get("time"),
                )
            return

        t = str(payload.get("T", "") or "")
        d = payload.get("d")
        if not isinstance(d, dict):
            inner = payload.get("data")
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    inner = None
            if isinstance(inner, dict):
                d = inner
        if not isinstance(d, dict):
            return
        subj = str(payload.get("subject", "")).lower()
        is_trade = t == "trade.SPOT" or (subj.startswith("trade.") and "ticker" not in subj)
        if is_trade:
            side = str(d.get("S", d.get("side", ""))).lower()
            trade = RawTrade(
                trade_id=str(d.get("tradeId") or d.get("sequence") or "") or None,
                timestamp_ms=_kucoin_ws_trade_time_ms(d),
                price=float(d.get("p", d.get("price", 0)) or 0),
                qty=float(d.get("size", d.get("q", d.get("volume", 0))) or 0),
                side="sell" if side == "sell" else "buy",
                exchange=self.exchange,
                symbol=self.symbol,
            )
            await self._trade_queue.put(self._wire.scale_trade_from_api(trade))
            return
        if t == "obu.SPOT" and ("a" in d or "b" in d):
            await self._emit_book_from_levels(d.get("b"), d.get("a"), d.get("M"))

    async def _emit_trade_from_classic(self, data: dict) -> None:
        side = str(data.get("side", "")).lower()
        trade = RawTrade(
            trade_id=str(data.get("tradeId") or data.get("sequence") or "") or None,
            timestamp_ms=_kucoin_ws_trade_time_ms(data),
            price=float(data.get("p", data.get("price", 0)) or 0),
            qty=float(data.get("size", data.get("q", data.get("volume", 0))) or 0),
            side="sell" if side == "sell" else "buy",
            exchange=self.exchange,
            symbol=self.symbol,
        )
        await self._trade_queue.put(self._wire.scale_trade_from_api(trade))

    async def _emit_book_from_levels(self, bids: Any, asks: Any, ts: Any) -> None:
        if not isinstance(bids, list) or not isinstance(asks, list):
            return
        if not bids and not asks:
            return
        book = RawOrderbook(
            exchange=self.exchange,
            symbol=self.symbol,
            timestamp_ms=parse_timestamp_ms(ts),
            bids=rows_to_levels(bids, limit=self.depth_limit, reverse=True),
            asks=rows_to_levels(asks, limit=self.depth_limit),
            is_snapshot=True,
        )
        await self._book_queue.put(self._wire.scale_orderbook_from_api(book))

    async def _publish_error(self, source: str, exc: Exception) -> None:
        await self._error_queue.put(
            {
                "source": source,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        )
