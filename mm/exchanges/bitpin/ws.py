from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import websockets

from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.core.models.market import RawOrderbook, RawTrade
from mm.exchanges.bitpin.private_rest import api_symbol
from mm.exchanges.shared.provider_adapter import parse_timestamp_ms, rows_to_levels


class BitpinWsMarketDataAdapter(AbstractMarketDataAdapter):
    WS_URL = "wss://centrifugo.bitpin.ir/connection/websocket"

    def __init__(self, symbol: str, *, depth_limit: int = 50):
        self.exchange = "bitpin"
        self.symbol = symbol
        self.depth_limit = max(1, int(depth_limit))
        self._wire_symbol = api_symbol(symbol)
        self._connected = False
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
            try:
                async with websockets.connect(self.WS_URL, ping_interval=20, ping_timeout=10) as ws:
                    reconnect_s = 1.0
                    await self._send_connect(ws)
                    await self._subscribe(ws, f"matches:{self._wire_symbol}")
                    await self._subscribe(ws, f"orderbook:{self._wire_symbol}")
                    async for raw in ws:
                        for payload in _centrifuge_payloads(raw):
                            await self.handle_message(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("websocket", exc)
                if self._connected:
                    await asyncio.sleep(reconnect_s)
                    reconnect_s = min(30.0, reconnect_s * 2.0)

    async def _send_connect(self, ws: Any) -> None:
        await ws.send(json.dumps({"id": 1, "connect": {}}))

    async def _subscribe(self, ws: Any, channel: str) -> None:
        await ws.send(json.dumps({"id": _request_id(channel), "subscribe": {"channel": channel}}))

    async def handle_message(self, payload: Any) -> None:
        data = _publication_data(payload)
        if not isinstance(data, dict):
            return
        event = str(data.get("event") or "")
        if event == "market_data":
            await self._book_queue.put(
                RawOrderbook(
                    exchange=self.exchange,
                    symbol=self.symbol,
                    timestamp_ms=parse_timestamp_ms(data.get("event_time")),
                    bids=rows_to_levels(data.get("bids"), limit=self.depth_limit, reverse=True),
                    asks=rows_to_levels(data.get("asks"), limit=self.depth_limit),
                    is_snapshot=True,
                )
            )
            return
        if event == "matches_update":
            for item in data.get("matches") or []:
                if not isinstance(item, dict):
                    continue
                await self._trade_queue.put(
                    RawTrade(
                        trade_id=str(item.get("id")) if item.get("id") is not None else None,
                        timestamp_ms=parse_timestamp_ms(item.get("time") or data.get("event_time")),
                        price=_safe_float(item.get("price")),
                        qty=_safe_float(item.get("base_amount")),
                        side="buy" if str(item.get("side")).lower() == "buy" else "sell",
                        exchange=self.exchange,
                        symbol=self.symbol,
                    )
                )

    async def _publish_error(self, source: str, exc: Exception) -> None:
        await self._error_queue.put({"source": source, "error": type(exc).__name__, "message": str(exc)})


def _publication_data(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    push = payload.get("push")
    if isinstance(push, dict):
        pub = push.get("pub")
        if isinstance(pub, dict):
            data = pub.get("data")
            if isinstance(data, str):
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return None
            return data
    return payload if payload.get("event") else None


def _centrifuge_payloads(raw: Any) -> list[Any]:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    frames = [part for part in text.split("\n") if part.strip()]
    out: list[Any] = []
    for frame in frames:
        try:
            out.append(json.loads(frame))
        except json.JSONDecodeError:
            continue
    return out


def _request_id(channel: str) -> int:
    return abs(hash(channel)) % 1_000_000 + 2


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float(default)
