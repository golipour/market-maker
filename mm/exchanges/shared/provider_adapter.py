from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from collections import deque
from datetime import datetime
from typing import Any, Literal

import requests
import websockets

from mm.core.interfaces.market_data import AbstractMarketDataAdapter
from mm.core.models.market import PriceLevel, RawOrderbook, RawTrade

TransportMode = Literal["rest", "websocket"]


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_timestamp_ms(raw: Any) -> int:
    if raw is None:
        return now_ms()
    text = str(raw).strip()
    if not text:
        return now_ms()
    if "T" in text:
        try:
            return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            return now_ms()
    try:
        value = int(float(text.replace(",", "")))
    except (TypeError, ValueError):
        return now_ms()
    if value >= 10**15:
        return value // 1_000_000
    if value >= 10**12:
        return value
    if value >= 10**9:
        return value * 1000
    return value


class RestClient:
    def __init__(self, base_url: str, timeout_s: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    async def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return await asyncio.to_thread(self._get_json_sync, path, params or {})

    def _get_json_sync(self, path: str, params: dict[str, Any]) -> Any:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()


class AsyncRateLimiter:
    def __init__(self, min_interval_ms: int):
        self._min_interval_s = max(0.0, float(min_interval_ms) / 1000.0)
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        if self._min_interval_s <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            sleep_s = self._min_interval_s - (now - self._last_call)
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            self._last_call = time.monotonic()


class RestPollingMarketDataAdapter(AbstractMarketDataAdapter):
    def __init__(
        self,
        exchange: str,
        symbol: str,
        *,
        poll_interval_ms: int = 1000,
        max_seen_trades: int = 20_000,
    ):
        self.exchange = exchange.lower().strip()
        self.symbol = symbol
        self.poll_interval_ms = max(100, int(poll_interval_ms))
        self._rate_limiter = AsyncRateLimiter(self.poll_interval_ms)
        self._connected = False
        self._error_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        self._max_seen_trades = max(1, int(max_seen_trades))
        self._seen_trade_order: deque[tuple] = deque()
        self._seen_trades: set[tuple] = set()

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def trade_stream(self) -> AsyncIterator[RawTrade]:
        while self._connected:
            try:
                await self._rate_limiter.wait()
                for trade in await self.fetch_trades():
                    if not self._remember_trade(trade):
                        continue
                    yield trade
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("trade_stream", exc)
            await asyncio.sleep(self.poll_interval_ms / 1000.0)

    async def orderbook_stream(self) -> AsyncIterator[RawOrderbook]:
        last_key: tuple[int, float | None, float | None] | None = None
        while self._connected:
            try:
                await self._rate_limiter.wait()
                book = await self.fetch_orderbook()
                bid = book.bids[0].price if book.bids else None
                ask = book.asks[0].price if book.asks else None
                key = (book.timestamp_ms, bid, ask)
                if key != last_key:
                    last_key = key
                    yield book
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("orderbook_stream", exc)
            await asyncio.sleep(self.poll_interval_ms / 1000.0)

    async def error_stream(self) -> AsyncIterator[dict[str, str]]:
        while self._connected:
            yield await self._error_queue.get()

    async def fetch_trades(self) -> list[RawTrade]:
        raise NotImplementedError

    async def fetch_orderbook(self) -> RawOrderbook:
        raise NotImplementedError

    async def _publish_error(self, source: str, exc: Exception) -> None:
        await self._error_queue.put(
            {
                "source": source,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        )

    def _remember_trade(self, trade: RawTrade) -> bool:
        key = self._trade_key(trade)
        if key in self._seen_trades:
            return False
        self._seen_trades.add(key)
        self._seen_trade_order.append(key)
        while len(self._seen_trade_order) > self._max_seen_trades:
            old = self._seen_trade_order.popleft()
            self._seen_trades.discard(old)
        return True

    @staticmethod
    def _trade_key(trade: RawTrade) -> tuple:
        if trade.trade_id:
            return (trade.exchange, trade.symbol, trade.trade_id)
        return (
            trade.exchange,
            trade.symbol,
            trade.timestamp_ms,
            trade.price,
            trade.qty,
            trade.side,
        )


class WebSocketMarketDataAdapter(AbstractMarketDataAdapter):
    def __init__(
        self,
        exchange: str,
        symbol: str,
        *,
        reconnect_initial_ms: int = 1000,
        reconnect_max_ms: int = 30_000,
    ):
        self.exchange = exchange.lower().strip()
        self.symbol = symbol
        self.reconnect_initial_ms = max(100, int(reconnect_initial_ms))
        self.reconnect_max_ms = max(self.reconnect_initial_ms, int(reconnect_max_ms))
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

    def websocket_url(self) -> str:
        raise NotImplementedError

    async def subscribe(self, ws: Any) -> None:
        raise NotImplementedError

    async def handle_message(self, payload: Any) -> None:
        raise NotImplementedError

    async def _run_forever(self) -> None:
        reconnect_ms = self.reconnect_initial_ms
        while self._connected:
            try:
                async with websockets.connect(
                    self.websocket_url(),
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    reconnect_ms = self.reconnect_initial_ms
                    await self.subscribe(ws)
                    async for message in ws:
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError as exc:
                            await self._publish_error("websocket_decode", exc)
                            continue
                        await self.handle_message(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error("websocket", exc)
                if self._connected:
                    await asyncio.sleep(reconnect_ms / 1000.0)
                    reconnect_ms = min(self.reconnect_max_ms, reconnect_ms * 2)

    async def emit_trade(self, trade: RawTrade) -> None:
        await self._trade_queue.put(trade)

    async def emit_orderbook(self, book: RawOrderbook) -> None:
        await self._book_queue.put(book)

    async def _publish_error(self, source: str, exc: Exception) -> None:
        await self._error_queue.put(
            {
                "source": source,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        )


def rows_to_levels(rows: Any, *, limit: int, reverse: bool = False) -> tuple[PriceLevel, ...]:
    levels: list[PriceLevel] = []
    if not isinstance(rows, list):
        return ()
    for row in rows[:limit]:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        try:
            price = float(row[0])
            qty = float(row[1])
        except (TypeError, ValueError):
            continue
        if qty > 0:
            levels.append(PriceLevel(price=price, qty=qty))
    levels.sort(key=lambda level: level.price, reverse=reverse)
    return tuple(levels)


def side_from_taker(is_taker_buy: bool) -> str:
    return "buy" if is_taker_buy else "sell"


def update_depth_side(
    current: list[list[str]],
    rows: Any,
    *,
    descending: bool,
    limit: int,
    row_parser: Callable[[Any], tuple[str, str] | None],
) -> list[list[str]]:
    parsed = []
    if isinstance(rows, list):
        for row in rows:
            item = row_parser(row)
            if item is not None:
                parsed.append([item[0], item[1]])
    parsed.sort(key=lambda row: float(row[0]), reverse=descending)
    return parsed[:limit] if parsed else current
