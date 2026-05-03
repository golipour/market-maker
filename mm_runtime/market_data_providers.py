"""
Concrete market-data adapters: REST, buffered polling, and websocket-backed buffers.

Implements ``mm_core.ports.market_data`` protocols; composed by ``MarketDataMux``.
Venue-specific HTTP/WS handling stays in ``trade_stats`` / ``trade_stats_market_ws``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque

from trade_stats import (
    Orderbook,
    get_orderbook,
    get_trades_in_range,
    normalize_orderbook_from_raw,
    normalize_trade_to_tick_lot_units,
)
from trade_stats_market_ws import create_market_ws_session, kucoin_ping_pong_json

__all__ = [
    "RestTradeDataProvider",
    "RestOrderbookDataProvider",
    "BufferedPollingTradeDataProvider",
    "BufferedPollingOrderbookDataProvider",
    "WebSocketBufferedMarketDataProvider",
]


class RestTradeDataProvider:
    """On-demand REST query for each step window."""

    def __init__(self, exchange: str, symbol: str, trades_limit: int = 1000):
        self.exchange = exchange
        self.symbol = symbol
        self.trades_limit = trades_limit

    def get_trades(self, t_start_ms: int, t_end_ms: int) -> list[dict]:
        try:
            return get_trades_in_range(
                self.symbol,
                t_start_ms,
                t_end_ms,
                limit=self.trades_limit,
                exchange=self.exchange,
            )
        except Exception:
            return []

    def start(self) -> None:
        return

    def stop(self) -> None:
        return


class RestOrderbookDataProvider:
    """On-demand REST query for orderbook snapshots."""

    def __init__(self, exchange: str, symbol: str, depth_limit: int = 100):
        self.exchange = exchange
        self.symbol = symbol
        self.depth_limit = depth_limit

    def get_orderbook(self, t_end_ms: int) -> Orderbook | None:
        try:
            return get_orderbook(
                self.symbol,
                limit=self.depth_limit,
                exchange=self.exchange,
            )
        except Exception:
            return None

    def start(self) -> None:
        return

    def stop(self) -> None:
        return


class BufferedPollingTradeDataProvider:
    """
    Continuously poll trades in the background and keep a rolling buffer.
    Step windows are then read from memory (exact timing windows, decoupled from fetch latency).
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        *,
        trades_limit: int = 1000,
        poll_interval_ms: int = 200,
        max_buffer_ms: int = 10 * 60 * 1000,
        overlap_ms: int = 200,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.trades_limit = max(1, trades_limit)
        self.poll_interval_ms = max(10, poll_interval_ms)
        self.max_buffer_ms = max(1000, max_buffer_ms)
        self.overlap_ms = max(0, overlap_ms)

        self._trades: deque[dict] = deque()
        self._trade_keys: set[tuple] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_fetch_end_ms: int | None = None

    @staticmethod
    def _trade_key(t: dict) -> tuple:
        return (
            int(t.get("time", 0)),
            str(t.get("price", "")),
            int(t.get("qty", 0) or 0),
            bool(t.get("isBuyerMaker", False)),
        )

    def _prune_locked(self, now_ms: int) -> None:
        min_ts = now_ms - self.max_buffer_ms
        while self._trades and int(self._trades[0].get("time", 0)) < min_ts:
            old = self._trades.popleft()
            self._trade_keys.discard(self._trade_key(old))

    def _poll_once(self) -> None:
        now_ms = int(time.time() * 1000)
        if self._last_fetch_end_ms is None:
            start_ms = now_ms - self.max_buffer_ms
        else:
            start_ms = max(0, self._last_fetch_end_ms - self.overlap_ms)

        try:
            batch = get_trades_in_range(
                self.symbol,
                start_ms,
                now_ms,
                limit=self.trades_limit,
                exchange=self.exchange,
            )
        except Exception:
            batch = []

        with self._lock:
            for t in batch:
                key = self._trade_key(t)
                if key in self._trade_keys:
                    continue
                self._trades.append(t)
                self._trade_keys.add(key)
            self._prune_locked(now_ms)

        self._last_fetch_end_ms = now_ms

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(self.poll_interval_ms / 1000.0)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def get_trades(self, t_start_ms: int, t_end_ms: int) -> list[dict]:
        # Lazy-start so users don't need explicit lifecycle wiring.
        self.start()
        with self._lock:
            out = [
                t for t in self._trades
                if t_start_ms <= int(t.get("time", 0)) <= t_end_ms
            ]
        out.sort(key=lambda x: int(x.get("time", 0)))
        return out


class BufferedPollingOrderbookDataProvider:
    """
    Continuously poll orderbook snapshots in the background and keep a rolling buffer.
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        *,
        depth_limit: int = 100,
        poll_interval_ms: int = 500,
        max_buffer_ms: int = 10 * 60 * 1000,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.depth_limit = max(1, depth_limit)
        self.poll_interval_ms = max(50, poll_interval_ms)
        self.max_buffer_ms = max(1000, max_buffer_ms)

        self._obs: deque[Orderbook] = deque()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _prune_locked(self, now_ms: int) -> None:
        min_ts = now_ms - self.max_buffer_ms
        while self._obs and int(self._obs[0].t_ms) < min_ts:
            self._obs.popleft()

    def _poll_once(self) -> None:
        ob = None
        try:
            ob = get_orderbook(
                self.symbol,
                limit=self.depth_limit,
                exchange=self.exchange,
            )
        except Exception:
            ob = None
        if ob is None:
            return
        now_ms = int(time.time() * 1000)
        with self._lock:
            self._obs.append(ob)
            self._prune_locked(now_ms)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(self.poll_interval_ms / 1000.0)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def get_orderbook(self, t_end_ms: int) -> Orderbook | None:
        self.start()
        with self._lock:
            if not self._obs:
                return None
            selected = None
            for ob in reversed(self._obs):
                if int(ob.t_ms) <= t_end_ms:
                    selected = ob
                    break
            return selected if selected is not None else self._obs[-1]


class WebSocketBufferedMarketDataProvider:
    """
    Websocket buffer for order book and trades (same exchange push stream where supported).
    Per-exchange URL, subscribe, and JSON parsing live in trade_stats_market_ws (uses trade_stats adapters).
    get_trades returns only trades ingested from the websocket (no REST fallback on trades).
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        *,
        trades_limit: int = 1000,
        depth_limit: int = 100,
        max_buffer_ms: int = 10 * 60 * 1000,
        reconnect_ms: int = 2000,
        private_api_token: str | None = None,
    ):
        self.exchange = exchange.lower().strip()
        self.symbol = symbol
        self.trades_limit = max(1, trades_limit)
        self.depth_limit = max(1, depth_limit)
        self.max_buffer_ms = max(1000, max_buffer_ms)
        self.reconnect_ms = max(250, reconnect_ms)

        self._trades: deque[dict] = deque()
        self._trade_keys: set[tuple] = set()
        self._private_trades: deque[dict] = deque()
        self._private_orders: deque[dict] = deque()
        self._private_trade_keys: set[tuple] = set()
        self._private_order_keys: set[tuple] = set()
        self._obs: deque[Orderbook] = deque()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._ws_session = create_market_ws_session(
            self.exchange,
            self.symbol,
            self.depth_limit,
            private_api_token=private_api_token,
        )

    @staticmethod
    def _trade_key(t: dict) -> tuple:
        return (
            int(t.get("time", 0)),
            str(t.get("price", "")),
            int(t.get("qty", 0) or 0),
            bool(t.get("isBuyerMaker", False)),
        )

    def _prune_locked(self, now_ms: int) -> None:
        min_ts = now_ms - self.max_buffer_ms
        while self._trades and int(self._trades[0].get("time", 0)) < min_ts:
            old = self._trades.popleft()
            self._trade_keys.discard(self._trade_key(old))
        while self._private_trades and int(self._private_trades[0].get("time", 0)) < min_ts:
            old = self._private_trades.popleft()
            self._private_trade_keys.discard(self._private_trade_key(old))
        while self._private_orders and int(self._private_orders[0].get("time", 0)) < min_ts:
            old = self._private_orders.popleft()
            self._private_order_keys.discard(self._private_order_key(old))
        while self._obs and int(self._obs[0].t_ms) < min_ts:
            self._obs.popleft()

    def _add_trade(self, t: dict) -> None:
        """Store trade in tick/lot units to match normalize_orderbook_from_raw (REST does this in get_trades_in_range)."""
        now_ms = int(time.time() * 1000)
        t_norm = normalize_trade_to_tick_lot_units(self.symbol, self.exchange, t)
        with self._lock:
            key = self._trade_key(t_norm)
            if key in self._trade_keys:
                return
            self._trades.append(t_norm)
            self._trade_keys.add(key)
            self._prune_locked(now_ms)

    def _add_orderbook_raw(self, raw_depth: dict, t_ms: int | None = None) -> None:
        try:
            ob = normalize_orderbook_from_raw(
                symbol=self.symbol,
                exchange=self.exchange,
                raw_depth=raw_depth,
                t_ms=t_ms,
                limit=self.depth_limit,
            )
        except Exception:
            return
        now_ms = int(time.time() * 1000)
        with self._lock:
            self._obs.append(ob)
            self._prune_locked(now_ms)

    @staticmethod
    def _private_trade_key(t: dict) -> tuple:
        return (
            int(t.get("time", 0)),
            str(t.get("id", "")),
            str(t.get("orderId", "")),
            str(t.get("side", "")),
            str(t.get("price", "")),
            str(t.get("qty", "")),
        )

    @staticmethod
    def _private_order_key(o: dict) -> tuple:
        return (
            int(o.get("time", 0)),
            str(o.get("orderId", "")),
            str(o.get("tradeId", "")),
            str(o.get("status", "")),
            str(o.get("side", "")),
            str(o.get("price", "")),
            str(o.get("qty", "")),
        )

    def _add_private_trade(self, t: dict) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock:
            key = self._private_trade_key(t)
            if key in self._private_trade_keys:
                return
            self._private_trades.append(t)
            self._private_trade_keys.add(key)
            self._prune_locked(now_ms)

    def _add_private_order(self, o: dict) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock:
            key = self._private_order_key(o)
            if key in self._private_order_keys:
                return
            self._private_orders.append(o)
            self._private_order_keys.add(key)
            self._prune_locked(now_ms)

    def _run_ws(self) -> None:
        try:
            import websocket  # type: ignore
        except Exception:
            return

        while not self._stop_event.is_set():
            try:
                self._run_one_ws_connection(websocket)
            except Exception:
                pass
            if self._stop_event.is_set():
                break
            time.sleep(self.reconnect_ms / 1000.0)

    def _run_one_ws_connection(self, websocket_module) -> None:
        session = self._ws_session
        if session is None:
            return
        url = session.websocket_url()
        if not url:
            return
        ex = self.exchange

        def on_open(ws):
            d = session.on_connect_open(ws)
            for t in d.trades:
                self._add_trade(t)
            for t in d.private_trades:
                self._add_private_trade(t)
            for o in d.private_orders:
                self._add_private_order(o)
            for raw, tms in d.orderbooks:
                self._add_orderbook_raw(raw, tms)

        def on_message(ws, message: str):
            try:
                payload = json.loads(message)
            except Exception:
                return
            if ex == "kucoin" and isinstance(payload, dict):
                pong = kucoin_ping_pong_json(payload)
                if pong is not None:
                    try:
                        ws.send(pong)
                    except Exception:
                        pass
                    return
            d = session.handle_json(payload)
            for t in d.trades:
                self._add_trade(t)
            for t in d.private_trades:
                self._add_private_trade(t)
            for o in d.private_orders:
                self._add_private_order(o)
            for raw, tms in d.orderbooks:
                self._add_orderbook_raw(raw, tms)

        ws_app = websocket_module.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
        )
        ws_app.run_forever(ping_interval=20, ping_timeout=10)

    def start(self) -> None:
        if self._ws_thread is not None and self._ws_thread.is_alive():
            return
        self._stop_event.clear()
        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=1.0)
        self._ws_thread = None

    def get_trades(self, t_start_ms: int, t_end_ms: int) -> list[dict]:
        self.start()
        with self._lock:
            out = [
                t for t in self._trades
                if t_start_ms <= int(t.get("time", 0)) <= t_end_ms
            ]
        if not out:
            return []
        out.sort(key=lambda x: int(x.get("time", 0)))
        return out

    def get_orderbook(self, t_end_ms: int) -> Orderbook | None:
        self.start()
        with self._lock:
            selected = None
            for ob in reversed(self._obs):
                if int(ob.t_ms) <= t_end_ms:
                    selected = ob
                    break
            if selected is not None:
                return selected
            if self._obs:
                return self._obs[-1]
        return None

    def get_private_trades(self, t_start_ms: int, t_end_ms: int) -> list[dict]:
        self.start()
        with self._lock:
            out = [
                t for t in self._private_trades
                if t_start_ms <= int(t.get("time", 0)) <= t_end_ms
            ]
        out.sort(key=lambda x: int(x.get("time", 0)))
        return out

    def get_private_orders(self, t_start_ms: int, t_end_ms: int) -> list[dict]:
        self.start()
        with self._lock:
            out = [
                o for o in self._private_orders
                if t_start_ms <= int(o.get("time", 0)) <= t_end_ms
            ]
        out.sort(key=lambda x: int(x.get("time", 0)))
        return out
