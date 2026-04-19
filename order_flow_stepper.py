"""
Order-flow stepper with pluggable market-data providers.
advance(t_end_ms) can be called on an exact schedule independent from animation timing.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from typing import Protocol

from trade_stats import (
    EXCHANGES,
    Orderbook,
    get_balances,
    get_lot_size,
    get_orderbook,
    get_tick_size,
    get_trades_in_range,
    normalize_orderbook_from_raw,
    normalize_trade_to_tick_lot_units,
)
from trade_stats_market_ws import create_market_ws_session, kucoin_ping_pong_json

from market_making_controller.models.flow_implied import Step

# Step fields — see flow_implied.Step:
# - buy_lots/sell_lots/inventory/private_* fill quantities are in integer lots.
# - prices/spreads are in ticks (relative to current mid where relevant).
# - cash and private_trade_fee_tick_lot are in tick*lot units.


class TradeDataProvider(Protocol):
    """Provider for time-windowed trades in tick/lot units."""

    def get_trades(self, t_start_ms: int, t_end_ms: int) -> list[dict]:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


class OrderbookDataProvider(Protocol):
    """Provider for point-in-time orderbook snapshots."""

    def get_orderbook(self, t_end_ms: int) -> Orderbook | None:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


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


class OrderFlowStepper:
    """
    advance(t_end_ms) returns one step over [t_end_ms - T_ms, t_end_ms].
    With use_websocket_market_data: same WebSocketBufferedMarketDataProvider for book + trades
    (websocket only — no REST fallback on trades). Otherwise trades use BufferedPollingTradeDataProvider.
    Order book: websocket or REST per flags.
    Balance updates are configurable via balance_update_mode:
    - "events": apply private trade/order deltas on local balances
    - "rest": refresh balances from adapter REST endpoint each step
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        T_ms: int,
        trades_limit: int = 1000,
        depth_limit: int = 100,
        trade_provider: TradeDataProvider | None = None,
        orderbook_provider: OrderbookDataProvider | None = None,
        use_buffered_orderbook_provider: bool = False,
        use_websocket_market_data: bool = False,
        trades_poll_interval_ms: int = 200,
        trade_buffer_ms: int = 10 * 60 * 1000,
        orderbook_poll_interval_ms: int = 500,
        orderbook_buffer_ms: int = 10 * 60 * 1000,
        websocket_reconnect_ms: int = 2000,
        private_api_token: str | None = None,
        step_window_mode: str = "lookback",
        balance_update_mode: str = "events",
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.private_api_token = private_api_token
        self.T_ms = max(1, T_ms)
        self.trades_limit = trades_limit
        self.depth_limit = depth_limit
        self.step_window_mode = str(step_window_mode or "lookback").strip().lower()
        if self.step_window_mode not in {"lookback", "previous_end"}:
            raise ValueError("step_window_mode must be 'lookback' or 'previous_end'")
        self.balance_update_mode = str(balance_update_mode or "events").strip().lower()
        if self.balance_update_mode not in {"events", "rest"}:
            raise ValueError("balance_update_mode must be 'events' or 'rest'")
        self._lot_size = max(1e-12, float(get_lot_size(symbol, exchange)))
        self._tick_size = max(1e-12, float(get_tick_size(symbol, exchange)))
        self.steps: list[Step] = []
        # Buffered private exchange events (if websocket private channels are enabled).
        self.last_private_trades: list[dict] = []
        self.last_private_orders: list[dict] = []
        # Private event fills in execution-like shape:
        # bids/asks are list[(spread_ticks, qty_lots)].
        self.last_private_trade_fills_bid: list[tuple[int, int]] = []
        self.last_private_trade_fills_ask: list[tuple[int, int]] = []
        self.last_private_order_fills_bid: list[tuple[int, int]] = []
        self.last_private_order_fills_ask: list[tuple[int, int]] = []
        # Account balances initialized from adapter private API and updated by
        # inferred deltas from private order/trade websocket events.
        self.balances: dict[str, dict[str, float]] = {}
        self._order_locked_by_id: dict[str, tuple[str, float]] = {}
        self._applied_private_trade_ids: set[str] = set()
        # Nobitex: merge GET /market/trades/list into private events when WS misses (auth/subscription).
        self._nobitex_rest_private_merge: bool = str(exchange or "").lower().strip() == "nobitex"

        if use_websocket_market_data and trade_provider is None and orderbook_provider is None:
            # One websocket provider: depth + trades from push only (no REST fallback on trades).
            ws = WebSocketBufferedMarketDataProvider(
                exchange=exchange,
                symbol=symbol,
                trades_limit=trades_limit,
                depth_limit=depth_limit,
                max_buffer_ms=max(trade_buffer_ms, orderbook_buffer_ms),
                reconnect_ms=websocket_reconnect_ms,
                private_api_token=private_api_token,
            )
            self.orderbook_provider = ws
            self.trade_provider = ws
        else:
            if trade_provider is not None:
                self.trade_provider = trade_provider
            else:
                self.trade_provider = BufferedPollingTradeDataProvider(
                    exchange=exchange,
                    symbol=symbol,
                    trades_limit=trades_limit,
                    poll_interval_ms=trades_poll_interval_ms,
                    max_buffer_ms=trade_buffer_ms,
                )
            if orderbook_provider is not None:
                self.orderbook_provider = orderbook_provider
            elif use_buffered_orderbook_provider:
                self.orderbook_provider = BufferedPollingOrderbookDataProvider(
                    exchange=exchange,
                    symbol=symbol,
                    depth_limit=depth_limit,
                    poll_interval_ms=orderbook_poll_interval_ms,
                    max_buffer_ms=orderbook_buffer_ms,
                )
            else:
                self.orderbook_provider = RestOrderbookDataProvider(
                    exchange=exchange,
                    symbol=symbol,
                    depth_limit=depth_limit,
                )
        self.trade_provider.start()
        self.orderbook_provider.start()
        self._initialize_balances()

    @staticmethod
    def _to_float(v, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    def _balance_bucket(self, asset: str) -> dict[str, float]:
        a = str(asset or "").upper()
        if not a:
            a = "UNKNOWN"
        if a not in self.balances:
            self.balances[a] = {"free": 0.0, "locked": 0.0, "total": 0.0}
        return self.balances[a]

    def _normalize_balances(self, raw: dict | None) -> dict[str, dict[str, float]]:
        """
        Normalize adapter balance payload to {ASSET: {free, locked, total}}.
        If total is missing, derive it as free + locked.
        """
        out: dict[str, dict[str, float]] = {}
        if not isinstance(raw, dict):
            return out
        for asset, bucket in raw.items():
            a = str(asset or "").upper().strip()
            if not a:
                continue
            b = bucket if isinstance(bucket, dict) else {}
            free = self._to_float(b.get("free", 0.0))
            locked = self._to_float(b.get("locked", 0.0))
            total_raw = b.get("total", None)
            if total_raw is None:
                total = free + locked
            else:
                total = self._to_float(total_raw, free + locked)
            out[a] = {
                "free": float(free),
                "locked": float(max(0.0, locked)),
                "total": float(total),
            }
        return out

    def _add_free_total(self, asset: str, delta: float) -> None:
        b = self._balance_bucket(asset)
        b["free"] = float(b.get("free", 0.0) + delta)
        b["total"] = float(b.get("total", 0.0) + delta)

    def _move_free_locked(self, asset: str, delta_locked: float) -> None:
        # Locking/unlocking should keep total unchanged.
        b = self._balance_bucket(asset)
        b["locked"] = float(max(0.0, b.get("locked", 0.0) + delta_locked))
        b["free"] = float(b.get("free", 0.0) - delta_locked)

    def _initialize_balances(self) -> None:
        raw = get_balances(self.exchange, api_token=self.private_api_token) or {}
        self.balances = self._normalize_balances(raw)

    def _refresh_balances_from_rest(self) -> None:
        fresh = get_balances(self.exchange, api_token=self.private_api_token) or {}
        if fresh:
            self.balances = self._normalize_balances(fresh)

    def _apply_private_trade_balance_deltas(self, events: list[dict]) -> None:
        for e in events:
            tid = str(e.get("id", "") or "")
            if tid and tid in self._applied_private_trade_ids:
                continue
            side = str(e.get("side", "")).strip().lower()
            src = str(e.get("srcCurrency", "")).upper()
            dst = str(e.get("dstCurrency", "")).upper()
            amount = self._to_float(e.get("qty", 0))
            total = self._to_float(e.get("total", 0))
            fee = self._to_float(e.get("fee", 0))
            if amount <= 0.0 and total <= 0.0:
                continue
            # Nobitex private trade payload:
            # - sell: src decreases by amount; dst increases by (total - fee)
            # - buy: src increases by amount; dst decreases by (total + fee)
            if side.startswith("sell"):
                if src:
                    self._add_free_total(src, -amount)
                if dst:
                    self._add_free_total(dst, total - fee)
            elif side.startswith("buy"):
                if src:
                    self._add_free_total(src, amount)
                if dst:
                    self._add_free_total(dst, -(total + fee))
            if tid:
                self._applied_private_trade_ids.add(tid)

    def _apply_private_order_lock_deltas(self, events: list[dict]) -> None:
        # Keep a simple per-order locked estimate so lock/unlock transitions can
        # move balance between free and locked without changing total.
        for e in events:
            oid = str(e.get("orderId", "") or "")
            if not oid:
                continue
            side = str(e.get("side", "")).strip().lower()
            status = str(e.get("status", "")).strip().lower()
            src = str(e.get("srcCurrency", "")).upper()
            dst = str(e.get("dstCurrency", "")).upper()
            amount = max(0.0, self._to_float(e.get("amount", 0)))
            filled = max(0.0, self._to_float(e.get("filledAmount", 0)))
            remaining = max(0.0, amount - filled)
            price = max(0.0, self._to_float(e.get("price", 0)))

            lock_asset = ""
            lock_now = 0.0
            if side.startswith("sell"):
                # Sell locks base asset quantity.
                lock_asset = src
                lock_now = remaining
            elif side.startswith("buy"):
                # Buy locks quote notional (approx).
                lock_asset = dst
                lock_now = remaining * price if price > 0.0 else 0.0

            if status in {"done", "canceled", "rejected"}:
                lock_now = 0.0

            prev_asset, prev_locked = self._order_locked_by_id.get(oid, ("", 0.0))
            if prev_asset and prev_asset != lock_asset and prev_locked != 0.0:
                self._move_free_locked(prev_asset, -prev_locked)
                prev_locked = 0.0
            if lock_asset:
                delta_locked = lock_now - prev_locked
                if abs(delta_locked) > 0.0:
                    self._move_free_locked(lock_asset, delta_locked)
            self._order_locked_by_id[oid] = (lock_asset, lock_now)

    def _private_events_to_side_fills(
        self,
        events: list[dict],
        ob: Orderbook,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Convert private events to bid/ask fill buckets: [(spread_ticks, qty_lots), ...]."""
        bid_acc: dict[int, int] = defaultdict(int)
        ask_acc: dict[int, int] = defaultdict(int)
        for e in events:
            side = str(e.get("side", "")).strip().lower()
            if side.startswith("buy"):
                target = bid_acc
                is_buyer_maker = True
            elif side.startswith("sell"):
                target = ask_acc
                is_buyer_maker = False
            else:
                continue
            try:
                t_norm = normalize_trade_to_tick_lot_units(
                    self.symbol,
                    self.exchange,
                    {
                        "time": int(e.get("time", 0) or 0),
                        "price": str(e.get("price", "0")),
                        "qty": str(e.get("qty", "0")),
                        "isBuyerMaker": is_buyer_maker,
                    },
                )
            except Exception:
                continue
            qty_lots = int(t_norm.get("qty", 0) or 0)
            if qty_lots <= 0:
                continue
            price_ticks = int(t_norm.get("price", 0) or 0)
            if target is bid_acc:
                spread = max(0, int(round(ob.mid - price_ticks)))
            else:
                spread = max(0, int(round(price_ticks - ob.mid)))
            target[spread] += qty_lots
        bids = sorted(bid_acc.items(), key=lambda x: x[0])
        asks = sorted(ask_acc.items(), key=lambda x: x[0])
        return bids, asks

    @staticmethod
    def _buy_sell_lots_from_trades(trades_list: list[dict]) -> tuple[int, int]:
        buy_lots = 0
        sell_lots = 0
        for t in trades_list:
            qty = t.get("qty", 0)
            if not isinstance(qty, int):
                try:
                    qty = int(qty)
                except (TypeError, ValueError):
                    qty = 0
            if t.get("isBuyerMaker") is True:
                sell_lots += qty
            else:
                buy_lots += qty
        return buy_lots, sell_lots

    def _symbol_assets(self) -> tuple[str, str]:
        s = str(self.symbol or "").upper().strip()
        if "_" in s:
            b, q = s.split("_", 1)
            return b, q
        for q in ("USDT", "IRT", "TMN", "RLS", "USD", "BTC", "ETH"):
            if s.endswith(q) and len(s) > len(q):
                return s[: -len(q)], q
        return "", ""

    def _balance_asset_aliases(self, asset: str) -> list[str]:
        a = str(asset or "").upper().strip()
        if not a:
            return []
        # Exchange naming aliases (notably Nobitex: RLS vs IRT).
        if a == "IRT":
            return ["IRT", "RLS", "TMN"]
        if a == "RLS":
            return ["RLS", "IRT"]
        if a == "TMN":
            return ["TMN", "IRT"]
        return [a]

    def _balance_total(self, asset: str) -> float:
        for a in self._balance_asset_aliases(asset):
            if a in self.balances:
                return self._to_float((self.balances.get(a) or {}).get("total", 0.0))
        return 0.0

    def _inventory_cash_from_balances(self) -> tuple[int, int]:
        base, quote = self._symbol_assets()
        # Convert native balances to model units:
        # - inventory: base units -> lots
        # - cash: quote units -> (tick * lot) units
        inv_lots = 0.0
        cash_tick_lot = 0.0
        if base:
            inv_native = self._balance_total(base)
            inv_lots = inv_native / self._lot_size
        if quote:
            cash_native = self._balance_total(quote)
            cash_tick_lot = cash_native / (self._tick_size * self._lot_size)
        return int(round(inv_lots)), int(round(cash_tick_lot))

    @staticmethod
    def _nobitex_rest_trade_to_private_event(row: dict) -> dict:
        """Map Nobitex GET /market/trades/list row to the same shape as websocket private:trades."""
        from trade_stats_market_ws import _parse_iso_time_ms

        ts_raw = row.get("timestamp") or row.get("time")
        t_ms = _parse_iso_time_ms(ts_raw)
        mkt = str(row.get("market") or "").replace(" ", "").upper()
        src_c = str(row.get("srcCurrency") or "").upper()
        dst_c = str(row.get("dstCurrency") or "").upper()
        if "-" in mkt:
            parts = mkt.split("-", 1)
            if len(parts) == 2:
                src_c, dst_c = parts[0], parts[1]
        return {
            "time": t_ms,
            "price": str(row.get("price", "0")),
            "qty": str(row.get("amount", "0")),
            "side": str(row.get("type", "")).lower(),
            "id": row.get("id"),
            "orderId": row.get("orderId"),
            "srcCurrency": src_c,
            "dstCurrency": dst_c,
            "total": str(row.get("total", "0")),
            "fee": str(row.get("fee", "0")),
        }

    def _merge_nobitex_private_trades_with_rest(
        self,
        ws_trades: list[dict],
        t_start_ms: int,
        t_end_ms: int,
    ) -> list[dict]:
        """
        When websocket private:trades misses (auth, subscription, or client parse), fill gaps
        using Nobitex GET /market/trades/list for the same pair (documented 3-day history).
        """
        if not self._nobitex_rest_private_merge or not self.private_api_token:
            return ws_trades
        nob = EXCHANGES.get("nobitex")
        if nob is None or not hasattr(nob, "user_trades_list"):
            return ws_trades
        try:
            rows = nob.user_trades_list(self.symbol, self.private_api_token, limit=200)
        except Exception:
            return ws_trades
        by_id: dict[str, dict] = {}
        for t in ws_trades:
            tid = str(t.get("id", "") or "")
            if tid:
                by_id[tid] = t
            else:
                by_id[f"ws:{t.get('time')},{t.get('price')},{t.get('qty')}"] = t
        for row in rows:
            if not isinstance(row, dict):
                continue
            ev = self._nobitex_rest_trade_to_private_event(row)
            tm = int(ev.get("time") or 0)
            if not (t_start_ms <= tm <= t_end_ms):
                continue
            tid = str(ev.get("id", "") or "")
            if tid and tid in by_id:
                continue
            if tid:
                by_id[tid] = ev
            else:
                by_id[f"rest:{tm},{ev.get('price')},{ev.get('qty')}"] = ev
        return sorted(by_id.values(), key=lambda x: int(x.get("time", 0)))

    def _private_trades_fee_tick_lot(self) -> float:
        """
        Sum `fee` from last_private_trades (exchange quote currency) and convert to tick*lot units
        to match ``cash`` in Step (quote_native / (tick_size * lot_size)).
        """
        total_quote = 0.0
        for t in self.last_private_trades or []:
            if not isinstance(t, dict):
                continue
            try:
                total_quote += abs(float(str(t.get("fee", "0") or "0").replace(",", "")))
            except (TypeError, ValueError):
                continue
        denom = max(1e-18, float(self._tick_size) * float(self._lot_size))
        return float(total_quote / denom)

    def close(self) -> None:
        self.trade_provider.stop()
        self.orderbook_provider.stop()

    def advance(self, t_end_ms: int | None = None) -> Step | None:
        """Fetch one step for [t_end_ms - T_ms, t_end_ms]. Return None on orderbook failure."""
        if t_end_ms is None:
            t_end_ms = int(time.time() * 1000)
        if self.step_window_mode == "previous_end" and self.steps:
            t_start_ms = int(self.steps[-1][0])
        else:
            t_start_ms = t_end_ms - self.T_ms

        ob = self.orderbook_provider.get_orderbook(t_end_ms)
        if ob is None:
            return None
        trades_list = self.trade_provider.get_trades(t_start_ms, t_end_ms)
        if hasattr(self.trade_provider, "get_private_trades"):
            try:
                self.last_private_trades = self.trade_provider.get_private_trades(t_start_ms, t_end_ms)
            except Exception:
                self.last_private_trades = []
        else:
            self.last_private_trades = []
        self.last_private_trades = self._merge_nobitex_private_trades_with_rest(
            self.last_private_trades, t_start_ms, t_end_ms
        )
        if hasattr(self.trade_provider, "get_private_orders"):
            try:
                self.last_private_orders = self.trade_provider.get_private_orders(t_start_ms, t_end_ms)
            except Exception:
                self.last_private_orders = []
        else:
            self.last_private_orders = []

        self.last_private_trade_fills_bid, self.last_private_trade_fills_ask = (
            self._private_events_to_side_fills(self.last_private_trades, ob)
        )
        self.last_private_order_fills_bid, self.last_private_order_fills_ask = (
            self._private_events_to_side_fills(self.last_private_orders, ob)
        )
        if self.balance_update_mode == "events":
            self._apply_private_trade_balance_deltas(self.last_private_trades)
            self._apply_private_order_lock_deltas(self.last_private_orders)
        else:
            # REST-mode balance source of truth: replace balances each step.
            self._refresh_balances_from_rest()

        buy_lots, sell_lots = self._buy_sell_lots_from_trades(trades_list)
        inventory, cash = self._inventory_cash_from_balances()
        private_fills_bid = tuple(self.last_private_trade_fills_bid)
        private_fills_ask = tuple(self.last_private_trade_fills_ask)
        private_trade_fee_tick_lot = self._private_trades_fee_tick_lot()

        step: Step = (
            t_end_ms,
            ob,
            buy_lots,
            sell_lots,
            inventory,
            cash,
            private_trade_fee_tick_lot,
            private_fills_bid,
            private_fills_ask,
        )
        self.steps.append(step)
        return step
