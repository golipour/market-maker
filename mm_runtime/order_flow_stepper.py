"""
Order-flow stepper with pluggable market-data providers.
advance(t_end_ms) can be called on an exact schedule independent from animation timing.

Canonical implementation (root ``order_flow_stepper.py`` re-exports via PEP 562 ``__getattr__``).
WS/REST/polling selection for market data is composed via ``mm_runtime.market_data_mux``; venue glue remains in ``trade_stats`` / ``trade_stats_market_ws`` for now.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict, deque
import numpy as np

from mm_core.ports.market_data import OrderbookDataProvider, TradeDataProvider
from mm_runtime.market_data_mux import MarketDataMux

from trade_stats import (
    EXCHANGES,
    Level,
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
from mm_core.pricing import reference_prices_from_orderbook_ticks as reference_prices_ticks_from_orderbook

from market_making_controller.models.flow_implied import Step

# Step fields — see flow_implied.Step:
# - buy_lots/sell_lots/inventory/private_* fill quantities are in integer lots.
# - prices/spreads are in ticks (relative to current mid where relevant).
# - cash and private_trade_fee_tick_lot are in tick*lot units.


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


class AnchorGaussianRepPriceModel:
    """
    Gaussian model for anchor representative-price deltas (buy/sell), updated via
    EWMA sufficient statistics (exponentially weighted MLE).
    """

    def __init__(self, decay: float = 0.97, sigma_min_ticks: float = 1.0):
        self.decay = min(0.999999, max(0.0, float(decay)))
        self.sigma_min_ticks = max(1e-9, float(sigma_min_ticks))
        self._m1_buy: float | None = None  # E[delta_buy]
        self._m2_buy: float | None = None  # E[delta_buy^2]
        self._m1_sell: float | None = None  # E[delta_sell]
        self._m2_sell: float | None = None  # E[delta_sell^2]
        self._last_buy_obs: float | None = None
        self._last_sell_obs: float | None = None

    def _update_one(self, x: float, m1: float | None, m2: float | None) -> tuple[float, float]:
        if m1 is None or m2 is None:
            return float(x), float(x) * float(x)
        a = self.decay
        b = 1.0 - a
        m1_new = a * float(m1) + b * float(x)
        m2_new = a * float(m2) + b * float(x) * float(x)
        return float(m1_new), float(m2_new)

    @staticmethod
    def _var_from_moments(m1: float | None, m2: float | None) -> float:
        if m1 is None or m2 is None:
            return 0.0
        return max(0.0, float(m2) - float(m1) * float(m1))

    def update(self, buy_obs: float, sell_obs: float) -> None:
        b = float(buy_obs)
        s = float(sell_obs)
        if self._last_buy_obs is not None and self._last_sell_obs is not None:
            d_buy = b - float(self._last_buy_obs)
            d_sell = s - float(self._last_sell_obs)
            self._m1_buy, self._m2_buy = self._update_one(d_buy, self._m1_buy, self._m2_buy)
            self._m1_sell, self._m2_sell = self._update_one(d_sell, self._m1_sell, self._m2_sell)
        self._last_buy_obs = b
        self._last_sell_obs = s

    def forecast_next_prices(self) -> tuple[float | None, float | None]:
        """
        One-step forecast from delta model:
          p_{t+1|t} = p_t + E[delta].
        """
        if self._last_buy_obs is None or self._last_sell_obs is None:
            return None, None
        mu_db = float(self._m1_buy) if self._m1_buy is not None else 0.0
        mu_ds = float(self._m1_sell) if self._m1_sell is not None else 0.0
        return float(self._last_buy_obs + mu_db), float(self._last_sell_obs + mu_ds)

    def get_params(self) -> dict[str, float]:
        var_buy = self._var_from_moments(self._m1_buy, self._m2_buy)
        var_sell = self._var_from_moments(self._m1_sell, self._m2_sell)
        sigma_buy = max(self.sigma_min_ticks, math.sqrt(var_buy))
        sigma_sell = max(self.sigma_min_ticks, math.sqrt(var_sell))
        return {
            "mu_delta_buy": float(self._m1_buy) if self._m1_buy is not None else 0.0,
            "sigma_delta_buy": float(sigma_buy),
            "mu_delta_sell": float(self._m1_sell) if self._m1_sell is not None else 0.0,
            "sigma_delta_sell": float(sigma_sell),
            "decay": float(self.decay),
        }

    def sample_deltas(
        self,
        shape: tuple[int, int],
        *,
        device=None,
        dtype=None,
        torch_mod=None,
    ):
        """
        Sample Gaussian delta shocks for (buy, sell) representative prices.

        Returns (d_buy, d_sell) with the given shape, in ticks.
        If ``torch_mod`` is provided, returns torch tensors on (device, dtype).
        Otherwise returns numpy arrays (float64).
        """
        n_samples, horizon = int(shape[0]), int(shape[1])
        p = self.get_params()
        mu_db = float(p.get("mu_delta_buy", 0.0) or 0.0)
        mu_ds = float(p.get("mu_delta_sell", 0.0) or 0.0)
        sg_db = max(1e-12, float(p.get("sigma_delta_buy", 1e-6) or 1e-6))
        sg_ds = max(1e-12, float(p.get("sigma_delta_sell", 1e-6) or 1e-6))
        if torch_mod is not None:
            d_buy = torch_mod.randn(n_samples, horizon, device=device, dtype=dtype) * sg_db + mu_db
            d_sell = torch_mod.randn(n_samples, horizon, device=device, dtype=dtype) * sg_ds + mu_ds
            return d_buy, d_sell
        z_buy = np.random.randn(n_samples, horizon)
        z_sell = np.random.randn(n_samples, horizon)
        return z_buy * sg_db + mu_db, z_sell * sg_ds + mu_ds


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
        anchor_exchange: str | None = None,
        anchor_symbol: str | None = None,
        use_anchor_websocket_market_data: bool = True,
        anchor_depth_limit: int | None = None,
        anchor_orderbook_poll_interval_ms: int | None = None,
        anchor_orderbook_buffer_ms: int | None = None,
        anchor_websocket_reconnect_ms: int | None = None,
        main_representative_price_target_depth_lots: float | None = None,
        anchor_representative_price_target_depth_lots: float = 1000.0,
        use_anchor_gaussian_price_model: bool = True,
        anchor_price_model_decay: float = 0.97,
        anchor_price_model_sigma_min_ticks: float = 1.0,
        anchor_private_api_token: str | None = None,
        use_anchor_portfolio_balances: bool = False,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.private_api_token = private_api_token
        self.anchor_private_api_token = anchor_private_api_token or private_api_token
        self._exchange_adapter = EXCHANGES.get(str(exchange or "").lower().strip())
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
        self._main_symbol_asset_base, _main_q_raw = self._symbol_assets_for(self.symbol)
        self._main_symbol_asset_quote = self._normalize_quote_asset_for_exchange(
            _main_q_raw, self.exchange
        )
        if main_representative_price_target_depth_lots is None:
            main_representative_price_target_depth_lots = (
                anchor_representative_price_target_depth_lots
            )
        # Use a single depth target for BOTH main and anchor representative prices.
        # This keeps Step semantics consistent across venues.
        self._representative_price_target_depth_lots = max(
            1e-9, float(main_representative_price_target_depth_lots)
        )
        self.steps: list[Step] = []
        self.use_anchor_portfolio_balances = bool(use_anchor_portfolio_balances)
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
        # Active own orders inferred from private websocket order events.
        # order_id -> (side, price_ticks, remaining_qty_lots, last_event_time_ms)
        self._active_orders_by_id: dict[str, tuple[str, int, int, int]] = {}

        bundle = MarketDataMux.build_providers(
            exchange=exchange,
            symbol=symbol,
            trades_limit=trades_limit,
            depth_limit=depth_limit,
            trade_provider=trade_provider,
            orderbook_provider=orderbook_provider,
            use_buffered_orderbook_provider=use_buffered_orderbook_provider,
            use_websocket_market_data=use_websocket_market_data,
            trades_poll_interval_ms=trades_poll_interval_ms,
            trade_buffer_ms=trade_buffer_ms,
            orderbook_poll_interval_ms=orderbook_poll_interval_ms,
            orderbook_buffer_ms=orderbook_buffer_ms,
            websocket_reconnect_ms=websocket_reconnect_ms,
            private_api_token=private_api_token,
            anchor_exchange=anchor_exchange,
            anchor_symbol=anchor_symbol,
            use_anchor_websocket_market_data=use_anchor_websocket_market_data,
            anchor_depth_limit=anchor_depth_limit,
            anchor_orderbook_poll_interval_ms=anchor_orderbook_poll_interval_ms,
            anchor_orderbook_buffer_ms=anchor_orderbook_buffer_ms,
            anchor_websocket_reconnect_ms=anchor_websocket_reconnect_ms,
        )
        self.trade_provider = bundle.trade_provider
        self.orderbook_provider = bundle.orderbook_provider
        self.anchor_orderbook_provider = bundle.anchor_orderbook_provider

        self.anchor_exchange = str(anchor_exchange).strip() if anchor_exchange else ""
        self.anchor_symbol = (
            str(anchor_symbol).strip() if anchor_symbol else str(symbol).strip()
        )
        self._anchor_tick_size: float | None = None
        self._anchor_lot_size: float | None = None
        self._anchor_symbol_asset_base: str = ""
        self._anchor_symbol_asset_quote: str = ""
        self.anchor_balances: dict[str, dict[str, float]] = {}
        if self.anchor_exchange:
            self._anchor_tick_size = max(
                1e-12, float(get_tick_size(self.anchor_symbol, self.anchor_exchange))
            )
            self._anchor_lot_size = max(
                1e-12, float(get_lot_size(self.anchor_symbol, self.anchor_exchange))
            )
            self._anchor_symbol_asset_base, _anchor_q_raw = self._symbol_assets_for(
                self.anchor_symbol
            )
            self._anchor_symbol_asset_quote = self._normalize_quote_asset_for_exchange(
                _anchor_q_raw, self.anchor_exchange
            )
        self.trade_provider.start()
        self.orderbook_provider.start()
        if self.anchor_orderbook_provider is not None:
            self.anchor_orderbook_provider.start()
        self._initialize_balances()
        self._initialize_anchor_balances()

    def _convert_anchor_ob_to_main_units(
        self, anchor_ob: Orderbook
    ) -> Orderbook | None:
        """Map anchor orderbook ticks/lots to main exchange tick/lot units."""
        if self._anchor_tick_size is None or self._anchor_lot_size is None:
            return None
        price_scale = float(self._anchor_tick_size / self._tick_size)
        quote_scale = float(self._anchor_quote_to_main_quote_scale())
        qty_scale = float(self._anchor_lot_size / self._lot_size)

        bids = []
        asks = []
        for lev in anchor_ob.bids:
            p_main = max(1, int(round(float(lev.price) * price_scale * quote_scale)))
            q_main = max(0, int(round(float(lev.qty) * qty_scale)))
            if q_main <= 0:
                continue
            bids.append((p_main, q_main))
        for lev in anchor_ob.asks:
            p_main = max(1, int(round(float(lev.price) * price_scale * quote_scale)))
            q_main = max(0, int(round(float(lev.qty) * qty_scale)))
            if q_main <= 0:
                continue
            asks.append((p_main, q_main))
        if not bids or not asks:
            return None
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_main = (best_bid + best_ask) / 2.0
        bid_levels = [
            Level(
                price=p,
                qty=q,
                spread_ticks=int(round(mid_main - float(p))),
            )
            for (p, q) in bids
        ]
        ask_levels = [
            Level(
                price=p,
                qty=q,
                spread_ticks=int(round(float(p) - mid_main)),
            )
            for (p, q) in asks
        ]
        return Orderbook(
            t_ms=int(anchor_ob.t_ms),
            mid=float(mid_main),
            tick_size=float(self._tick_size),
            bids=bid_levels,
            asks=ask_levels,
        )

    def _anchor_quote_to_main_quote_scale(self) -> float:
        """
        Convert anchor quote-currency price units into main quote-currency units.
        Most markets are already aligned (scale=1). Handle common Iran quote aliases:
          1 TMN = 10 IRT (Rial).
        """
        main_q = str(self._main_symbol_asset_quote or "").upper().strip()
        anchor_q = str(self._anchor_symbol_asset_quote or "").upper().strip()
        if not main_q or not anchor_q or main_q == anchor_q:
            return 1.0
        # Toman -> Rial
        if anchor_q == "TMN" and main_q in {"IRT", "RLS"}:
            return 10.0
        # Rial -> Toman
        if anchor_q in {"IRT", "RLS"} and main_q == "TMN":
            return 0.1
        return 1.0

    @staticmethod
    def _normalize_quote_asset_for_exchange(asset: str, exchange: str) -> str:
        """
        Normalize quote aliases based on exchange conventions.
        Wallex uses TMN for toman-priced markets; users may still pass IRT symbols.
        """
        a = str(asset or "").upper().strip()
        ex = str(exchange or "").lower().strip()
        if ex == "wallex" and a in {"IRT", "RLS"}:
            return "TMN"
        return a

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

    def _private_order_remaining_lots(self, event: dict) -> int:
        """Parse remaining order quantity from private order event in model lot units."""
        amount = max(0.0, self._to_float(event.get("amount", 0.0)))
        filled = max(0.0, self._to_float(event.get("filledAmount", 0.0)))
        remaining = max(0.0, amount - filled)
        if remaining <= 0.0:
            return 0
        return max(0, int(round(remaining / max(1e-12, self._lot_size))))

    def _private_order_price_ticks(self, event: dict) -> int:
        """Parse private order event price into model tick units."""
        price_native = self._to_float(event.get("price", 0.0))
        if price_native <= 0.0:
            return 0
        return max(1, int(round(price_native / max(1e-12, self._tick_size))))

    def _update_active_orders_from_private_events(self, events: list[dict]) -> None:
        """
        Update active order map using private order websocket events.
        This state is persistent across steps and is used to de-bias orderbook observations.
        """
        if not events:
            return
        for e in events:
            oid = str(e.get("orderId", "") or "")
            if not oid:
                continue
            t_ms = int(self._to_float(e.get("time", 0.0)))
            side = str(e.get("side", "")).strip().lower()
            status = str(e.get("status", "")).strip().lower()
            if status in {"done", "canceled", "rejected"}:
                self._active_orders_by_id.pop(oid, None)
                continue
            if not side.startswith("buy") and not side.startswith("sell"):
                continue
            price_ticks = self._private_order_price_ticks(e)
            remaining_lots = self._private_order_remaining_lots(e)
            if price_ticks <= 0 or remaining_lots <= 0:
                self._active_orders_by_id.pop(oid, None)
                continue
            self._active_orders_by_id[oid] = (side, price_ticks, remaining_lots, t_ms)

    def _subtract_active_orders_from_orderbook(self, ob: Orderbook) -> Orderbook:
        """
        Return orderbook with our currently active orders removed level-wise by price.
        Keeps the original book unchanged and rebuilds spread ticks from corrected mid.
        """
        if not self._active_orders_by_id:
            return ob

        own_bid_by_price: dict[int, int] = defaultdict(int)
        own_ask_by_price: dict[int, int] = defaultdict(int)
        for side, price_ticks, remaining_lots, _t_ms in self._active_orders_by_id.values():
            if price_ticks <= 0 or remaining_lots <= 0:
                continue
            if side.startswith("buy"):
                own_bid_by_price[price_ticks] += int(remaining_lots)
            elif side.startswith("sell"):
                own_ask_by_price[price_ticks] += int(remaining_lots)

        if not own_bid_by_price and not own_ask_by_price:
            return ob

        bids_pairs: list[tuple[int, int]] = []
        for lev in ob.bids:
            rem = int(lev.qty) - int(own_bid_by_price.get(int(lev.price), 0))
            if rem > 0:
                bids_pairs.append((int(lev.price), int(rem)))

        asks_pairs: list[tuple[int, int]] = []
        for lev in ob.asks:
            rem = int(lev.qty) - int(own_ask_by_price.get(int(lev.price), 0))
            if rem > 0:
                asks_pairs.append((int(lev.price), int(rem)))

        # If correction would invalidate one side, keep original snapshot.
        if not bids_pairs or not asks_pairs:
            return ob

        bids_pairs.sort(key=lambda x: x[0], reverse=True)
        asks_pairs.sort(key=lambda x: x[0])
        mid_corr = (float(bids_pairs[0][0]) + float(asks_pairs[0][0])) / 2.0
        bid_levels = [
            Level(
                price=p,
                qty=q,
                spread_ticks=int(round(mid_corr - float(p))),
            )
            for (p, q) in bids_pairs
        ]
        ask_levels = [
            Level(
                price=p,
                qty=q,
                spread_ticks=int(round(float(p) - mid_corr)),
            )
            for (p, q) in asks_pairs
        ]
        return Orderbook(
            t_ms=int(ob.t_ms),
            mid=float(mid_corr),
            tick_size=float(ob.tick_size),
            bids=bid_levels,
            asks=ask_levels,
        )

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

    @staticmethod
    def _symbol_assets_for(symbol: str) -> tuple[str, str]:
        s = str(symbol or "").upper().strip()
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

    def _anchor_balance_total(self, asset: str) -> float:
        for a in self._balance_asset_aliases(asset):
            if a in self.anchor_balances:
                return self._to_float((self.anchor_balances.get(a) or {}).get("total", 0.0))
        return 0.0

    def _initialize_anchor_balances(self) -> None:
        if not (self.use_anchor_portfolio_balances and self.anchor_exchange):
            self.anchor_balances = {}
            return
        raw = get_balances(self.anchor_exchange, api_token=self.anchor_private_api_token) or {}
        self.anchor_balances = self._normalize_balances(raw)

    def _refresh_anchor_balances_from_rest(self) -> None:
        if not (self.use_anchor_portfolio_balances and self.anchor_exchange):
            self.anchor_balances = {}
            return
        fresh = get_balances(self.anchor_exchange, api_token=self.anchor_private_api_token) or {}
        if fresh:
            self.anchor_balances = self._normalize_balances(fresh)

    def _anchor_inventory_cash_in_main_units(self) -> tuple[int, int]:
        """
        Convert anchor-account balances to main model units:
        - inventory in main lots
        - cash in main tick*lot units
        """
        if not (self.use_anchor_portfolio_balances and self.anchor_exchange):
            return 0, 0
        base = self._anchor_symbol_asset_base
        quote = self._anchor_symbol_asset_quote
        if not base and not quote:
            return 0, 0
        inv_lots = 0.0
        cash_tick_lot = 0.0
        if base:
            inv_native = self._anchor_balance_total(base)
            inv_lots = inv_native / max(1e-12, self._lot_size)
        if quote:
            cash_native = self._anchor_balance_total(quote)
            cash_tick_lot = cash_native / max(1e-12, self._tick_size * self._lot_size)
        return int(round(inv_lots)), int(round(cash_tick_lot))

    def _merge_private_trades_with_adapter(
        self,
        private_trades: list[dict],
        t_start_ms: int,
        t_end_ms: int,
    ) -> list[dict]:
        """
        Let adapter optionally supplement private websocket events with REST/private API
        and normalize exchange-specific payloads.
        """
        ad = self._exchange_adapter
        if ad is None or not hasattr(ad, "supplement_private_trades"):
            return private_trades
        try:
            out = ad.supplement_private_trades(
                self.symbol,
                self.private_api_token,
                private_trades,
                t_start_ms,
                t_end_ms,
            )
            return out if isinstance(out, list) else private_trades
        except Exception:
            return private_trades

    def _private_trades_fee_tick_lot(self) -> float:
        """
        Sum private trade fees in quote-native units and convert to tick*lot units:
        quote_native / (tick_size * lot_size).
        If event includes ``feeQuote`` use it directly; otherwise fallback to adapter hook,
        then to raw absolute ``fee``.
        """
        total_quote = 0.0
        ad = self._exchange_adapter
        for t in self.last_private_trades or []:
            if not isinstance(t, dict):
                continue
            try:
                fee_quote = float(str(t.get("feeQuote", "")).replace(",", ""))
                total_quote += abs(fee_quote)
                continue
            except (TypeError, ValueError):
                pass
            if ad is not None and hasattr(ad, "private_trade_fee_quote"):
                try:
                    total_quote += abs(float(ad.private_trade_fee_quote(t)))
                    continue
                except Exception:
                    pass
            try:
                total_quote += abs(float(str(t.get("fee", "0") or "0").replace(",", "")))
            except (TypeError, ValueError):
                continue
        denom = max(1e-18, float(self._tick_size) * float(self._lot_size))
        return float(total_quote / denom)

    def close(self) -> None:
        self.trade_provider.stop()
        self.orderbook_provider.stop()
        if self.anchor_orderbook_provider is not None:
            self.anchor_orderbook_provider.stop()

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
        m_buy, m_sell = reference_prices_ticks_from_orderbook(
            ob,
            target_depth_lots=self._representative_price_target_depth_lots,
        )
        main_buy_ref = float(m_buy)
        main_sell_ref = float(m_sell)

        # Anchor representative prices (ticks), converted into main tick units.
        anchor_buy_ref: float | None = None
        anchor_sell_ref: float | None = None
        if self.anchor_orderbook_provider is not None:
            try:
                anchor_ob_raw = self.anchor_orderbook_provider.get_orderbook(t_end_ms)
            except Exception:
                anchor_ob_raw = None
            if anchor_ob_raw is not None:
                anchor_ob_main = self._convert_anchor_ob_to_main_units(anchor_ob_raw)
                if anchor_ob_main is not None:
                    a_buy, a_sell = reference_prices_ticks_from_orderbook(
                        anchor_ob_main,
                        target_depth_lots=self._representative_price_target_depth_lots,
                    )
                    anchor_buy_ref = float(a_buy)
                    anchor_sell_ref = float(a_sell)
        trades_list = self.trade_provider.get_trades(t_start_ms, t_end_ms)
        if hasattr(self.trade_provider, "get_private_trades"):
            try:
                self.last_private_trades = self.trade_provider.get_private_trades(t_start_ms, t_end_ms)
            except Exception:
                self.last_private_trades = []
        else:
            self.last_private_trades = []
        self.last_private_trades = self._merge_private_trades_with_adapter(
            self.last_private_trades, t_start_ms, t_end_ms
        )
        if hasattr(self.trade_provider, "get_private_orders"):
            try:
                self.last_private_orders = self.trade_provider.get_private_orders(t_start_ms, t_end_ms)
            except Exception:
                self.last_private_orders = []
        else:
            self.last_private_orders = []
        self._update_active_orders_from_private_events(self.last_private_orders)
        ob = self._subtract_active_orders_from_orderbook(ob)

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
        self._refresh_anchor_balances_from_rest()

        buy_lots, sell_lots = self._buy_sell_lots_from_trades(trades_list)
        inventory, cash = self._inventory_cash_from_balances()
        anchor_inventory, anchor_cash = self._anchor_inventory_cash_in_main_units()
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
            main_buy_ref,
            main_sell_ref,
            anchor_buy_ref,
            anchor_sell_ref,
            int(anchor_inventory),
            int(anchor_cash),
        )
        self.steps.append(step)
        return step
