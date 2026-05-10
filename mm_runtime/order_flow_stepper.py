"""
Order-flow stepper with pluggable market-data providers.
advance(t_end_ms) can be called on an exact schedule independent from animation timing.

Canonical implementation (root ``order_flow_stepper.py`` re-exports via PEP 562 ``__getattr__``).
WS/REST/polling selection for market data is composed via ``mm_runtime.market_data_mux``; venue glue remains in ``trade_stats`` / ``trade_stats_market_ws`` for now.
"""

from __future__ import annotations

import math
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

from mm_runtime.market_data_providers import (
    BufferedPollingOrderbookDataProvider,
    BufferedPollingTradeDataProvider,
    RestOrderbookDataProvider,
    RestTradeDataProvider,
    WebSocketBufferedMarketDataProvider,
)


class HedgeGaussianPriceModel:
    """
    Gaussian model for hedge representative-price deltas (buy/sell), updated via
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
        hedge_exchange: str | None = None,
        hedge_symbol: str | None = None,
        use_hedge_websocket_market_data: bool = True,
        hedge_depth_limit: int | None = None,
        hedge_orderbook_poll_interval_ms: int | None = None,
        hedge_orderbook_buffer_ms: int | None = None,
        hedge_websocket_reconnect_ms: int | None = None,
        main_representative_price_target_depth_lots: float | None = None,
        hedge_representative_price_target_depth_lots: float = 1000.0,
        use_hedge_gaussian_price_model: bool = True,
        hedge_price_model_decay: float = 0.97,
        hedge_price_model_sigma_min_ticks: float = 1.0,
        hedge_private_api_token: str | None = None,
        use_hedge_portfolio_balances: bool = False,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.private_api_token = private_api_token
        self.hedge_private_api_token = hedge_private_api_token or private_api_token
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
                hedge_representative_price_target_depth_lots
            )
        # Use a single depth target for BOTH main and hedge representative prices.
        # This keeps Step semantics consistent across venues.
        self._representative_price_target_depth_lots = max(
            1e-9, float(main_representative_price_target_depth_lots)
        )
        self.steps: list[Step] = []
        self.use_hedge_portfolio_balances = bool(use_hedge_portfolio_balances)
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
            hedge_exchange=hedge_exchange,
            hedge_symbol=hedge_symbol,
            use_hedge_websocket_market_data=use_hedge_websocket_market_data,
            hedge_depth_limit=hedge_depth_limit,
            hedge_orderbook_poll_interval_ms=hedge_orderbook_poll_interval_ms,
            hedge_orderbook_buffer_ms=hedge_orderbook_buffer_ms,
            hedge_websocket_reconnect_ms=hedge_websocket_reconnect_ms,
        )
        self.trade_provider = bundle.trade_provider
        self.orderbook_provider = bundle.orderbook_provider
        self.hedge_orderbook_provider = bundle.hedge_orderbook_provider

        self.hedge_exchange = str(hedge_exchange).strip() if hedge_exchange else ""
        self.hedge_symbol = (
            str(hedge_symbol).strip() if hedge_symbol else str(symbol).strip()
        )
        self._hedge_tick_size: float | None = None
        self._hedge_lot_size: float | None = None
        self._hedge_symbol_asset_base: str = ""
        self._hedge_symbol_asset_quote: str = ""
        self.hedge_balances: dict[str, dict[str, float]] = {}
        if self.hedge_exchange:
            self._hedge_tick_size = max(
                1e-12, float(get_tick_size(self.hedge_symbol, self.hedge_exchange))
            )
            self._hedge_lot_size = max(
                1e-12, float(get_lot_size(self.hedge_symbol, self.hedge_exchange))
            )
            self._hedge_symbol_asset_base, _hedge_q_raw = self._symbol_assets_for(
                self.hedge_symbol
            )
            self._hedge_symbol_asset_quote = self._normalize_quote_asset_for_exchange(
                _hedge_q_raw, self.hedge_exchange
            )
        self.trade_provider.start()
        self.orderbook_provider.start()
        if self.hedge_orderbook_provider is not None:
            self.hedge_orderbook_provider.start()
        self._initialize_balances()
        self._initialize_hedge_balances()

    def _convert_hedge_ob_to_main_units(
        self, hedge_ob: Orderbook
    ) -> Orderbook | None:
        """Map hedge orderbook ticks/lots to main exchange tick/lot units."""
        if self._hedge_tick_size is None or self._hedge_lot_size is None:
            return None
        price_scale = float(self._hedge_tick_size / self._tick_size)
        quote_scale = float(self._hedge_quote_to_main_quote_scale())
        qty_scale = float(self._hedge_lot_size / self._lot_size)

        bids = []
        asks = []
        for lev in hedge_ob.bids:
            p_main = max(1, int(round(float(lev.price) * price_scale * quote_scale)))
            q_main = max(0, int(round(float(lev.qty) * qty_scale)))
            if q_main <= 0:
                continue
            bids.append((p_main, q_main))
        for lev in hedge_ob.asks:
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
            t_ms=int(hedge_ob.t_ms),
            mid=float(mid_main),
            tick_size=float(self._tick_size),
            bids=bid_levels,
            asks=ask_levels,
        )

    def _hedge_quote_to_main_quote_scale(self) -> float:
        """
        Convert hedge quote-currency price units into main quote-currency units.
        Most markets are already aligned (scale=1). Handle common Iran quote aliases:
          1 TMN = 10 IRT (Rial).
        """
        main_q = str(self._main_symbol_asset_quote or "").upper().strip()
        hedge_q = str(self._hedge_symbol_asset_quote or "").upper().strip()
        if not main_q or not hedge_q or main_q == hedge_q:
            return 1.0
        # Toman -> Rial
        if hedge_q == "TMN" and main_q in {"IRT", "RLS"}:
            return 10.0
        # Rial -> Toman
        if hedge_q in {"IRT", "RLS"} and main_q == "TMN":
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

    def _hedge_balance_total(self, asset: str) -> float:
        for a in self._balance_asset_aliases(asset):
            if a in self.hedge_balances:
                return self._to_float((self.hedge_balances.get(a) or {}).get("total", 0.0))
        return 0.0

    def _initialize_hedge_balances(self) -> None:
        if not (self.use_hedge_portfolio_balances and self.hedge_exchange):
            self.hedge_balances = {}
            return
        raw = get_balances(self.hedge_exchange, api_token=self.hedge_private_api_token) or {}
        self.hedge_balances = self._normalize_balances(raw)

    def _refresh_hedge_balances_from_rest(self) -> None:
        if not (self.use_hedge_portfolio_balances and self.hedge_exchange):
            self.hedge_balances = {}
            return
        fresh = get_balances(self.hedge_exchange, api_token=self.hedge_private_api_token) or {}
        if fresh:
            self.hedge_balances = self._normalize_balances(fresh)

    def _hedge_inventory_cash_in_main_units(self) -> tuple[int, int]:
        """
        Convert hedge-account balances to main model units:
        - inventory in main lots
        - cash in main tick*lot units
        """
        if not (self.use_hedge_portfolio_balances and self.hedge_exchange):
            return 0, 0
        base = self._hedge_symbol_asset_base
        quote = self._hedge_symbol_asset_quote
        if not base and not quote:
            return 0, 0
        inv_lots = 0.0
        cash_tick_lot = 0.0
        if base:
            inv_native = self._hedge_balance_total(base)
            inv_lots = inv_native / max(1e-12, self._lot_size)
        if quote:
            cash_native = self._hedge_balance_total(quote)
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
        if self.hedge_orderbook_provider is not None:
            self.hedge_orderbook_provider.stop()

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

        # Hedge representative prices (ticks), converted into main tick units.
        hedge_buy_ref: float | None = None
        hedge_sell_ref: float | None = None
        if self.hedge_orderbook_provider is not None:
            try:
                hedge_ob_raw = self.hedge_orderbook_provider.get_orderbook(t_end_ms)
            except Exception:
                hedge_ob_raw = None
            if hedge_ob_raw is not None:
                hedge_ob_main = self._convert_hedge_ob_to_main_units(hedge_ob_raw)
                if hedge_ob_main is not None:
                    a_buy, a_sell = reference_prices_ticks_from_orderbook(
                        hedge_ob_main,
                        target_depth_lots=self._representative_price_target_depth_lots,
                    )
                    hedge_buy_ref = float(a_buy)
                    hedge_sell_ref = float(a_sell)
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
        self._refresh_hedge_balances_from_rest()

        buy_lots, sell_lots = self._buy_sell_lots_from_trades(trades_list)
        inventory, cash = self._inventory_cash_from_balances()
        hedge_inventory, hedge_cash = self._hedge_inventory_cash_in_main_units()
        private_fills_bid = tuple(self.last_private_trade_fills_bid)
        private_fills_ask = tuple(self.last_private_trade_fills_ask)
        private_trade_fee_tick_lot = self._private_trades_fee_tick_lot()

        step = Step(
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
            hedge_buy_ref,
            hedge_sell_ref,
            int(hedge_inventory),
            int(hedge_cash),
        )
        self.steps.append(step)
        return step
