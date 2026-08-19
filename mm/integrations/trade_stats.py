"""
Order fill statistics for market making (public API only).

Design (simple):
  At times 0, T, 2T, ... we get orderbook and trades.
  X = orderbook at (N-1)*T
  Y = which orders from X got filled in the next T (trades in [(N-1)*T, N*T])
  Dataset = list of (X, Y) samples.

So: previous orderbook is X; trades in the next window give fills for X → Y.

Supported exchanges: tabdeal, nobitex, wallex, bitpin, kucoin, binance, bitwana.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol
import os

import requests
from urllib.parse import quote

# -----------------------------------------------------------------------------
# Exchange adapters (public API: depth, trades, server time, exchange info)
# Trades must return list of dicts with: time (ms), price, qty, isBuyerMaker
# -----------------------------------------------------------------------------


class ExchangeAdapter(Protocol):
    """Adapter for exchange public API + optional private trading methods."""

    def server_time_ms(self) -> int: ...
    def depth(self, symbol: str, limit: int) -> dict[str, Any]: ...
    def trades(self, symbol: str, limit: int) -> list[dict]: ...
    def exchange_info(self, symbol: str) -> dict[str, Any]: ...
    def balances(self, api_token: str | None = None) -> dict[str, dict[str, float]]: ...
    def list_open_orders(self, symbol: str, api_token: str | None) -> list[dict]: ...
    def cancel_order(self, order_id: int, api_token: str | None) -> dict[str, Any]: ...
    def cancel_all_orders(self, symbol: str, api_token: str | None, execution: str = "limit") -> dict[str, Any]: ...
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: str | float,
        price: str | float | int,
        api_token: str | None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]: ...


def _symbol_for_exchange(symbol: str, use_underscore: bool) -> str:
    """Normalize symbol: user always passes BTC_IRT. Tabdeal keeps it, others use BTCIRT."""
    return symbol if use_underscore else symbol.replace("_", "")


class TabdealAdapter:
    """Tabdeal: https://docs.tabdeal.org/  Accepts BTC_IRT (underscore)."""
    BASE_URL = "https://api1.tabdeal.org"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        r = requests.get(f"{self.BASE_URL}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    def _symbol(self, symbol: str) -> str:
        return _symbol_for_exchange(symbol, use_underscore=True)

    def server_time_ms(self) -> int:
        return int(self._get("/r/api/v1/time")["serverTime"])

    def depth(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        s = self._symbol(symbol)
        return self._get("/r/api/v1/depth", {"tabdealSymbol": s, "limit": limit})

    def trades(self, symbol: str, limit: int = 1000) -> list[dict]:
        s = self._symbol(symbol)
        raw = self._get("/r/api/v1/trades", {"tabdealSymbol": s, "limit": limit})
        return [{"time": t["time"], "price": t["price"], "qty": t["qty"], "isBuyerMaker": t.get("isBuyerMaker", False)} for t in raw]

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        s = self._symbol(symbol)
        out = self._get("/r/api/v1/exchangeInfo", {"tabdealSymbol": s})
        return out[0] if isinstance(out, list) and out else (out if isinstance(out, dict) else {})

    def balances(self, api_token: str | None = None) -> dict[str, dict[str, float]]:
        return {}

    def list_open_orders(self, symbol: str, api_token: str | None) -> list[dict]:
        return []

    def cancel_order(self, order_id: int, api_token: str | None) -> dict[str, Any]:
        return {"status": "failed", "message": "not supported"}

    def cancel_all_orders(self, symbol: str, api_token: str | None, execution: str = "limit") -> dict[str, Any]:
        return {"status": "failed", "message": "not supported"}

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: str | float,
        price: str | float | int,
        api_token: str | None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        return {"status": "failed", "message": "not supported"}



class NobitexAdapter:
    """Nobitex: https://apidocs.nobitex.ir/  Accepts BTC_IRT → uses BTCIRT."""
    BASE_URL = "https://apiv2.nobitex.ir"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        r = requests.get(f"{self.BASE_URL}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    def _get_auth(self, path: str, api_token: str, params: dict[str, Any] | None = None) -> Any:
        headers = {"Authorization": f"Token {api_token}"}
        r = requests.get(f"{self.BASE_URL}{path}", headers=headers, params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post_auth(self, path: str, api_token: str, data: dict[str, Any] | None = None) -> Any:
        headers = {"Authorization": f"Token {api_token}"}
        r = requests.post(f"{self.BASE_URL}{path}", headers=headers, data=data or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    def _symbol(self, symbol: str) -> str:
        return _symbol_for_exchange(symbol, use_underscore=False)

    def server_time_ms(self) -> int:
        return int(time.time() * 1000)

    def depth(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        s = self._symbol(symbol)
        out = self._get(f"/v3/orderbook/{s}")
        if out.get("status") != "ok":
            raise ValueError(out.get("message", "orderbook error"))
        return {"bids": out.get("bids", []), "asks": out.get("asks", [])}

    def trades(self, symbol: str, limit: int = 1000) -> list[dict]:
        s = self._symbol(symbol)
        out = self._get(f"/v2/trades/{s}")
        if out.get("status") != "ok":
            raise ValueError(out.get("message", "trades error"))
        # type "sell" = taker sold = hit bid = maker was buyer -> isBuyerMaker True
        # type "buy" = taker bought = hit ask -> isBuyerMaker False
        lst = []
        for t in (out.get("trades") or [])[:limit]:
            lst.append({
                "time": t.get("time", 0),
                "price": str(t.get("price", "0")),
                "qty": str(t.get("volume", "0")),
                "isBuyerMaker": (t.get("type") == "sell"),
            })
        return lst

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        s = self._symbol(symbol)
        opt = self._get("/v2/options")
        nob = opt.get("nobitex") or {}
        price_prec = nob.get("pricePrecisions") or {}
        amount_prec = nob.get("amountPrecisions") or {}
        min_orders = nob.get("minOrders") or {}
        s_up = s.upper()
        raw = price_prec.get(s_up)
        try:
            # API returns tick size directly as string: "10" = 10 Rial, "0.01" = 0.01 (not decimal places)
            tick = Decimal(str(raw)) if raw is not None else Decimal("1")
            if tick <= 0:
                tick = Decimal("1")
        except Exception:
            tick = Decimal("1")
        step_raw = amount_prec.get(s_up)
        try:
            step = Decimal(str(step_raw)) if step_raw is not None else Decimal("0.0001")
            if step <= 0:
                step = Decimal("0.0001")
        except Exception:
            step = Decimal("0.0001")

        # minOrders is quote-currency keyed (e.g. {"rls": "3000000"}). Infer quote by
        # suffix match with IRT/RLS/TMN aliases so *_IRT symbols can still resolve
        # Nobitex's RLS notional minimum.
        min_notional = None
        if isinstance(min_orders, dict) and min_orders:
            quote_key = None
            norm_keys: dict[str, str] = {str(x).upper(): str(x) for x in min_orders.keys()}
            candidates: list[str] = []
            # Exact suffix candidates from known market symbols.
            if s_up.endswith("RLS") or s_up.endswith("IRT") or s_up.endswith("TMN"):
                candidates.extend(["RLS", "IRT", "TMN"])
            # Generic fallback: try all keys by suffix match.
            for k in sorted(norm_keys.keys(), key=len, reverse=True):
                if s_up.endswith(k):
                    candidates.append(k)
            # Preserve order while deduplicating.
            seen: set[str] = set()
            for cand in candidates:
                c = str(cand).upper()
                if c in seen:
                    continue
                seen.add(c)
                # Direct key first.
                if c in norm_keys:
                    quote_key = norm_keys[c]
                    break
                # Alias key fallback.
                if c == "IRT" and "RLS" in norm_keys:
                    quote_key = norm_keys["RLS"]
                    break
                if c == "RLS" and "IRT" in norm_keys:
                    quote_key = norm_keys["IRT"]
                    break
                if c == "TMN" and "IRT" in norm_keys:
                    quote_key = norm_keys["IRT"]
                    break
                if c == "IRT" and "TMN" in norm_keys:
                    quote_key = norm_keys["TMN"]
                    break
            if quote_key is not None:
                try:
                    mn = Decimal(str(min_orders.get(quote_key)))
                    if mn > 0:
                        min_notional = mn
                except Exception:
                    min_notional = None

        filters: list[dict[str, str]] = [
            {"filterType": "PRICE_FILTER", "tickSize": str(tick)},
            {"filterType": "LOT_SIZE", "stepSize": str(step), "minQty": str(step)},
        ]
        if min_notional is not None:
            filters.append({"filterType": "MIN_NOTIONAL", "minNotional": str(min_notional)})
        return {"filters": filters}

    def websocket_connection_token(self, api_token: str | None) -> str | None:
        """GET /auth/ws/token/ and return connection token for websocket auth."""
        if not api_token:
            return None
        try:
            out = self._get_auth("/auth/ws/token/", api_token)
        except Exception:
            return None
        if out.get("status") != "ok":
            return None
        tok = out.get("token")
        return str(tok) if tok else None

    def websocket_auth_param(self, api_token: str | None) -> str | None:
        """GET /users/profile and return websocketAuthParam for private channels."""
        if not api_token:
            return None
        try:
            out = self._get_auth("/users/profile", api_token)
        except Exception:
            return None
        if out.get("status") != "ok":
            return None
        profile = out.get("profile") or {}
        if not isinstance(profile, dict):
            profile = {}
        p = (
            profile.get("websocketAuthParam")
            or profile.get("websocket_auth_param")
            or out.get("websocketAuthParam")
            or out.get("websocket_auth_param")
        )
        return str(p) if p else None

    def user_trades_list(
        self,
        symbol: str,
        api_token: str | None,
        *,
        limit: int = 100,
        from_id: int | None = None,
    ) -> list[dict]:
        """
        GET /market/trades/list — authenticated user trades (last ~3 days, paginated).
        See Nobitex docs: market/trades/list with srcCurrency & dstCurrency.
        """
        if not api_token:
            return []
        src, dst = self._split_symbol_pair(symbol)
        params: dict[str, Any] = {"srcCurrency": src, "dstCurrency": dst}
        if from_id is not None:
            params["fromId"] = int(from_id)
        try:
            out = self._get_auth("/market/trades/list", api_token, params=params)
        except Exception:
            return []
        if out.get("status") != "ok":
            return []
        trades = out.get("trades") or []
        if not isinstance(trades, list):
            return []
        return [t for t in trades[:limit] if isinstance(t, dict)]

    def balances(self, api_token: str | None = None) -> dict[str, dict[str, float]]:
        """
        Wallet balances by asset from private endpoints.
        Shape: { "BTC": {"free": x, "locked": y, "total": x+y}, ... }
        """
        if not api_token:
            return {}
        result: dict[str, dict[str, float]] = {}
        # Primary path: documented wallet list endpoint.
        try:
            out = self._get_auth("/users/wallets/list", api_token)
            if out.get("status") == "ok":
                wallets = out.get("wallets") or []
                if isinstance(wallets, list):
                    for w in wallets:
                        if not isinstance(w, dict):
                            continue
                        asset = str(w.get("currency", "")).upper()
                        if not asset:
                            continue
                        free = self._safe_float(
                            w.get("activeBalance", w.get("balance", 0))
                        )
                        locked = self._safe_float(
                            w.get("blockedBalance", 0)
                        )
                        total = self._safe_float(
                            w.get("balance", free + locked)
                        )
                        result[asset] = {"free": free, "locked": locked, "total": total}
        except Exception:
            pass
        if result:
            return result

        # Fallback path: v2 wallets format (wallets object by asset key).
        try:
            out = self._get_auth("/v2/wallets", api_token)
            if out.get("status") != "ok":
                return {}
            wallets = out.get("wallets") or {}
            if not isinstance(wallets, dict):
                return {}
            for asset, w in wallets.items():
                if not isinstance(w, dict):
                    continue
                free = self._safe_float(w.get("balance", 0))
                locked = self._safe_float(w.get("blocked", 0))
                total = free + locked
                result[str(asset).upper()] = {"free": free, "locked": locked, "total": total}
        except Exception:
            return {}
        return result

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _split_symbol_pair(symbol: str) -> tuple[str, str]:
        s = str(symbol or "").strip().upper()
        if "_" in s:
            base, quote = s.split("_", 1)
        else:
            base, quote = s[:-3], s[-3:]
        quote = "RLS" if quote == "IRT" else quote
        return base.lower(), quote.lower()

    @staticmethod
    def _parse_time_ms(value: Any) -> int:
        if value is None:
            return int(time.time() * 1000)
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value).strip()
        if not s:
            return int(time.time() * 1000)
        try:
            return int(float(s))
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return int(time.time() * 1000)

    def normalize_private_trade_event(self, event: dict, *, source: str = "ws") -> dict | None:
        """Normalize private trade payloads to a unified shape used by stepper/providers."""
        if not isinstance(event, dict):
            return None
        ts_raw = event.get("timestamp") or event.get("time")
        src_c = str(event.get("srcCurrency") or "").upper()
        dst_c = str(event.get("dstCurrency") or "").upper()
        market = str(event.get("market") or "").replace(" ", "").upper()
        if "-" in market:
            parts = market.split("-", 1)
            if len(parts) == 2:
                src_c = src_c or parts[0]
                dst_c = dst_c or parts[1]
        side = str(event.get("type", event.get("side", ""))).strip().lower()
        price = str(event.get("price", "0"))
        qty = str(event.get("amount", event.get("tradeAmount", event.get("qty", "0"))))
        fee = str(event.get("fee", "0"))
        try:
            fee_val = abs(float(str(fee).replace(",", "")))
        except Exception:
            fee_val = 0.0
        try:
            price_val = float(str(price).replace(",", ""))
        except Exception:
            price_val = 0.0
        # Nobitex semantics:
        # - buy fee is typically in base -> convert to quote using price
        # - sell fee is in quote
        fee_quote = fee_val * price_val if side.startswith("buy") and price_val > 0.0 else fee_val
        return {
            "time": self._parse_time_ms(ts_raw),
            "price": price,
            "qty": qty,
            "side": side,
            "id": event.get("id"),
            "orderId": event.get("orderId"),
            "srcCurrency": src_c,
            "dstCurrency": dst_c,
            "total": str(event.get("total", "0")),
            "fee": fee,
            "feeQuote": str(fee_quote),
            "source": str(source or "ws"),
        }

    def private_trade_fee_quote(self, event: dict) -> float:
        """Return private trade fee in quote-native units from a normalized event."""
        if not isinstance(event, dict):
            return 0.0
        try:
            return abs(float(str(event.get("feeQuote", "")).replace(",", "")))
        except Exception:
            pass
        try:
            fee_val = abs(float(str(event.get("fee", "0")).replace(",", "")))
        except Exception:
            fee_val = 0.0
        side = str(event.get("side", "")).strip().lower()
        try:
            px = float(str(event.get("price", "0")).replace(",", ""))
        except Exception:
            px = 0.0
        if side.startswith("buy") and px > 0.0:
            return fee_val * px
        return fee_val

    def normalize_private_order_event(self, event: dict) -> dict | None:
        """Normalize private order payloads to a unified shape used by stepper/providers."""
        if not isinstance(event, dict):
            return None
        qty = event.get("tradeAmount")
        if qty is None:
            qty = event.get("filledAmount")
        if qty is None:
            qty = event.get("amount")
        px = event.get("tradePrice")
        if px is None:
            px = event.get("avgFilledPrice")
        if px is None:
            px = event.get("price")
        return {
            "time": self._parse_time_ms(event.get("eventTime", event.get("time"))),
            "price": str(px if px is not None else "0"),
            "qty": str(qty if qty is not None else "0"),
            "side": str(event.get("side", "")).lower(),
            "status": str(event.get("status", "")).lower(),
            "orderId": event.get("orderId"),
            "tradeId": event.get("tradeId"),
            "srcCurrency": str(event.get("srcCurrency", "")).upper(),
            "dstCurrency": str(event.get("dstCurrency", "")).upper(),
            "amount": str(event.get("amount", "0")),
            "filledAmount": str(event.get("filledAmount", "0")),
        }

    def supplement_private_trades(
        self,
        symbol: str,
        api_token: str | None,
        ws_trades: list[dict],
        t_start_ms: int,
        t_end_ms: int,
        *,
        limit: int = 200,
    ) -> list[dict]:
        """
        Merge websocket private trades with authenticated REST user trades list.
        Returns normalized events for the requested time window.
        """
        if not api_token:
            return ws_trades
        try:
            rows = self.user_trades_list(symbol, api_token, limit=limit)
        except Exception:
            return ws_trades
        by_id: dict[str, dict] = {}
        for t in ws_trades:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id", "") or "")
            if tid:
                by_id[tid] = t
            else:
                by_id[f"ws:{t.get('time')},{t.get('price')},{t.get('qty')}"] = t
        for row in rows:
            norm = self.normalize_private_trade_event(row, source="rest")
            if not isinstance(norm, dict):
                continue
            tm = int(norm.get("time") or 0)
            if not (t_start_ms <= tm <= t_end_ms):
                continue
            tid = str(norm.get("id", "") or "")
            if tid and tid in by_id:
                continue
            if tid:
                by_id[tid] = norm
            else:
                by_id[f"rest:{tm},{norm.get('price')},{norm.get('qty')}"] = norm
        return sorted(by_id.values(), key=lambda x: int(x.get("time", 0)))

    def list_open_orders(self, symbol: str, api_token: str | None) -> list[dict]:
        if not api_token:
            return []
        src, dst = self._split_symbol_pair(symbol)
        try:
            out = self._get_auth(
                "/market/orders/list",
                api_token,
                params={"status": "open", "srcCurrency": src, "dstCurrency": dst, "details": 2},
            )
        except Exception:
            return []
        if out.get("status") != "ok":
            return []
        orders = out.get("orders") or []
        return [o for o in orders if isinstance(o, dict)]

    def cancel_order(self, order_id: int, api_token: str | None) -> dict[str, Any]:
        if not api_token:
            return {"status": "failed", "message": "missing api token"}
        try:
            out = self._post_auth(
                "/market/orders/update-status",
                api_token,
                data={"order": int(order_id), "status": "canceled"},
            )
        except Exception as e:
            return {"status": "failed", "message": str(e)}
        return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}

    def cancel_all_orders(self, symbol: str, api_token: str | None, execution: str = "limit") -> dict[str, Any]:
        if not api_token:
            return {"status": "failed", "message": "missing api token"}
        src, dst = self._split_symbol_pair(symbol)
        payload = {"srcCurrency": src, "dstCurrency": dst, "execution": execution}
        try:
            out = self._post_auth("/market/orders/cancel-old", api_token, data=payload)
        except Exception as e:
            return {"status": "failed", "message": str(e)}
        return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: str | float,
        price: str | float | int,
        api_token: str | None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if not api_token:
            return {"status": "failed", "message": "missing api token"}
        src, dst = self._split_symbol_pair(symbol)
        payload: dict[str, Any] = {
            "type": str(side).lower(),
            "execution": "limit",
            "srcCurrency": src,
            "dstCurrency": dst,
            "amount": str(amount),
            "price": str(price),
        }
        if client_order_id:
            payload["clientOrderId"] = str(client_order_id)[:32]
        try:
            out = self._post_auth("/market/orders/add", api_token, data=payload)
        except Exception as e:
            return {"status": "failed", "message": str(e)}
        return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}


class WallexAdapter:
    """Wallex: https://api-docs.wallex.ir/  Accepts BTC_IRT → uses BTCTMN. Wallex uses TMN (Toman), not IRT."""
    BASE_URL = "https://api.wallex.ir"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        r = requests.get(f"{self.BASE_URL}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    def _get_auth(self, path: str, api_token: str, params: dict[str, Any] | None = None) -> Any:
        headers = {
            "X-API-Key": api_token,
            "Accept": "application/json",
        }
        r = requests.get(
            f"{self.BASE_URL}{path}",
            params=params or {},
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def _symbol(self, symbol: str) -> str:
        # Wallex uses TMN (Toman), not IRT. Map e.g. USDT_IRT → USDTTMN, BTC_IRT → BTCTMN.
        s = _symbol_for_exchange(symbol, use_underscore=False)
        if s.endswith("IRT"):
            return s[:-3] + "TMN"
        return s

    def server_time_ms(self) -> int:
        return int(time.time() * 1000)

    def depth(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        s = self._symbol(symbol)
        out = self._get("/v1/depth", {"symbol": s})
        if not out.get("success"):
            raise ValueError(out.get("message", "depth error"))
        res = out.get("result") or {}
        # ask/bid are [{price, quantity, sum}]
        def rows(arr: list, n: int) -> list[list[str]]:
            return [[str(x["price"]), str(x["quantity"])] for x in (arr or [])[:n]]
        return {"bids": rows(res.get("bid"), limit), "asks": rows(res.get("ask"), limit)}

    def trades(self, symbol: str, limit: int = 1000) -> list[dict]:
        s = self._symbol(symbol)
        out = self._get("/v1/trades", {"symbol": s})
        if not out.get("success"):
            raise ValueError(out.get("message", "trades error"))
        res = out.get("result") or {}
        lst = []
        for t in (res.get("latestTrades") or [])[:limit]:
            ts = t.get("timestamp") or ""
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                time_ms = int(dt.timestamp() * 1000)
            except Exception:
                time_ms = 0
            # isBuyOrder true = taker bought = hit ask -> maker seller -> isBuyerMaker False
            lst.append({
                "time": time_ms,
                "price": str(t.get("price", "0")),
                "qty": str(t.get("quantity", "0")),
                "isBuyerMaker": not t.get("isBuyOrder", False),
            })
        return lst

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        s = self._symbol(symbol)
        out = self._get("/v1/markets")
        if not out.get("success"):
            return {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "1"}]}
        syms = (out.get("result") or {}).get("symbols") or {}
        info = syms.get(s) or {}
        tick_decimals = info.get("tickSize", 2)
        try:
            tick = Decimal(10) ** (-int(tick_decimals))
        except Exception:
            tick = Decimal("1")
        qty_decimals = info.get("stepSize")
        try:
            qty_step = Decimal(10) ** (-int(qty_decimals))
        except Exception:
            qty_step = Decimal("0.0001")
        filters = [
            {"filterType": "PRICE_FILTER", "tickSize": str(tick)},
            {
                "filterType": "LOT_SIZE",
                "stepSize": str(qty_step),
                "minQty": str(info.get("minQty", "0") or "0"),
            },
        ]
        min_notional = info.get("minNotional")
        if min_notional is not None and str(min_notional).strip():
            filters.append({"filterType": "MIN_NOTIONAL", "minNotional": str(min_notional)})
        return {"filters": filters}

    def balances(self, api_token: str | None = None) -> dict[str, dict[str, float]]:
        if not api_token:
            return {}
        # Wallex docs: GET /v1/account/balances with X-API-Key.
        # Shape:
        # {
        #   "success": true,
        #   "result": {
        #     "balances": {
        #       "TMN": {"asset":"TMN","value":"...","locked":"..."},
        #       ...
        #     }
        #   }
        # }
        # Normalize to {ASSET: {free, locked, total}}.
        candidates = [
            "/v1/account/balances",
            "/v1/account/wallets",
            "/v1/account/profile",
            "/v1/account",
        ]
        for path in candidates:
            try:
                out = self._get_auth(path, api_token)
            except Exception:
                continue
            result: dict[str, dict[str, float]] = {}
            if isinstance(out, dict) and out.get("success") is False:
                continue
            payload = out.get("result", out) if isinstance(out, dict) else {}
            rows = None
            asset_keyed = False
            if isinstance(payload, dict):
                rows = (
                    payload.get("balances")
                    or payload.get("wallets")
                    or payload.get("assets")
                    or payload.get("accounts")
                )
            if rows is None and isinstance(payload, list):
                rows = payload
            if isinstance(rows, dict):
                # Most likely Wallex shape: dict keyed by asset symbol.
                asset_keyed = True
                rows = list(rows.items())
            if not isinstance(rows, list):
                continue
            for w in rows:
                row = None
                row_asset = ""
                if asset_keyed and isinstance(w, tuple) and len(w) == 2:
                    k, v = w
                    row_asset = str(k or "").upper()
                    row = v if isinstance(v, dict) else {}
                elif isinstance(w, dict):
                    row = w
                if not isinstance(row, dict):
                    continue
                asset = str(
                    row.get("asset")
                    or row.get("currency")
                    or row.get("coin")
                    or row.get("symbol")
                    or row_asset
                    or ""
                ).upper()
                if not asset:
                    continue
                free = self._safe_float(
                    row.get(
                        "free",
                        row.get(
                            "available",
                            row.get(
                                "value",
                                row.get("balance", 0.0),
                            ),
                        ),
                    )
                )
                locked = self._safe_float(
                    row.get("locked", row.get("inOrder", row.get("hold", 0.0)))
                )
                total = self._safe_float(row.get("total", free + locked))
                result[asset] = {"free": free, "locked": locked, "total": total}
            if result:
                return result
        return {}

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)


class BitPinAdapter:
    """BitPin: https://docs.bitpin.ir/  Accepts BTC_IRT (underscore). API: api.bitpin.org."""
    BASE_URL = "https://api.bitpin.org"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        r = requests.get(f"{self.BASE_URL}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    def _symbol(self, symbol: str) -> str:
        # BitPin uses underscore (BTC_IRT) per /api/v1/mkt/markets/
        return _symbol_for_exchange(symbol, use_underscore=True)

    def server_time_ms(self) -> int:
        return int(time.time() * 1000)

    def depth(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        s = self._symbol(symbol)
        out = self._get(f"/api/v1/mth/orderbook/{s}/")
        # Normalize to bids/asks [[price, qty], ...]. BitPin may return "buy"/"sell" or "bids"/"asks".
        bids = out.get("bids") or out.get("buy") or []
        asks = out.get("asks") or out.get("sell") or []
        def rows(arr: list, n: int) -> list[list[str]]:
            result = []
            for x in (arr or [])[:n]:
                if isinstance(x, (list, tuple)) and len(x) >= 2:
                    result.append([str(x[0]), str(x[1])])
                elif isinstance(x, dict):
                    p = x.get("price") or x.get("price_str")
                    q = x.get("amount") or x.get("base_amount") or x.get("quantity") or x.get("size")
                    if p is not None and q is not None:
                        result.append([str(p), str(q)])
            return result
        return {"bids": rows(bids, limit), "asks": rows(asks, limit)}

    def trades(self, symbol: str, limit: int = 1000) -> list[dict]:
        s = self._symbol(symbol)
        out = self._get(f"/api/v1/mth/matches/{s}/")
        # Normalize to [{time (ms), price, qty, isBuyerMaker}, ...].
        raw_list = out if isinstance(out, list) else (out.get("results") or out.get("matches") or out.get("trades") or [])
        lst = []
        for t in raw_list[:limit]:
            time_ms, price, qty, is_buyer_maker = 0, "0", "0", False
            if isinstance(t, dict):
                ts = t.get("time") or t.get("created_at") or t.get("timestamp") or t.get("date")
                if isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        time_ms = int(dt.timestamp() * 1000)
                    except Exception:
                        time_ms = 0
                else:
                    time_ms = int(ts) if ts is not None else 0
                    if time_ms > 0 and time_ms < 1e12:
                        time_ms = time_ms * 1000
                price = str(t.get("price") or t.get("price_str") or "0")
                qty = str(t.get("amount") or t.get("base_amount") or t.get("quantity") or t.get("volume") or "0")
                side = (t.get("side") or t.get("type") or "").upper()
                # Taker buy -> hit ask -> maker was seller -> isBuyerMaker False. Taker sell -> isBuyerMaker True.
                is_buyer_maker = side in ("SELL", "SELLER") or (side == "" and t.get("is_sell"))
            lst.append({"time": time_ms, "price": price, "qty": qty, "isBuyerMaker": is_buyer_maker})
        return lst

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        s = self._symbol(symbol)
        out = self._get("/api/v1/mkt/markets/")
        markets = out if isinstance(out, list) else (out.get("results") or out.get("markets") or [])
        for m in markets:
            if isinstance(m, dict) and (m.get("symbol") or m.get("market")) == s:
                prec = m.get("price_precision")
                try:
                    tick = Decimal(10) ** (-int(prec)) if prec is not None else Decimal("1")
                except Exception:
                    tick = Decimal("1")
                return {"filters": [{"filterType": "PRICE_FILTER", "tickSize": str(tick)}]}
        return {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "1"}]}

    def balances(self, api_token: str | None = None) -> dict[str, dict[str, float]]:
        return {}


class KucoinAdapter:
    """KuCoin: https://www.kucoin.com/docs-new/  Spot REST (api.kucoin.com). Symbol: BASE_QUOTE → BASE-QUOTE."""
    BASE_URL = "https://api.kucoin.com"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        r = requests.get(f"{self.BASE_URL}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        out = r.json()
        if out.get("code") != "200000":
            raise ValueError(out.get("msg", "KuCoin API error"))
        return out.get("data")

    def _symbol(self, symbol: str) -> str:
        # KuCoin uses hyphen: BTC-USDT
        return symbol.replace("_", "-")

    def server_time_ms(self) -> int:
        data = self._get("/api/v1/timestamp")
        return int(data) if isinstance(data, (int, float)) else int(time.time() * 1000)

    def bullet_public_websocket_url(self) -> str | None:
        """Classic spot public websocket: POST /api/v1/bullet-public → endpoint?token=…"""
        try:
            r = requests.post(
                f"{self.BASE_URL}/api/v1/bullet-public",
                json={},
                timeout=15,
            )
            r.raise_for_status()
            body = r.json()
            if str(body.get("code")) != "200000":
                return None
            data = body.get("data") or {}
            token = data.get("token")
            servers = data.get("instanceServers") or []
            if not token or not servers:
                return None
            endpoint = str(servers[0].get("endpoint", "")).rstrip("/")
            if not endpoint:
                return None
            return f"{endpoint}/?token={quote(str(token), safe='')}"
        except Exception:
            return None

    def depth(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        s = self._symbol(symbol)
        size = min(max(limit, 1), 100)  # API supports 1–100 for level2_{size}
        data = self._get(f"/api/v1/market/orderbook/level2_{size}", {"symbol": s})
        bids = [[str(r[0]), str(r[1])] for r in (data.get("bids") or [])[:limit]]
        asks = [[str(r[0]), str(r[1])] for r in (data.get("asks") or [])[:limit]]
        return {"bids": bids, "asks": asks}

    def trades(self, symbol: str, limit: int = 1000) -> list[dict]:
        s = self._symbol(symbol)
        # /api/v1/market/histories returns last 100 only; no pagination for public trades
        data = self._get("/api/v1/market/histories", {"symbol": s})
        lst = []
        for t in (data or [])[:limit]:
            time_raw = t.get("time")
            if time_raw is not None:
                time_ms = int(time_raw)
                if time_ms > 1e15:  # nanoseconds
                    time_ms = time_ms // 1_000_000
                elif time_ms < 1e12:
                    time_ms = time_ms * 1000
            else:
                time_ms = 0
            price = str(t.get("price", "0"))
            qty = str(t.get("size", "0"))
            # side "sell" = taker sold = maker was buyer → isBuyerMaker True
            is_buyer_maker = (t.get("side") or "").lower() == "sell"
            lst.append({"time": time_ms, "price": price, "qty": qty, "isBuyerMaker": is_buyer_maker})
        return lst

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        s = self._symbol(symbol)
        data = self._get(f"/api/v2/symbols/{s}")
        if not data or not isinstance(data, dict):
            return {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "1"}, {"filterType": "LOT_SIZE", "stepSize": "0.0001", "minQty": "0.00001"}]}
        tick = str(data.get("priceIncrement") or "1")
        base_inc = str(data.get("baseIncrement") or "0.00000001")
        base_min = str(data.get("baseMinSize") or base_inc)
        try:
            if Decimal(tick) <= 0:
                tick = "1"
            if Decimal(base_inc) <= 0:
                base_inc = "0.0001"
        except Exception:
            pass
        return {
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": tick},
                {"filterType": "LOT_SIZE", "stepSize": base_inc, "minQty": base_min},
            ]
        }

    def balances(self, api_token: str | None = None) -> dict[str, dict[str, float]]:
        return {}


class BinanceAdapter:
    """Binance Spot: https://developers.binance.com/docs/binance-spot-api-docs  Symbol: BASE_QUOTE → BTCUSDT."""
    BASE_URL = "https://api.binance.com"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        r = requests.get(f"{self.BASE_URL}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    def _symbol(self, symbol: str) -> str:
        # Binance uses no separator, uppercase: BTCUSDT
        return symbol.replace("_", "").upper()

    def server_time_ms(self) -> int:
        out = self._get("/api/v3/time")
        return int(out["serverTime"])

    def depth_snapshot(self, symbol: str, limit: int = 1000) -> dict[str, Any]:
        """Raw REST /api/v3/depth including lastUpdateId (for websocket book sync)."""
        s = self._symbol(symbol)
        limit = min(max(limit, 1), 5000)
        return self._get("/api/v3/depth", {"symbol": s, "limit": limit})

    def depth(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        out = self.depth_snapshot(symbol, limit)
        lim = min(max(limit, 1), 5000)
        bids = [[str(r[0]), str(r[1])] for r in (out.get("bids") or [])[:lim]]
        asks = [[str(r[0]), str(r[1])] for r in (out.get("asks") or [])[:lim]]
        return {"bids": bids, "asks": asks}

    def trades(self, symbol: str, limit: int = 1000) -> list[dict]:
        s = self._symbol(symbol)
        limit = min(max(limit, 1), 1000)
        out = self._get("/api/v3/trades", {"symbol": s, "limit": limit})
        lst = []
        for t in out or []:
            lst.append({
                "time": int(t.get("time", 0)),
                "price": str(t.get("price", "0")),
                "qty": str(t.get("qty", "0")),
                "isBuyerMaker": bool(t.get("isBuyerMaker", False)),
            })
        return lst

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        s = self._symbol(symbol)
        out = self._get("/api/v3/exchangeInfo", {"symbol": s})
        symbols = out.get("symbols") or []
        for sym in symbols:
            if isinstance(sym, dict) and sym.get("symbol") == s:
                filters = sym.get("filters") or []
                # Binance already uses PRICE_FILTER / LOT_SIZE; pass through in our shape
                result_filters = []
                for f in filters:
                    ft = f.get("filterType")
                    if ft == "PRICE_FILTER":
                        result_filters.append({"filterType": "PRICE_FILTER", "tickSize": str(f.get("tickSize", "1"))})
                    elif ft == "LOT_SIZE":
                        result_filters.append({
                            "filterType": "LOT_SIZE",
                            "minQty": str(f.get("minQty", "0.00001")),
                            "stepSize": str(f.get("stepSize", "0.00001")),
                        })
                if result_filters:
                    return {"filters": result_filters}
                break
        return {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"}, {"filterType": "LOT_SIZE", "minQty": "0.00001", "stepSize": "0.00001"}]}

    def balances(self, api_token: str | None = None) -> dict[str, dict[str, float]]:
        return {}


class BitwanaAdapter:
    """Bitwana / YSHM public REST. Domain symbols match wire ids (``USDT_IRT``)."""

    BASE_URL = os.environ.get("BITWANA_API_BASE_URL", "https://bit.yshm.ir").rstrip("/")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        r = requests.get(f"{self.BASE_URL}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict) and "content" in payload:
            code = payload.get("statusCode")
            if code not in (None, 200, 201):
                raise ValueError(payload.get("message") or f"bitwana HTTP {code}")
            return payload.get("content")
        return payload

    def _symbol(self, symbol: str) -> str:
        from mm.exchanges.instrument_wire import build_instrument_wire

        return build_instrument_wire("bitwana", symbol).api_market_symbol

    def server_time_ms(self) -> int:
        return int(time.time() * 1000)

    def depth(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        s = self._symbol(symbol)
        content = self._get(f"/api/1/orders/orderBook/market/{s}/level2") or {}
        if not isinstance(content, dict):
            raise ValueError("bitwana depth error")

        def rows(arr: Any, n: int) -> list[list[str]]:
            out: list[list[str]] = []
            for item in (arr or [])[:n]:
                if isinstance(item, dict):
                    out.append([str(item.get("price", "0")), str(item.get("size", "0"))])
            return out

        return {"bids": rows(content.get("bids"), limit), "asks": rows(content.get("asks"), limit)}

    def trades(self, symbol: str, limit: int = 1000) -> list[dict]:
        s = self._symbol(symbol)
        rows = self._get(f"/api/1/trades/recent/market/{s}") or []
        lst: list[dict] = []
        for item in rows[:limit]:
            if not isinstance(item, dict):
                continue
            taker = str(item.get("takerSide") or "").upper()
            ts = item.get("time")
            try:
                ts_i = int(float(ts))
            except (TypeError, ValueError):
                ts_i = int(time.time())
            if ts_i < 10**12:
                ts_i *= 1000
            lst.append(
                {
                    "time": ts_i,
                    "price": str(item.get("unitPrice", "0")),
                    "qty": str(item.get("amount", "0")),
                    "isBuyerMaker": taker.startswith("S"),
                }
            )
        return lst

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        s = self._symbol(symbol)
        market = self._get(f"/api/1/markets/{s}") or {}
        qty_prec = int(market.get("quantityPrecision") or 2)
        price_prec = int(market.get("userOrderPricePrecision") or 0)
        min_notional = market.get("minimumOrder") or 0
        qty_step = 10 ** (-qty_prec) if qty_prec >= 0 else 0.01
        tick = 10 ** (-price_prec) if price_prec > 0 else 1.0
        return {
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": str(tick)},
                {"filterType": "LOT_SIZE", "minQty": str(qty_step), "stepSize": str(qty_step)},
                {"filterType": "MIN_NOTIONAL", "minNotional": str(min_notional)},
            ]
        }

    def balances(self, api_token: str | None = None) -> dict[str, dict[str, float]]:
        if not api_token:
            return {}
        from mm.exchanges.bitwana.private_rest import get_balances

        return get_balances(api_token)


EXCHANGES: dict[str, ExchangeAdapter] = {
    "tabdeal": TabdealAdapter(),
    "nobitex": NobitexAdapter(),
    "wallex": WallexAdapter(),
    "bitpin": BitPinAdapter(),
    "kucoin": KucoinAdapter(),
    "binance": BinanceAdapter(),
    "bitwana": BitwanaAdapter(),
}


def _adapter(exchange: str) -> ExchangeAdapter:
    ex = exchange.lower().strip()
    if ex not in EXCHANGES:
        raise ValueError(f"Unknown exchange: {exchange}. Choose from: {list(EXCHANGES.keys())}")
    return EXCHANGES[ex]


def get_server_time_ms(exchange: str) -> int:
    """Current server time in milliseconds for the given exchange."""
    return _adapter(exchange).server_time_ms()


def get_tick_size(symbol: str, exchange: str) -> float:
    """Tick size in quote-currency units for the symbol/exchange."""
    return float(_tick_size(symbol, exchange))


def get_lot_size(symbol: str, exchange: str) -> float:
    """Lot size in base-asset units for the symbol/exchange."""
    return float(_lot_size(symbol, exchange))


def get_min_order_amount_tick_lot(symbol: str, exchange: str) -> float:
    """
    Minimum order amount in model cash units (tick * lot), if exchange provides it.
    Returns 0.0 when not available.
    """
    return float(_min_order_amount_tick_lot(symbol, exchange))


def get_min_order_qty_lots(symbol: str, exchange: str) -> float:
    """
    Minimum order quantity in model lot units, if exchange provides LOT_SIZE.minQty.
    Returns 0.0 when not available.
    """
    return float(_min_order_qty_lots(symbol, exchange))


def get_balances(exchange: str, api_token: str | None = None) -> dict[str, dict[str, float]]:
    """Current per-asset balances from exchange adapter private APIs (if supported)."""
    adapter = _adapter(exchange)
    if hasattr(adapter, "balances"):
        try:
            return adapter.balances(api_token=api_token)
        except Exception:
            return {}
    return {}


def list_open_orders(symbol: str, exchange: str, api_token: str | None = None) -> list[dict]:
    """List currently open user orders for symbol (if supported by adapter)."""
    adapter = _adapter(exchange)
    if hasattr(adapter, "list_open_orders"):
        try:
            return adapter.list_open_orders(symbol=symbol, api_token=api_token) or []
        except Exception:
            return []
    return []


def cancel_all_orders(symbol: str, exchange: str, api_token: str | None = None, execution: str = "limit") -> dict[str, Any]:
    """Cancel all active user orders for symbol/execution (if supported)."""
    adapter = _adapter(exchange)
    if hasattr(adapter, "cancel_all_orders"):
        try:
            out = adapter.cancel_all_orders(symbol=symbol, api_token=api_token, execution=execution)
            return out if isinstance(out, dict) else {"status": "failed"}
        except Exception as e:
            return {"status": "failed", "message": str(e)}
    return {"status": "failed", "message": "not supported"}


def place_limit_order_lots(
    symbol: str,
    exchange: str,
    side: str,
    price_ticks: int,
    qty_lots: int,
    api_token: str | None = None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    """
    Place one limit order where inputs are model units:
    - price_ticks in ticks
    - qty_lots in integer lots
    Converts to adapter monetary units using tick_size and lot_size.
    """
    adapter = _adapter(exchange)
    if not hasattr(adapter, "place_limit_order"):
        return {"status": "failed", "message": "not supported"}
    q_lots_i = int(qty_lots)
    p_ticks_i = int(price_ticks)
    if q_lots_i <= 0 or p_ticks_i <= 0:
        return {"status": "failed", "message": "non-positive qty or price"}
    lot = max(1e-12, _lot_size(symbol, exchange))
    tick = max(1e-12, _tick_size(symbol, exchange))
    amount = Decimal(str(q_lots_i)) * Decimal(str(lot))
    price = Decimal(str(p_ticks_i)) * Decimal(str(tick))
    amount_s = format(amount.normalize(), "f")
    price_s = format(price.normalize(), "f")
    try:
        out = adapter.place_limit_order(
            symbol=symbol,
            side=side,
            amount=amount_s,
            price=price_s,
            api_token=api_token,
            client_order_id=client_order_id,
        )
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}


# Cache exchange info per (exchange, symbol) so exchange_info API is called only once.
_exchange_info_cache: dict[tuple[str, str], dict[str, Any]] = {}


def _exchange_info(symbol: str, exchange: str) -> dict[str, Any]:
    """Cached exchange info for symbol; avoids repeated exchange_info API calls."""
    key = (exchange.lower().strip(), symbol)
    if key not in _exchange_info_cache:
        adapter = _adapter(exchange)
        _exchange_info_cache[key] = adapter.exchange_info(symbol) or {}
    return _exchange_info_cache[key]


def get_exchange_info(symbol: str, exchange: str) -> dict[str, Any]:
    """Public read of cached ``exchange_info`` (filters, tick/lot metadata) for a symbol."""

    return dict(_exchange_info(symbol, exchange))


# -----------------------------------------------------------------------------
# Orderbook (X)
# -----------------------------------------------------------------------------


@dataclass
class Level:
    """Price in tick units (int), qty in integer lots, spread_ticks = (mid - price) or (price - mid) as integer."""
    price: int
    qty: int
    spread_ticks: int


@dataclass
class Orderbook:
    """X: orderbook state at some time t. mid and tick_size in tick/currency units as float."""
    t_ms: int
    mid: float
    tick_size: float
    bids: list[Level]
    asks: list[Level]


def normalize_orderbook_from_raw(
    symbol: str,
    exchange: str,
    raw_depth: dict[str, Any],
    *,
    t_ms: int | None = None,
    limit: int | None = None,
) -> Orderbook:
    """
    Convert raw adapter depth payload to normalized Orderbook in tick/lot units.
    """
    bids_raw = (raw_depth or {}).get("bids") or []
    asks_raw = (raw_depth or {}).get("asks") or []
    if limit is not None and limit > 0:
        bids_raw = bids_raw[:limit]
        asks_raw = asks_raw[:limit]
    if not bids_raw or not asks_raw:
        raise ValueError("Empty depth")

    tick_size = _tick_size(symbol, exchange)
    lot_size = _lot_size(symbol, exchange)
    if lot_size <= 0:
        lot_size = 0.0001

    best_bid = float(bids_raw[0][0])
    best_ask = float(asks_raw[0][0])
    mid_currency = (best_bid + best_ask) / 2.0
    mid = mid_currency / tick_size  # mid in tick units (float)

    ob_t_ms = int(t_ms) if t_ms is not None else _adapter(exchange).server_time_ms()
    bids: list[Level] = []
    for p, q in bids_raw:
        price_currency = float(p)
        price_ticks = int(round(price_currency / tick_size))
        spread_ticks = int(round(mid - price_ticks))
        qty_lots = max(0, int(round(float(q) / lot_size)))
        bids.append(Level(price_ticks, qty_lots, spread_ticks))
    asks: list[Level] = []
    for p, q in asks_raw:
        price_currency = float(p)
        price_ticks = int(round(price_currency / tick_size))
        spread_ticks = int(round(price_ticks - mid))
        qty_lots = max(0, int(round(float(q) / lot_size)))
        asks.append(Level(price_ticks, qty_lots, spread_ticks))
    return Orderbook(t_ms=ob_t_ms, mid=mid, tick_size=tick_size, bids=bids, asks=asks)


def _tick_size(symbol: str, exchange: str) -> float:
    info = _exchange_info(symbol, exchange)
    if not info:
        return 1.0
    for f in info.get("filters") or []:
        if f.get("filterType") == "PRICE_FILTER":
            return float(f.get("tickSize", "1"))
    return 1.0


def _lot_size(symbol: str, exchange: str) -> float:
    """
    Base-asset amount represented by one integer lot in our Orderbook / Step volumes.

    Prefer LOT_SIZE **stepSize** (exchange increment / KuCoin baseIncrement): trade and depth
    sizes are multiples of this. Using **minQty** first made volumes ~10× too small vs the
    book when minQty = 10 × stepSize (common on KuCoin/Binance).
    """
    info = _exchange_info(symbol, exchange)
    if not info:
        return 0.0001
    for f in info.get("filters") or []:
        if f.get("filterType") == "LOT_SIZE":
            try:
                step = f.get("stepSize")
                if step is not None and str(step).strip():
                    s = float(step)
                    if s > 0:
                        return s
                min_qty = f.get("minQty")
                if min_qty is not None and str(min_qty).strip():
                    m = float(min_qty)
                    if m > 0:
                        return m
            except Exception:
                pass
            break
    return 0.0001


def _min_order_amount_tick_lot(symbol: str, exchange: str) -> float:
    """
    Minimum order amount in (tick * lot) units.

    If exchange exposes MIN_NOTIONAL in quote currency, convert as:
      min_tick_lot = min_notional / (tick_size * lot_size)

    Fallback: if MIN_NOTIONAL is missing but LOT_SIZE.minQty exists, return
    minQty expressed in model lots as a lower-bound proxy in tick*lot units.
    (Equivalent to assuming price_ticks = 1.)
    """
    info = _exchange_info(symbol, exchange)
    if not info:
        return 0.0
    min_notional = None
    for f in info.get("filters") or []:
        if f.get("filterType") == "MIN_NOTIONAL":
            try:
                mn = float(f.get("minNotional", 0) or 0)
                if mn > 0:
                    min_notional = mn
                    break
            except Exception:
                pass
    if min_notional is None:
        # Some exchanges (including some Nobitex symbols) expose only minQty.
        # Return a non-zero lower-bound proxy so callers expecting a scalar
        # minimum do not silently disable minimum-order constraints.
        return float(_min_order_qty_lots(symbol, exchange))
    tick = max(1e-12, _tick_size(symbol, exchange))
    lot = max(1e-12, _lot_size(symbol, exchange))
    return float(min_notional / (tick * lot))


def _min_order_qty_lots(symbol: str, exchange: str) -> float:
    """
    Minimum order quantity in model lot units (base minQty / lot_size).
    Returns 0.0 when exchange does not expose a positive LOT_SIZE.minQty.
    """
    info = _exchange_info(symbol, exchange)
    if not info:
        return 0.0
    min_qty = None
    for f in info.get("filters") or []:
        if f.get("filterType") == "LOT_SIZE":
            try:
                mq = float(f.get("minQty", 0) or 0)
                if mq > 0:
                    min_qty = mq
                    break
            except Exception:
                pass
    if min_qty is None:
        return 0.0
    lot = max(1e-12, _lot_size(symbol, exchange))
    return float(min_qty / lot)


def get_orderbook(symbol: str, limit: int = 100, exchange: str = "tabdeal") -> Orderbook:
    """
    Fetch current orderbook. Returns X (orderbook state).
    Prices (mid, Level.price) are in tick units; Level.qty is in integer lots.
    tick_size is in currency units so that price_currency = price_ticks * tick_size.
    """
    adapter = _adapter(exchange)
    raw = adapter.depth(symbol, limit=limit)
    t_ms = adapter.server_time_ms()
    return normalize_orderbook_from_raw(
        symbol=symbol,
        exchange=exchange,
        raw_depth=raw,
        t_ms=t_ms,
        limit=limit,
    )


# -----------------------------------------------------------------------------
# Trades in range → fills for an orderbook (Y)
# -----------------------------------------------------------------------------


def normalize_trade_to_tick_lot_units(symbol: str, exchange: str, t: dict) -> dict:
    """
    Convert one raw adapter trade to the same units as Orderbook:
    price → integer tick index, qty → integer lots (base qty / exchange lot_size).
    Use for websocket-ingested trades; REST batch path uses get_trades_in_range which calls this.
    """
    tick_size = _tick_size(symbol, exchange)
    lot_size = _lot_size(symbol, exchange)
    if lot_size <= 0:
        lot_size = 0.0001
    if tick_size <= 0:
        tick_size = 1.0
    price_currency = float(t.get("price", 0) or 0)
    qty_base = float(t.get("qty", 0) or 0)
    price_ticks = int(round(price_currency / tick_size))
    qty_lots = max(0, int(round(qty_base / lot_size)))
    return {
        "time": t.get("time", 0),
        "price": price_ticks,
        "qty": qty_lots,
        "isBuyerMaker": bool(t.get("isBuyerMaker", False)),
    }


def get_trades_in_range(
    symbol: str, t_start_ms: int, t_end_ms: int, limit: int = 1000, exchange: str = "tabdeal"
) -> list[dict]:
    """
    Get last trades from API, keep only those in [t_start_ms, t_end_ms].
    Each returned dict has price in tick units and qty as integer lots.
    """
    batch = _adapter(exchange).trades(symbol, limit=limit)
    out = []
    for t in batch:
        if not (t_start_ms <= int(t.get("time", 0)) <= t_end_ms):
            continue
        out.append(normalize_trade_to_tick_lot_units(symbol, exchange, t))
    out.sort(key=lambda x: x.get("time", 0))
    return out


def fills_for_orderbook(
    ob: Orderbook,
    trades_list: list[dict],
    base_volume: float,
    tick_bucket_width: float = 1.0,
    binary: bool = True,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """
    Y: given orderbook X and trades in the next T, which of X's levels got filled?
    Returns (bid_buckets, ask_buckets). Each bucket = (spread_ticks, filled, M); spread_ticks is int.
    filled = 0/1 if binary else filled_count.
    When ob and trades use tick/lot units (from get_orderbook/get_trades_in_range),
    pass base_volume=1 so M and filled are in lots.
    """
    # Sum filled volume per price by side (isBuyerMaker=true → bid at that price filled)
    bid_filled: dict[int, float] = defaultdict(float)
    ask_filled: dict[int, float] = defaultdict(float)
    for t in trades_list:
        price = int(t["price"])
        qty = float(t["qty"])
        if t.get("isBuyerMaker") is True:
            bid_filled[price] += qty
        else:
            ask_filled[price] += qty

    # Map price → bucket (spread_ticks) from orderbook levels; bucket key is int
    price_to_bucket: dict[int, int] = {}
    for lev in ob.bids:
        b = int(round(lev.spread_ticks / tick_bucket_width) * tick_bucket_width)
        price_to_bucket[lev.price] = b
    for lev in ob.asks:
        b = int(round(lev.spread_ticks / tick_bucket_width) * tick_bucket_width)
        price_to_bucket[lev.price] = b

    # Aggregate volume and filled by bucket (bids)
    bucket_vol: dict[int, float] = defaultdict(float)
    bucket_filled: dict[int, float] = defaultdict(float)
    for lev in ob.bids:
        b = int(round(lev.spread_ticks / tick_bucket_width) * tick_bucket_width)
        bucket_vol[b] += lev.qty
    for price, vol in bid_filled.items():
        b = price_to_bucket.get(price)
        if b is not None:
            bucket_filled[b] += vol

    bid_buckets = []
    for b in sorted(bucket_vol.keys()):
        v = bucket_vol[b]
        f = bucket_filled.get(b, 0.0)
        M = max(0, int(v / base_volume))
        fc = max(0, int(f / base_volume))
        # Cap filled to M: new orders in [0,T] can be filled, so raw fills may exceed snapshot volume
        fc = min(fc, M)
        bid_buckets.append((b, 1 if (binary and fc > 0) else fc, M))

    bucket_vol.clear()
    bucket_filled.clear()
    for lev in ob.asks:
        b = int(round(lev.spread_ticks / tick_bucket_width) * tick_bucket_width)
        bucket_vol[b] += lev.qty
    for price, vol in ask_filled.items():
        b = price_to_bucket.get(price)
        if b is not None:
            bucket_filled[b] += vol
    ask_buckets = []
    for b in sorted(bucket_vol.keys()):
        v = bucket_vol[b]
        f = bucket_filled.get(b, 0.0)
        M = max(0, int(v / base_volume))
        fc = max(0, int(f / base_volume))
        # Cap filled to M: new orders in [0,T] can be filled, so raw fills may exceed snapshot volume
        fc = min(fc, M)
        ask_buckets.append((b, 1 if (binary and fc > 0) else fc, M))

    return bid_buckets, ask_buckets


# -----------------------------------------------------------------------------
# Dataset: list of (X, Y)
# -----------------------------------------------------------------------------


@dataclass
class Sample:
    """One (X, Y) sample."""
    X: Orderbook
    Y_bid: list[tuple[int, int, int]]   # (spread_ticks, filled, M)
    Y_ask: list[tuple[int, int, int]]
    t_start_ms: int
    t_end_ms: int


def collect_dataset(
    symbol: str,
    N: int,
    T_ms: int,
    base_volume: float,
    exchange: str = "tabdeal",
    tick_bucket_width: float = 1.0,
    depth_limit: int = 100,
    binary_y: bool = True,
) -> list[Sample]:
    """
    Collect N (X, Y) samples.
    At t=0 get orderbook (X_0). At t=T get orderbook and trades in [0,T]; Y_0 = fills for X_0. (X_0, Y_0).
    At t=2T get orderbook and trades in [T,2T]; Y_1 = fills for X_1. (X_1, Y_1). etc.
    exchange: tabdeal, nobitex, wallex, bitpin, kucoin, binance.
    get_orderbook/get_trades_in_range return tick/lot units; pass base_volume=1 for fills_for_orderbook.
    """
    adapter = _adapter(exchange)
    dataset: list[Sample] = []
    t_start = adapter.server_time_ms()
    prev_ob = get_orderbook(symbol, limit=depth_limit, exchange=exchange)

    for i in range(N):
        # Wait until next time point
        t_end = t_start + (i + 1) * T_ms
        now = adapter.server_time_ms()
        wait_ms = t_end - now
        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)

        # At time (i+1)*T: get orderbook and trades in [i*T, (i+1)*T]
        ob_now = get_orderbook(symbol, limit=depth_limit, exchange=exchange)
        t_win_start = t_start + i * T_ms
        t_win_end = t_start + (i + 1) * T_ms
        trades_list = get_trades_in_range(symbol, t_win_start, t_win_end, exchange=exchange)

        # Y = fills for prev orderbook (X) in this window
        y_bid, y_ask = fills_for_orderbook(
            prev_ob, trades_list, base_volume,
            tick_bucket_width=tick_bucket_width,
            binary=binary_y,
        )
        dataset.append(Sample(
            X=prev_ob,
            Y_bid=y_bid,
            Y_ask=y_ask,
            t_start_ms=t_win_start,
            t_end_ms=t_win_end,
        ))
        prev_ob = ob_now

    return dataset


def dataset_to_rows(
    dataset: list[Sample],
    side: str = "bid",
) -> list[tuple[int, int, int]]:
    """
    Flatten dataset to rows (spread_ticks, filled, M), one row per (sample, spread bucket).
    empirical_fill_rates then sums filled and M across all rows per spread bucket,
    so fill rate = cumulated fill count / cumulated M over all time steps.
    """
    rows: list[tuple[int, int, int]] = []
    for s in dataset:
        buckets = s.Y_bid if side == "bid" else s.Y_ask
        for spread, filled, M in buckets:
            if M > 0:
                rows.append((spread, filled, M))
    return rows


# -----------------------------------------------------------------------------
# Fill probability from count data (non-parametric)
# -----------------------------------------------------------------------------


@dataclass
class FillRateBucket:
    """Empirical fill rate at a spread bucket."""
    spread_center: float
    p_fill: float
    total_filled: int
    total_M: int


def empirical_fill_rates(
    rows: list[tuple[int, int, int]],
    spread_bucket_width: float = 1.0,
    min_total_M: int = 1,
) -> list[FillRateBucket]:
    """
    Per spread bucket: fill rate = (cumulated fill count over all time steps)
                        / (cumulated M over all time steps).

    Each row is (spread_ticks, filled, M) from one sample. We bin rows by spread
    (spread_bucket_width), then sum filled and M across all rows in each bin.
    p_fill = sum(filled) / sum(M) per bin — not an average of per-sample rates.
    """
    # Accumulate filled and M across all rows (all time steps) per spread bucket
    cumulated_filled: dict[float, int] = defaultdict(int)
    cumulated_M: dict[float, int] = defaultdict(int)
    for spread, filled, M in rows:
        center = round(spread / spread_bucket_width) * spread_bucket_width
        cumulated_filled[center] += filled
        cumulated_M[center] += M
    out: list[FillRateBucket] = []
    for center in sorted(cumulated_M.keys()):
        total_M = cumulated_M[center]
        if total_M < min_total_M:
            continue
        total_filled = cumulated_filled[center]
        p = total_filled / total_M  # fill rate = cumulated fill / cumulated M
        out.append(FillRateBucket(
            spread_center=center,
            p_fill=p,
            total_filled=total_filled,
            total_M=total_M,
        ))
    return out


def empirical_fill_rates_shrinkage(
    rows: list[tuple[int, int, int]],
    spread_bucket_width: float = 1.0,
    min_total_M: int = 1,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> list[FillRateBucket]:
    """
    Same as empirical_fill_rates but with Beta(α,β) shrinkage toward prior mean.
    Posterior P(fill) = (α + total_filled) / (α + β + total_M).
    Prior mean = α/(α+β); use e.g. α=β=1 for 0.5, or set from global fill rate.
    """
    bucket_filled = defaultdict(int)
    bucket_M = defaultdict(int)
    for spread, filled, M in rows:
        center = round(spread / spread_bucket_width) * spread_bucket_width
        bucket_filled[center] += filled
        bucket_M[center] += M
    out = []
    for center in sorted(bucket_M.keys()):
        total_M = bucket_M[center]
        if total_M < min_total_M:
            continue
        total_filled = bucket_filled[center]
        p = (prior_alpha + total_filled) / (prior_alpha + prior_beta + total_M)
        out.append(FillRateBucket(
            spread_center=center,
            p_fill=p,
            total_filled=total_filled,
            total_M=total_M,
        ))
    return out


def fill_rate_kernel_smooth(
    rows: list[tuple[int, int, int]],
    spread_points: list[float],
    bandwidth: float = 1.5,
) -> list[float]:
    """
    Kernel-smoothed P(fill | spread) at given spread_points.
    Uses Gaussian kernel; each row contributes (filled/M) weighted by K((s - spread)/b).
    P_fill(s) = Σ_i w_i * (filled_i/M_i) / Σ_i w_i.
    """
    import math
    def gaussian(z: float) -> float:
        return math.exp(-0.5 * (z / bandwidth) ** 2)

    result: list[float] = []
    for s in spread_points:
        w_sum = 0.0
        weighted_rate = 0.0
        for spread, filled, M in rows:
            if M <= 0:
                continue
            w = gaussian(s - spread)
            w_sum += w
            weighted_rate += w * (filled / M)
        result.append(weighted_rate / w_sum if w_sum > 0 else 0.0)
    return result


# -----------------------------------------------------------------------------
# Animated empirical fill rate chart (spread vs fill rate, updated every T_ms)
# -----------------------------------------------------------------------------


def animate_empirical_fill_rates(
    symbol: str,
    N: int,
    T_ms: int,
    base_volume: float,
    exchange: str = "tabdeal",
    tick_bucket_width: float = 1.0,
    spread_bucket_width: float = 1.0,
    depth_limit: int = 100,
    binary_y: bool = True,
    min_total_M: int = 1,
) -> list[Sample]:
    """
    Collect N (X, Y) samples at T_ms intervals and animate a chart of empirical
    fill rate vs spread. X-axis: spread (negative = bid, positive = ask).
    Y-axis: fill rate. Chart updates after each new sample.
    Returns the full dataset when done.
    """
    import matplotlib.pyplot as plt

    adapter = _adapter(exchange)
    dataset: list[Sample] = []
    t_start = adapter.server_time_ms()
    prev_ob = get_orderbook(symbol, limit=depth_limit, exchange=exchange)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Spread (negative = bid, positive = ask)")
    ax.set_ylabel("Fill rate")
    ax.set_title(f"Empirical fill rate vs spread — {exchange} {symbol} (updating every T={T_ms} ms, N={N})")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-20, 20)  # will be adjusted when we have data
    fig.tight_layout()
    plt.ion()
    fig.show()

    for i in range(N):
        # Wait until next time point
        t_end = t_start + (i + 1) * T_ms
        now = adapter.server_time_ms()
        wait_ms = t_end - now
        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)

        # At time (i+1)*T: get orderbook and trades in [i*T, (i+1)*T]
        ob_now = get_orderbook(symbol, limit=depth_limit, exchange=exchange)
        t_win_start = t_start + i * T_ms
        t_win_end = t_start + (i + 1) * T_ms
        trades_list = get_trades_in_range(symbol, t_win_start, t_win_end, exchange=exchange)

        y_bid, y_ask = fills_for_orderbook(
            prev_ob, trades_list, base_volume,
            tick_bucket_width=tick_bucket_width,
            binary=binary_y,
        )
        dataset.append(Sample(
            X=prev_ob,
            Y_bid=y_bid,
            Y_ask=y_ask,
            t_start_ms=t_win_start,
            t_end_ms=t_win_end,
        ))
        prev_ob = ob_now

        # Recompute empirical fill rates on accumulated dataset
        bid_rows = dataset_to_rows(dataset, side="bid")
        ask_rows = dataset_to_rows(dataset, side="ask")
        bid_buckets = empirical_fill_rates(bid_rows, spread_bucket_width=spread_bucket_width, min_total_M=min_total_M)
        ask_buckets = empirical_fill_rates(ask_rows, spread_bucket_width=spread_bucket_width, min_total_M=min_total_M)

        # X-axis: negative for bid, positive for ask
        x_bid = [-b.spread_center for b in bid_buckets]
        y_bid_rates = [b.p_fill for b in bid_buckets]
        x_ask = [b.spread_center for b in ask_buckets]
        y_ask_rates = [b.p_fill for b in ask_buckets]

        ax.clear()
        ax.set_xlabel("Spread (negative = bid, positive = ask)")
        ax.set_ylabel("Fill rate")
        ax.set_title(f"Empirical fill rate vs spread — {exchange} {symbol} (sample {len(dataset)}/{N}, T={T_ms} ms)")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="-")
        ax.set_ylim(0, 1.05)

        if x_bid or x_ask:
            if x_bid:
                ax.bar(x_bid, y_bid_rates, width=spread_bucket_width * 0.8, color="steelblue", alpha=0.8, label="Bid", align="center")
            if x_ask:
                ax.bar(x_ask, y_ask_rates, width=spread_bucket_width * 0.8, color="darkorange", alpha=0.8, label="Ask", align="center")
            all_x = x_bid + x_ask
            x_min, x_max = min(all_x), max(all_x)
            margin = max((x_max - x_min) * 0.05, 1.0)
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.legend(loc="upper right")
        else:
            ax.set_xlim(-20, 20)

        fig.canvas.draw_idle()
        plt.pause(0.05)

    plt.ioff()
    plt.show()
    return dataset
