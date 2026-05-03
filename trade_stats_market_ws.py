"""
Live market websocket: URLs, subscribe payloads, and JSON parsing per exchange.

Stateful pieces (Binance merged book, KuCoin classic vs legacy) live in session objects.
Used by ``mm_runtime.market_data_providers.WebSocketBufferedMarketDataProvider``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from trade_stats import EXCHANGES


@dataclass
class WsMarketDispatch:
    """Normalized trades + orderbook snapshots to pass to normalize_* helpers."""

    trades: list[dict] = field(default_factory=list)
    orderbooks: list[tuple[dict, int | None]] = field(default_factory=list)
    private_trades: list[dict] = field(default_factory=list)
    private_orders: list[dict] = field(default_factory=list)


def kucoin_ws_trade_time_ms(d: dict) -> int:
    raw = d.get("E") or d.get("time") or d.get("ts") or d.get("T")
    if raw is None:
        return int(time.time() * 1000)
    try:
        t = int(float(str(raw).replace(",", "")))
    except (TypeError, ValueError):
        return int(time.time() * 1000)
    if t >= 10**15:
        return t // 1_000_000
    if t >= 10**12:
        return t
    if t >= 10**9:
        return t * 1000
    return t


def kucoin_ping_pong_json(payload: dict) -> str | None:
    if payload.get("type") == "ping":
        return json.dumps({"type": "pong", "id": payload.get("id")})
    return None


def _parse_iso_time_ms(raw: Any) -> int:
    if raw is None:
        return int(time.time() * 1000)
    s = str(raw).strip()
    if not s:
        return int(time.time() * 1000)
    try:
        # Wallex trade timestamps are ISO-8601 like: 2025-06-01T09:33:41Z
        from datetime import datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        pass
    try:
        t = int(float(s.replace(",", "")))
    except (TypeError, ValueError):
        return int(time.time() * 1000)
    if t >= 10**15:
        return t // 1_000_000
    if t >= 10**12:
        return t
    if t >= 10**9:
        return t * 1000
    return t


class BinanceMarketWsSession:
    """Combined diff depth + trade; REST snapshot + merge depthUpdate (see Binance book sync)."""

    def __init__(self, symbol: str, depth_limit: int):
        self.symbol = symbol
        self.depth_limit = max(1, depth_limit)
        self._adapter = EXCHANGES["binance"]
        self._bids: dict[str, float] = {}
        self._asks: dict[str, float] = {}
        self._rest_last_update_id: int | None = None

    def websocket_url(self) -> str:
        sym = self.symbol.replace("_", "").lower()
        return (
            "wss://stream.binance.com:9443/stream?streams="
            f"{sym}@depth@100ms/{sym}@trade"
        )

    def on_connect_open(self, _ws: Any) -> WsMarketDispatch:
        out = WsMarketDispatch()
        self._bids.clear()
        self._asks.clear()
        self._rest_last_update_id = None
        lim = min(5000, max(50, self.depth_limit * 10))
        try:
            raw = self._adapter.depth_snapshot(self.symbol, lim)
        except Exception:
            return out
        self._bids = {str(p): float(q) for p, q in (raw.get("bids") or [])}
        self._asks = {str(p): float(q) for p, q in (raw.get("asks") or [])}
        self._rest_last_update_id = int(raw.get("lastUpdateId") or 0)
        t_ms = int(time.time() * 1000)
        raw_ob = self._levels_to_raw()
        if raw_ob:
            out.orderbooks.append((raw_ob, t_ms))
        return out

    def _levels_to_raw(self) -> dict[str, Any] | None:
        if not self._bids or not self._asks:
            return None
        bid_keys = sorted(self._bids.keys(), key=lambda x: float(x), reverse=True)[: self.depth_limit]
        ask_keys = sorted(self._asks.keys(), key=lambda x: float(x))[: self.depth_limit]
        bids_raw = [[k, str(self._bids[k])] for k in bid_keys]
        asks_raw = [[k, str(self._asks[k])] for k in ask_keys]
        return {"bids": bids_raw, "asks": asks_raw}

    def _apply_depth_and_emit(self, data: dict) -> WsMarketDispatch:
        out = WsMarketDispatch()
        u_final = int(data.get("u", 0))
        if self._rest_last_update_id is not None and u_final <= self._rest_last_update_id:
            return out
        self._rest_last_update_id = None

        for p, q in data.get("b", []) or []:
            ps, qs = str(p), str(q)
            qf = float(qs)
            if qf == 0.0:
                self._bids.pop(ps, None)
            else:
                self._bids[ps] = qf
        for p, q in data.get("a", []) or []:
            ps, qs = str(p), str(q)
            qf = float(qs)
            if qf == 0.0:
                self._asks.pop(ps, None)
            else:
                self._asks[ps] = qf

        t_ms = int(data.get("E", time.time() * 1000))
        raw_ob = self._levels_to_raw()
        if raw_ob:
            out.orderbooks.append((raw_ob, t_ms))
        return out

    def handle_json(self, payload: dict) -> WsMarketDispatch:
        out = WsMarketDispatch()
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            return out
        ev = data.get("e")
        if ev in ("trade", "aggTrade"):
            out.trades.append({
                "time": int(data.get("T", data.get("E", 0))),
                "price": str(data.get("p", "0")),
                "qty": str(data.get("q", "0")),
                "isBuyerMaker": bool(data.get("m", False)),
            })
        if ev == "depthUpdate":
            return self._apply_depth_and_emit(data)
        if "bids" in data and "asks" in data:
            out.orderbooks.append((data, int(time.time() * 1000)))
        return out


class KucoinMarketWsSession:
    def __init__(self, symbol: str, depth_limit: int):
        self.symbol = symbol
        self.depth_limit = max(1, depth_limit)
        self._adapter = EXCHANGES["kucoin"]
        self.use_classic = False

    def websocket_url(self) -> str | None:
        bullet = self._adapter.bullet_public_websocket_url()
        if bullet:
            self.use_classic = True
            return bullet
        self.use_classic = False
        return "wss://x-push-spot.kucoin.com"

    def on_connect_open(self, ws: Any) -> WsMarketDispatch:
        symbol = self.symbol.replace("_", "-")
        base = int(time.time() * 1000)
        if self.use_classic:
            depth_tier = "50" if self.depth_limit >= 50 else "5"
            ws.send(json.dumps({
                "id": str(base),
                "type": "subscribe",
                "topic": f"/market/match:{symbol}",
                "privateChannel": False,
                "response": True,
            }))
            ws.send(json.dumps({
                "id": str(base + 1),
                "type": "subscribe",
                "topic": f"/spotMarket/level2Depth{depth_tier}:{symbol}",
                "privateChannel": False,
                "response": True,
            }))
        else:
            ws.send(json.dumps({
                "id": str(base),
                "action": "SUBSCRIBE",
                "channel": "obu",
                "tradeType": "SPOT",
                "symbol": symbol,
                "depth": "50",
            }))
            ws.send(json.dumps({
                "id": str(base + 1),
                "action": "SUBSCRIBE",
                "channel": "trade",
                "tradeType": "SPOT",
                "symbol": symbol,
            }))
        return WsMarketDispatch()

    def handle_json(self, payload: dict) -> WsMarketDispatch:
        out = WsMarketDispatch()
        ptype = payload.get("type")
        if ptype in ("ack", "welcome", "receipt", "error"):
            return out
        if ptype == "message":
            topic = str(payload.get("topic", ""))
            data = payload.get("data")
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    data = None
            if not isinstance(data, dict):
                return out
            if "/market/match:" in topic:
                side = str(data.get("side", "")).lower()
                out.trades.append({
                    "time": kucoin_ws_trade_time_ms(data),
                    "price": str(data.get("p", data.get("price", "0"))),
                    "qty": str(data.get("size", data.get("q", data.get("volume", "0")))),
                    "isBuyerMaker": side == "sell",
                })
                return out
            if "level2Depth" in topic or "/market/level2" in topic or "/spotMarket/level2" in topic:
                bids = data.get("bids")
                asks = data.get("asks")
                if isinstance(bids, list) and isinstance(asks, list) and (bids or asks):
                    ts = data.get("timestamp") or data.get("time")
                    t_ms = int(time.time() * 1000)
                    if ts is not None:
                        try:
                            tv = int(float(str(ts)))
                            if tv > 10**15:
                                tv //= 1_000_000
                            t_ms = tv if tv > 10**12 else int(tv * 1000)
                        except Exception:
                            pass
                    out.orderbooks.append(({"bids": bids, "asks": asks}, t_ms))
                return out
            return out

        t = str(payload.get("T", "") or "")
        d = payload.get("d")
        if not isinstance(d, dict) or not d:
            inner = payload.get("data")
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except Exception:
                    inner = None
            if isinstance(inner, dict):
                d = inner
        if not isinstance(d, dict):
            return out
        subj = str(payload.get("subject", "")).lower()
        is_trade = t == "trade.SPOT" or (subj.startswith("trade.") and "ticker" not in subj)
        if is_trade:
            side = str(d.get("S", d.get("side", ""))).lower()
            out.trades.append({
                "time": kucoin_ws_trade_time_ms(d),
                "price": str(d.get("p", d.get("price", "0"))),
                "qty": str(d.get("size", d.get("q", d.get("volume", "0")))),
                "isBuyerMaker": side == "sell",
            })
        elif t == "obu.SPOT" and ("a" in d or "b" in d):
            out.orderbooks.append((
                {"bids": d.get("b", []), "asks": d.get("a", [])},
                int(d.get("M", time.time_ns()) // 1_000_000),
            ))
        return out


class TabdealMarketWsSession:
    def __init__(self, symbol: str, depth_limit: int):
        self.symbol = symbol
        self.depth_limit = max(1, depth_limit)

    def websocket_url(self) -> str:
        return "wss://api1.tabdeal.org/stream/"

    def on_connect_open(self, ws: Any) -> WsMarketDispatch:
        stream = f"{self.symbol.replace('_', '').lower()}@depth@2000ms"
        ws.send(json.dumps({"method": "SUBSCRIBE", "params": [stream], "id": 1}))
        return WsMarketDispatch()

    def handle_json(self, payload: dict) -> WsMarketDispatch:
        out = WsMarketDispatch()
        data = payload.get("data")
        if isinstance(data, dict) and ("b" in data or "a" in data):
            out.orderbooks.append((
                {"bids": data.get("b", []), "asks": data.get("a", [])},
                int(data.get("E", time.time() * 1000)),
            ))
        return out


class NobitexMarketWsSession:
    def __init__(self, symbol: str, depth_limit: int, private_api_token: str | None = None):
        self.symbol = symbol
        self.depth_limit = max(1, depth_limit)
        self._adapter = EXCHANGES["nobitex"]
        self._private_api_token = private_api_token or os.environ.get("NOBITEX_API_TOKEN")
        self._connection_token: str | None = None
        self._websocket_auth_param: str | None = None

    def websocket_url(self) -> str:
        return "wss://ws.nobitex.ir/connection/websocket"

    def on_connect_open(self, ws: Any) -> WsMarketDispatch:
        symbol = self.symbol.replace("_", "").upper()
        connect_payload: dict[str, Any] = {"connect": {}, "id": 1}
        if self._private_api_token:
            self._connection_token = self._adapter.websocket_connection_token(self._private_api_token)
            self._websocket_auth_param = self._adapter.websocket_auth_param(self._private_api_token)
            if self._connection_token:
                connect_payload["connect"] = {"token": self._connection_token}
        ws.send(json.dumps(connect_payload))
        ws.send(json.dumps({"id": 2, "subscribe": {"channel": f"public:orderbook-{symbol}"}}))
        ws.send(json.dumps({"id": 3, "subscribe": {"channel": f"public:trades-{symbol}"}}))
        if self._websocket_auth_param:
            ws.send(json.dumps({
                "id": 4,
                "subscribe": {"channel": f"private:orders#{self._websocket_auth_param}"},
            }))
            ws.send(json.dumps({
                "id": 5,
                "subscribe": {"channel": f"private:trades#{self._websocket_auth_param}"},
            }))
        return WsMarketDispatch()

    def handle_json(self, payload: dict) -> WsMarketDispatch:
        out = WsMarketDispatch()
        push = payload.get("push")
        if not isinstance(push, dict):
            return out
        channel = str(push.get("channel", ""))
        pub = push.get("pub") or {}
        data = pub.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = None
        elif isinstance(data, dict):
            pass
        if not isinstance(data, dict):
            alt = push.get("data")
            if isinstance(alt, dict):
                data = alt
            elif isinstance(alt, str):
                try:
                    data = json.loads(alt)
                except Exception:
                    data = None
        if not isinstance(data, dict):
            return out
        if channel.startswith("public:orderbook-"):
            out.orderbooks.append((
                {"bids": data.get("bids", []), "asks": data.get("asks", [])},
                int(data.get("lastUpdate", time.time() * 1000)),
            ))
        elif channel.startswith("public:trades-"):
            side = str(data.get("type", "")).lower()
            out.trades.append({
                "time": int(data.get("time", 0)),
                "price": str(data.get("price", "0")),
                "qty": str(data.get("volume", "0")),
                "isBuyerMaker": side == "sell",
            })
        elif channel.startswith("private:trades#"):
            if hasattr(self._adapter, "normalize_private_trade_event"):
                ev = self._adapter.normalize_private_trade_event(data, source="ws")
                if isinstance(ev, dict):
                    out.private_trades.append(ev)
        elif channel.startswith("private:orders#"):
            if hasattr(self._adapter, "normalize_private_order_event"):
                ev = self._adapter.normalize_private_order_event(data)
                if isinstance(ev, dict):
                    out.private_orders.append(ev)
        return out


class WallexMarketWsSession:
    def __init__(self, symbol: str, depth_limit: int):
        self.symbol = symbol
        self.depth_limit = max(1, depth_limit)
        self._adapter = EXCHANGES["wallex"]
        self._buy_depth: list[list[str]] = []
        self._sell_depth: list[list[str]] = []

    def _symbol(self) -> str:
        # Wallex uses TMN instead of IRT (e.g. BTC_IRT -> BTCTMN).
        s = self.symbol.replace("_", "").upper()
        if s.endswith("IRT"):
            return s[:-3] + "TMN"
        return s

    def websocket_url(self) -> str:
        return "wss://api.wallex.ir/ws"

    def on_connect_open(self, ws: Any) -> WsMarketDispatch:
        out = WsMarketDispatch()
        market = self._symbol()
        self._buy_depth = []
        self._sell_depth = []
        # Seed both sides from REST snapshot so we can emit usable books even if one
        # websocket side is temporarily quiet.
        try:
            raw = self._adapter.depth(self.symbol, self.depth_limit)
            bids = raw.get("bids")
            asks = raw.get("asks")
            if isinstance(bids, list):
                self._buy_depth = [[str(p), str(q)] for p, q in bids[: self.depth_limit]]
            if isinstance(asks, list):
                self._sell_depth = [[str(p), str(q)] for p, q in asks[: self.depth_limit]]
            self._emit_orderbook_if_ready(out, int(time.time() * 1000))
        except Exception:
            pass
        ws.send(json.dumps(["subscribe", {"channel": f"{market}@buyDepth"}]))
        ws.send(json.dumps(["subscribe", {"channel": f"{market}@sellDepth"}]))
        ws.send(json.dumps(["subscribe", {"channel": f"{market}@trade"}]))
        return out

    def _emit_orderbook_if_ready(self, out: WsMarketDispatch, t_ms: int) -> None:
        if not self._buy_depth or not self._sell_depth:
            return
        out.orderbooks.append(({
            "bids": list(self._buy_depth),
            "asks": list(self._sell_depth),
        }, t_ms))

    @staticmethod
    def _rows_from_depth_payload(payload: Any, *, descending: bool) -> list[list[str]]:
        if not isinstance(payload, list):
            return []
        rows: list[list[str]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            p = item.get("price")
            q = item.get("quantity")
            if p is None or q is None:
                continue
            rows.append([str(p), str(q)])
        rows.sort(key=lambda r: float(r[0]), reverse=descending)
        return rows

    def handle_json(self, payload: Any) -> WsMarketDispatch:
        out = WsMarketDispatch()
        # Wallex public streams are array payloads: ["CHANNEL", <data>]
        if not isinstance(payload, list) or len(payload) < 2:
            return out
        channel = str(payload[0])
        data = payload[1]
        t_ms = int(time.time() * 1000)
        if channel.endswith("@trade") and isinstance(data, dict):
            out.trades.append({
                "time": _parse_iso_time_ms(data.get("timestamp")),
                "price": str(data.get("price", "0")),
                "qty": str(data.get("quantity", "0")),
                # isBuyOrder True => taker buy => maker sell => isBuyerMaker False
                "isBuyerMaker": not bool(data.get("isBuyOrder", False)),
            })
            return out
        if channel.endswith("@buyDepth"):
            self._buy_depth = self._rows_from_depth_payload(data, descending=True)[: self.depth_limit]
            self._emit_orderbook_if_ready(out, t_ms)
            return out
        if channel.endswith("@sellDepth"):
            self._sell_depth = self._rows_from_depth_payload(data, descending=False)[: self.depth_limit]
            self._emit_orderbook_if_ready(out, t_ms)
            return out
        return out


def create_market_ws_session(
    exchange: str,
    symbol: str,
    depth_limit: int,
    *,
    private_api_token: str | None = None,
) -> Any | None:
    ex = exchange.lower().strip()
    if ex == "binance":
        return BinanceMarketWsSession(symbol, depth_limit)
    if ex == "kucoin":
        return KucoinMarketWsSession(symbol, depth_limit)
    if ex == "tabdeal":
        return TabdealMarketWsSession(symbol, depth_limit)
    if ex == "nobitex":
        return NobitexMarketWsSession(symbol, depth_limit, private_api_token=private_api_token)
    if ex == "wallex":
        return WallexMarketWsSession(symbol, depth_limit)
    return None
