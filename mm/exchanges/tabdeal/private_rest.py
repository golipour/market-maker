"""Synchronous Tabdeal private REST helpers."""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests

BASE_URL = "https://api1.tabdeal.org"


@dataclass(frozen=True, slots=True)
class TabdealApiKeyAuth:
    api_key: str
    api_secret: str


def api_symbol(symbol: str) -> str:
    from mm.exchanges.instrument_wire import build_instrument_wire

    return build_instrument_wire("tabdeal", symbol).api_market_symbol


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _require_auth(auth: TabdealApiKeyAuth | None) -> TabdealApiKeyAuth | None:
    if isinstance(auth, TabdealApiKeyAuth) and auth.api_key and auth.api_secret:
        return auth
    return None


def _signed_params(params: dict[str, Any], auth: TabdealApiKeyAuth) -> dict[str, Any]:
    out = {k: v for k, v in (params or {}).items() if v is not None}
    out["timestamp"] = int(time.time() * 1000)
    query = urlencode(out)
    out["signature"] = hmac.new(auth.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    return out


def _request_auth(method: str, path: str, auth: TabdealApiKeyAuth, params: dict[str, Any] | None = None) -> requests.Response:
    headers = {"X-MBX-APIKEY": auth.api_key, "Accept": "application/json"}
    signed = _signed_params(params or {}, auth)
    return requests.request(method.upper(), f"{BASE_URL}{path}", params=signed, headers=headers, timeout=15)


def _user_request(method: str, path: str, api_key: str, params: dict[str, Any] | None = None) -> requests.Response:
    headers = {"X-MBX-APIKEY": api_key, "Accept": "application/json"}
    return requests.request(method.upper(), f"{BASE_URL}{path}", params=params or {}, headers=headers, timeout=15)


def _json_or_raise(response: requests.Response, path: str) -> Any:
    if not response.ok:
        try:
            detail: Any = response.json()
        except Exception:
            detail = (response.text or "")[:4000]
        raise RuntimeError(f"Tabdeal HTTP {response.status_code} {path}: {detail}")
    return response.json() if response.content else {}


def get_balances(auth: TabdealApiKeyAuth | None) -> dict[str, dict[str, float]]:
    auth = _require_auth(auth)
    if not auth:
        return {}
    rows: list[Any] = []
    try:
        out = _json_or_raise(_request_auth("GET", "/r/api/v1/account", auth), "/r/api/v1/account")
        if isinstance(out, dict) and isinstance(out.get("balances"), list):
            rows = list(out.get("balances") or [])
    except Exception:
        rows = []
    if not rows:
        out = _json_or_raise(_request_auth("GET", "/r/api/v1/asset/get-funding-asset", auth), "/r/api/v1/asset/get-funding-asset")
        rows = out if isinstance(out, list) else [out] if isinstance(out, dict) and out.get("asset") else []
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset = str(row.get("asset") or row.get("currency") or "").upper()
        if not asset:
            continue
        free = _safe_float(row.get("free", 0.0))
        locked = _safe_float(row.get("freeze", row.get("locked", 0.0)))
        result[asset] = {"free": free, "locked": locked, "total": free + locked}
    from mm.exchanges.instrument_wire import merge_wallet_balances_raw

    return merge_wallet_balances_raw("tabdeal", result)


def list_open_orders(symbol: str, auth: TabdealApiKeyAuth | None) -> list[dict]:
    auth = _require_auth(auth)
    if not auth:
        return []
    out = _json_or_raise(
        _request_auth("GET", "/r/api/v1/openOrders", auth, {"symbol": api_symbol(symbol)}),
        "/r/api/v1/openOrders",
    )
    if not isinstance(out, list):
        raise RuntimeError("Tabdeal open-order fetch failed: response is not a list")
    return [o for o in out if isinstance(o, dict)]


def list_filled_orders(symbol: str, auth: TabdealApiKeyAuth | None) -> list[dict]:
    auth = _require_auth(auth)
    if not auth:
        return []
    out = _json_or_raise(
        _request_auth("GET", "/r/api/v1/allOrders", auth, {"symbol": api_symbol(symbol), "limit": 100}),
        "/r/api/v1/allOrders",
    )
    if not isinstance(out, list):
        raise RuntimeError("Tabdeal order-history fetch failed: response is not a list")
    return [
        o for o in out
        if isinstance(o, dict)
        and (str(o.get("status", "")).upper() in {"FILLED", "PARTIALLY_FILLED"} or _safe_float(o.get("executedQty")) > 0.0)
    ]


def place_order(
    symbol: str,
    side: str,
    order_type: str,
    amount: str | float,
    auth: TabdealApiKeyAuth | None,
    *,
    price: str | float | int | None = None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    auth = _require_auth(auth)
    if not auth:
        return {"status": "failed", "message": "missing api key/secret"}
    from mm.exchanges.instrument_wire import build_instrument_wire

    wire = build_instrument_wire("tabdeal", symbol)
    params: dict[str, Any] = {
        "symbol": wire.api_market_symbol,
        "side": str(side).upper(),
        "type": str(order_type).upper(),
        "quantity": str(amount),
    }
    if price is not None:
        params["price"] = str(wire.scale_limit_price_to_api(_safe_float(price)))
    if client_order_id:
        params["newClientOrderId"] = str(client_order_id)[:64]
    try:
        out = _json_or_raise(_request_auth("POST", "/api/v1/order", auth, params), "/api/v1/order")
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}


def place_market_order(symbol: str, side: str, amount: str | float, auth: TabdealApiKeyAuth | None, client_order_id: str | None = None) -> dict[str, Any]:
    return place_order(symbol, side, "MARKET", amount, auth, client_order_id=client_order_id)


def place_limit_order(symbol: str, side: str, amount: str | float, price: str | float | int, auth: TabdealApiKeyAuth | None, client_order_id: str | None = None) -> dict[str, Any]:
    return place_order(symbol, side, "LIMIT", amount, auth, price=price, client_order_id=client_order_id)


def cancel_order(symbol: str, order_id: str, auth: TabdealApiKeyAuth | None) -> dict[str, Any]:
    auth = _require_auth(auth)
    if not auth:
        return {"status": "failed", "message": "missing api key/secret"}
    params = {"symbol": api_symbol(symbol)}
    if str(order_id).isdigit():
        params["orderId"] = str(order_id)
    else:
        params["origClientOrderId"] = str(order_id)
    try:
        out = _json_or_raise(_request_auth("DELETE", "/api/v1/order", auth, params), "/api/v1/order")
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}


def create_listen_key(api_key: str | None) -> str:
    if not api_key:
        return ""
    out = _json_or_raise(_user_request("POST", "/api/v1/userDataStream", api_key), "/api/v1/userDataStream")
    return str(out.get("listenKey") or "") if isinstance(out, dict) else ""


def keepalive_listen_key(api_key: str | None, listen_key: str) -> None:
    if api_key and listen_key:
        _json_or_raise(_user_request("PUT", "/api/v1/userDataStream", api_key, {"listenKey": listen_key}), "/api/v1/userDataStream")


def close_listen_key(api_key: str | None, listen_key: str) -> None:
    if api_key and listen_key:
        _json_or_raise(_user_request("DELETE", "/api/v1/userDataStream", api_key, {"listenKey": listen_key}), "/api/v1/userDataStream")
