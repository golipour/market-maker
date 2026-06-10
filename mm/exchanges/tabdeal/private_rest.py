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


class TabdealTransientError(RuntimeError):
    pass


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
    verb = method.upper()
    url = f"{BASE_URL}{path}"
    # Tabdeal signs query-string params; POST/PUT must send them as form body (not ?params).
    if verb in {"POST", "PUT"}:
        return requests.request(verb, url, data=signed, headers=headers, timeout=15)
    return requests.request(verb, url, params=signed, headers=headers, timeout=15)


def _user_request(method: str, path: str, api_key: str, params: dict[str, Any] | None = None) -> requests.Response:
    headers = {"X-MBX-APIKEY": api_key, "Accept": "application/json"}
    return requests.request(method.upper(), f"{BASE_URL}{path}", params=params or {}, headers=headers, timeout=15)


def _json_or_raise(response: requests.Response, path: str) -> Any:
    if not response.ok:
        try:
            detail: Any = response.json()
        except Exception:
            detail = (response.text or "")[:4000]
        message = f"Tabdeal HTTP {response.status_code} {path}: {detail}"
        if response.status_code == 429 or response.status_code >= 500:
            raise TabdealTransientError(message)
        raise RuntimeError(message)
    return response.json() if response.content else {}


def _tabdeal_wallet_legs_from_row(row: dict[str, Any]) -> dict[str, float]:
    """Tabdeal exposes available ``free`` and frozen ``freeze``; ``total`` is optional."""

    from mm.core.wallet import wallet_legs_from_available_and_locked, wallet_legs_from_total_and_locked

    locked = _safe_float(row.get("freeze", row.get("frozen", row.get("locked", 0.0))))
    if row.get("free") is not None or row.get("available") is not None:
        free = _safe_float(row.get("free", row.get("available", 0.0)))
        total_raw = row.get("total", row.get("totalAsset"))
        if total_raw is not None:
            return wallet_legs_from_total_and_locked(_safe_float(total_raw), locked)
        return wallet_legs_from_available_and_locked(free, locked)
    if row.get("total") is not None or row.get("totalAsset") is not None:
        total = _safe_float(row.get("total", row.get("totalAsset", 0.0)))
        return wallet_legs_from_total_and_locked(total, locked)
    free = _safe_float(row.get("balance", 0.0))
    return wallet_legs_from_available_and_locked(free, locked)


def get_balances(auth: TabdealApiKeyAuth | None) -> dict[str, dict[str, float]]:
    auth = _require_auth(auth)
    if not auth:
        raise RuntimeError("Tabdeal private credentials missing")
    rows: list[Any] = []
    errors: list[str] = []
    try:
        out = _json_or_raise(_request_auth("GET", "/r/api/v1/account", auth), "/r/api/v1/account")
        rows = _balance_rows(out)
        if not rows:
            errors.append("/r/api/v1/account empty balance response")
    except Exception as exc:
        errors.append(f"/r/api/v1/account {type(exc).__name__}: {exc}")
    if not rows:
        try:
            out = _json_or_raise(
                _request_auth("GET", "/r/api/v1/asset/get-funding-asset", auth),
                "/r/api/v1/asset/get-funding-asset",
            )
            rows = _balance_rows(out)
            if not rows:
                errors.append("/r/api/v1/asset/get-funding-asset empty balance response")
        except Exception as exc:
            errors.append(f"/r/api/v1/asset/get-funding-asset {type(exc).__name__}: {exc}")
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset = _asset_code(row)
        if not asset:
            continue
        legs = _tabdeal_wallet_legs_from_row(row)
        prev = result.get(asset)
        if prev is None:
            result[asset] = legs
        else:
            from mm.core.wallet import wallet_legs_from_available_and_locked

            result[asset] = wallet_legs_from_available_and_locked(
                prev["free"] + legs["free"],
                prev["locked"] + legs["locked"],
            )
    from mm.exchanges.instrument_wire import merge_wallet_balances_raw

    merged = merge_wallet_balances_raw("tabdeal", result)
    if not merged:
        detail = "; ".join(errors[-4:]) or "empty balance response"
        raise RuntimeError(f"Tabdeal balance fetch failed: {detail}")
    return merged


def _balance_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("balances", "assets", "rows", "data", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _balance_rows(value)
            if nested:
                return nested
    return [payload] if payload.get("asset") or payload.get("currency") else []


def _asset_code(row: dict[str, Any]) -> str:
    value = row.get("asset") or row.get("currency")
    if isinstance(value, dict):
        value = value.get("asset") or value.get("code") or value.get("symbol")
    return str(value or "").upper().strip()


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
