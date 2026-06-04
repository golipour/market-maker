"""Synchronous Bitpin private REST helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

BASE_URL = "https://api.bitpin.ir"


@dataclass(frozen=True, slots=True)
class BitpinApiKeyAuth:
    api_key: str
    api_secret: str


@dataclass(frozen=True, slots=True)
class BitpinTokenAuth:
    access_token: str
    refresh_token: str | None = None


@dataclass(frozen=True, slots=True)
class BitpinWebSocketInfo:
    user_identifier: str
    ws_token: str


BitpinAuth = BitpinApiKeyAuth | BitpinTokenAuth | str

_MARKET_ID_CACHE: dict[str, int] = {}
_ACCESS_TOKEN_CACHE: dict[tuple[str, str], str] = {}


def api_symbol(symbol: str) -> str:
    from mm.core.models.instrument import split_symbol

    base, quote = split_symbol(symbol)
    return f"{base}_{quote}".upper()


def resolve_market_id(symbol: str) -> int:
    market_symbol = api_symbol(symbol)
    cached = _MARKET_ID_CACHE.get(market_symbol)
    if cached is not None:
        return cached
    page = 1
    while True:
        payload = _get_public("mkt/markets/", {"page": page})
        for row in _result_rows(payload):
            if not isinstance(row, dict):
                continue
            if _same_market_symbol(_market_code(row), market_symbol):
                market_id = int(row["id"])
                _MARKET_ID_CACHE[market_symbol] = market_id
                return market_id
        if not isinstance(payload, dict) or not payload.get("next"):
            break
        page += 1
    raise RuntimeError(f"Bitpin market symbol not found: {symbol}")


def login(auth: BitpinApiKeyAuth) -> BitpinTokenAuth:
    key = (auth.api_key, auth.api_secret)
    cached = _ACCESS_TOKEN_CACHE.get(key)
    if cached:
        return BitpinTokenAuth(cached)
    payload = _post_public("usr/api/login/", {"api_key": auth.api_key, "api_secret": auth.api_secret})
    access = str(payload.get("access") or payload.get("access_token") or "").strip()
    refresh = str(payload.get("refresh") or payload.get("refresh_token") or "").strip() or None
    if not access:
        raise RuntimeError("Bitpin login failed: missing access token")
    _ACCESS_TOKEN_CACHE[key] = access
    return BitpinTokenAuth(access, refresh)


def get_websocket_info(auth: BitpinAuth | None) -> BitpinWebSocketInfo | None:
    if isinstance(auth, BitpinApiKeyAuth):
        payload = _post_public("usr/ws-info/", {"api_key": auth.api_key, "secret_key": auth.api_secret})
        user_identifier = str(payload.get("user_identifier") or "").strip()
        ws_token = str(payload.get("ws_token") or "").strip()
        if user_identifier and ws_token:
            return BitpinWebSocketInfo(user_identifier=user_identifier, ws_token=ws_token)
    return None


def get_balances(auth: BitpinAuth | None) -> dict[str, dict[str, float]]:
    token = _access_token(auth)
    if not token:
        return {}
    payload = _get_auth("wlt/wallets/", token)
    result: dict[str, dict[str, float]] = {}
    for row in _result_rows(payload):
        if not isinstance(row, dict):
            continue
        asset = _asset_code(row)
        if not asset:
            continue
        free = _safe_float(row.get("balance") or row.get("available") or row.get("free"))
        locked = _safe_float(row.get("blocked") or row.get("locked") or row.get("freeze"))
        total = _safe_float(row.get("total"), free + locked)
        result[asset] = {"free": free, "locked": locked, "total": total}
    from mm.exchanges.instrument_wire import merge_wallet_balances_raw

    return merge_wallet_balances_raw("bitpin", result)


def list_open_orders(symbol: str, auth: BitpinAuth | None) -> list[dict]:
    token = _access_token(auth)
    if not token:
        return []
    market_id = resolve_market_id(symbol)
    payload = _get_auth("odr/orders/", token, {"market_id": market_id, "state": "active"})
    return [row for row in _result_rows(payload) if isinstance(row, dict)]


def list_filled_orders(symbol: str, auth: BitpinAuth | None) -> list[dict]:
    token = _access_token(auth)
    if not token:
        return []
    market_id = resolve_market_id(symbol)
    payload = _get_auth("odr/matches/", token, {"market_id": market_id})
    rows = [row for row in _result_rows(payload) if isinstance(row, dict)]
    if rows:
        return rows
    payload = _get_auth("odr/orders/", token, {"market_id": market_id, "state": "closed"})
    return [
        row
        for row in _result_rows(payload)
        if isinstance(row, dict) and (_safe_float(row.get("exchanged1") or row.get("fulfilled")) > 0.0)
    ]


def place_market_order(
    symbol: str,
    side: str,
    amount: str | float,
    auth: BitpinAuth | None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    return place_order(symbol, side, amount, auth, mode="market", price=0, client_order_id=client_order_id)


def place_limit_order(
    symbol: str,
    side: str,
    amount: str | float,
    price: str | float | int,
    auth: BitpinAuth | None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    return place_order(symbol, side, amount, auth, mode="limit", price=price, client_order_id=client_order_id)


def place_order(
    symbol: str,
    side: str,
    amount: str | float,
    auth: BitpinAuth | None,
    *,
    mode: str,
    price: str | float | int,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    token = _access_token(auth)
    if not token:
        return {"status": "failed", "message": "missing bitpin credentials"}
    payload: dict[str, Any] = {
        "market": str(resolve_market_id(symbol)),
        "amount1": str(amount),
        "price": str(price),
        "mode": str(mode).lower(),
        "type": str(side).lower(),
    }
    if client_order_id:
        payload["identifier"] = str(client_order_id)[:64]
    try:
        return _post_auth("odr/orders/", token, payload)
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}


def place_bulk_orders(orders: list[dict[str, Any]], auth: BitpinAuth | None) -> dict[str, Any]:
    token = _access_token(auth)
    if not token:
        return {"status": "failed", "message": "missing bitpin credentials"}
    if not orders:
        return {"orders": []}
    if len(orders) > 10:
        return {"status": "failed", "message": "bitpin bulk order limit is 10"}
    symbols = {api_symbol(str(order.get("symbol", ""))) for order in orders if order.get("symbol")}
    if len(symbols) != 1:
        return {"status": "failed", "message": "bitpin bulk orders must share one market symbol"}
    payload = [_bulk_order_payload(order) for order in orders]
    try:
        return _post_auth("odr/orders/bulk", token, payload)
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}


def cancel_order(order_id: str, auth: BitpinAuth | None) -> dict[str, Any]:
    token = _access_token(auth)
    if not token:
        return {"status": "failed", "message": "missing bitpin credentials"}
    try:
        return _delete_auth(f"odr/orders/{order_id}/", token)
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}


def cancel_bulk_orders(
    order_ids: list[str] | None,
    auth: BitpinAuth | None,
    *,
    identifiers: list[str] | None = None,
) -> dict[str, Any]:
    token = _access_token(auth)
    if not token:
        return {"status": "failed", "message": "missing bitpin credentials"}
    ids = [str(order_id) for order_id in (order_ids or []) if str(order_id).strip()]
    idents = [str(identifier) for identifier in (identifiers or []) if str(identifier).strip()]
    if not ids and not idents:
        return {"canceled_orders": [], "not_canceled_orders": []}
    payload: dict[str, Any]
    if idents and not ids:
        payload = {"identifiers": idents}
    else:
        payload = {"ids": ids}
    try:
        return _delete_auth("odr/orders/bulk/", token, payload=payload)
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}


def _access_token(auth: BitpinAuth | None) -> str:
    if not auth:
        return ""
    if isinstance(auth, BitpinApiKeyAuth):
        return login(auth).access_token
    if isinstance(auth, BitpinTokenAuth):
        return auth.access_token
    raw = str(auth).strip()
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return raw


def _get_public(path: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(f"{BASE_URL}/v1/{path}", params=params or {}, timeout=15)
    response.raise_for_status()
    return response.json()


def _post_public(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{BASE_URL}/v1/{path}", json=payload, timeout=15)
    if not response.ok:
        raise RuntimeError(f"Bitpin HTTP {response.status_code} /v1/{path}: {_response_detail(response)}")
    out = response.json()
    return out if isinstance(out, dict) else {}


def _get_auth(path: str, access_token: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        f"{BASE_URL}/v1/{path}",
        params=params or {},
        headers=_auth_headers(access_token),
        timeout=15,
    )
    if not response.ok:
        raise RuntimeError(f"Bitpin HTTP {response.status_code} /v1/{path}: {_response_detail(response)}")
    return response.json()


def _post_auth(path: str, access_token: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{BASE_URL}/v1/{path}",
        json=payload,
        headers=_auth_headers(access_token),
        timeout=15,
    )
    if not response.ok:
        raise RuntimeError(f"Bitpin HTTP {response.status_code} /v1/{path}: {_response_detail(response)}")
    out = response.json()
    return out if isinstance(out, dict) else {}


def _delete_auth(path: str, access_token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.delete(
        f"{BASE_URL}/v1/{path}",
        json=payload,
        headers=_auth_headers(access_token),
        timeout=15,
    )
    if not response.ok:
        raise RuntimeError(f"Bitpin HTTP {response.status_code} /v1/{path}: {_response_detail(response)}")
    out = response.json() if response.content else {}
    return out if isinstance(out, dict) else {}


def _auth_headers(access_token: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }


def _response_detail(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return (response.text or "")[:4000]


def _bulk_order_payload(order: dict[str, Any]) -> dict[str, Any]:
    symbol = api_symbol(str(order.get("symbol") or ""))
    payload: dict[str, Any] = {
        "symbol": symbol,
        "base_amount": str(order.get("base_amount") or order.get("amount") or order.get("amount1") or order.get("qty")),
        "price": str(order.get("price", 0)),
        "side": str(order.get("side") or order.get("type") or "").lower(),
        "type": str(order.get("order_type") or order.get("mode") or "limit").lower(),
    }
    identifier = order.get("identifier") or order.get("client_order_id")
    if identifier:
        payload["identifier"] = str(identifier)[:64]
    return payload


def _market_code(row: dict[str, Any]) -> str:
    code = str(row.get("code") or row.get("symbol") or "").upper().strip()
    if code:
        return code.replace("-", "_")
    base = row.get("currency1") if isinstance(row.get("currency1"), dict) else {}
    quote = row.get("currency2") if isinstance(row.get("currency2"), dict) else {}
    b = str(base.get("code") or "").upper().strip()
    q = str(quote.get("code") or "").upper().strip()
    return f"{b}_{q}" if b and q else ""


def _same_market_symbol(left: str, right: str) -> bool:
    return left.replace("_", "").replace("-", "").upper() == right.replace("_", "").replace("-", "").upper()


def _result_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    rows = payload.get("results")
    if isinstance(rows, list):
        return rows
    if isinstance(rows, dict):
        return [rows]
    for key in ("orders", "matches", "wallets", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return [payload] if payload else []


def _asset_code(row: dict[str, Any]) -> str:
    for key in ("currency", "asset", "coin"):
        value = row.get(key)
        if isinstance(value, dict):
            code = str(value.get("code") or value.get("symbol") or "").upper().strip()
            if code:
                return code
        code = str(value or "").upper().strip()
        if code:
            return code
    return ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float(default)
