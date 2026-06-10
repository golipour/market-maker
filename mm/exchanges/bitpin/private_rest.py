"""Synchronous Bitpin private REST helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading
import time
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
_ACCESS_TOKEN_CACHE: dict[tuple[str, str], BitpinTokenAuth] = {}
_AUTH_THROTTLE_UNTIL: dict[tuple[str, str], float] = {}
_AUTH_LOCK = threading.Lock()


class BitpinAuthError(RuntimeError):
    def __init__(self, status_code: int, path: str, detail: Any):
        self.status_code = int(status_code)
        self.path = path
        self.detail = detail
        super().__init__(f"Bitpin HTTP {status_code} {path}: {detail}")


class BitpinAuthCooldown(RuntimeError):
    pass


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
    with _AUTH_LOCK:
        cached = _ACCESS_TOKEN_CACHE.get(key)
        if cached:
            return cached
        _raise_if_auth_throttled(key)
        try:
            payload = _post_public(
                "usr/authenticate/",
                {"api_key": auth.api_key, "secret_key": auth.api_secret},
            )
        except BitpinAuthError as exc:
            _remember_auth_throttle(key, exc)
            raise
        access = str(payload.get("access") or payload.get("access_token") or "").strip()
        refresh = str(payload.get("refresh") or payload.get("refresh_token") or "").strip() or None
        if not access:
            raise RuntimeError("Bitpin login failed: missing access token")
        token = BitpinTokenAuth(access, refresh)
        _ACCESS_TOKEN_CACHE[key] = token
        _AUTH_THROTTLE_UNTIL.pop(key, None)
        return token


def refresh_access_token(auth: BitpinApiKeyAuth) -> BitpinTokenAuth:
    key = (auth.api_key, auth.api_secret)
    with _AUTH_LOCK:
        cached = _ACCESS_TOKEN_CACHE.get(key)
        if cached and cached.refresh_token:
            try:
                payload = _post_public("usr/refresh_token/", {"refresh": cached.refresh_token})
            except BitpinAuthError as exc:
                if exc.status_code != 401:
                    _remember_auth_throttle(key, exc)
                    raise
            else:
                access = str(payload.get("access") or payload.get("access_token") or "").strip()
                if access:
                    token = BitpinTokenAuth(access, cached.refresh_token)
                    _ACCESS_TOKEN_CACHE[key] = token
                    return token
        _ACCESS_TOKEN_CACHE.pop(key, None)
    return login(auth)


def get_websocket_info(auth: BitpinAuth | None) -> BitpinWebSocketInfo | None:
    if isinstance(auth, BitpinApiKeyAuth):
        payload = _post_public("usr/ws-info/", {"api_key": auth.api_key, "secret_key": auth.api_secret})
        user_identifier = str(payload.get("user_identifier") or "").strip()
        ws_token = str(payload.get("ws_token") or "").strip()
        if user_identifier and ws_token:
            return BitpinWebSocketInfo(user_identifier=user_identifier, ws_token=ws_token)
    return None


def get_balances(auth: BitpinAuth | None) -> dict[str, dict[str, float]]:
    payload = _auth_get(auth, "wlt/wallets/")
    if payload is None:
        return {}
    result: dict[str, dict[str, float]] = {}
    for row in _result_rows(payload):
        if not isinstance(row, dict):
            continue
        asset = _asset_code(row)
        if not asset:
            continue
        locked = _safe_float(row.get("frozen") or row.get("blocked") or row.get("locked") or row.get("freeze"))
        total_raw = row.get("total")
        if total_raw is not None:
            from mm.core.wallet import wallet_legs_from_total_and_locked

            legs = wallet_legs_from_total_and_locked(_safe_float(total_raw), locked)
        else:
            from mm.core.wallet import wallet_legs_from_available_and_locked

            free = _safe_float(row.get("balance") or row.get("available") or row.get("free"))
            legs = wallet_legs_from_available_and_locked(free, locked)
        result[asset] = legs
    from mm.exchanges.instrument_wire import merge_wallet_balances_raw

    return merge_wallet_balances_raw("bitpin", result)


def list_open_orders(symbol: str, auth: BitpinAuth | None) -> list[dict]:
    market_id = resolve_market_id(symbol)
    payload = _auth_get(auth, "odr/orders/", {"market_id": market_id, "state": "active"})
    if payload is None:
        return []
    return [row for row in _result_rows(payload) if isinstance(row, dict)]


def list_filled_orders(symbol: str, auth: BitpinAuth | None) -> list[dict]:
    payload = _auth_get(
        auth,
        "odr/fills/",
        {"symbol": api_symbol(symbol), "limit": 100},
    )
    if payload is None:
        return []
    rows = [row for row in _result_rows(payload) if isinstance(row, dict)]
    if rows:
        return rows
    market_id = resolve_market_id(symbol)
    payload = _auth_get(auth, "odr/orders/", {"market_id": market_id, "state": "closed"})
    if payload is None:
        return []
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
        out = _auth_post(auth, "odr/orders/", payload)
        return out if isinstance(out, dict) else {"status": "failed", "message": "missing bitpin credentials"}
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}


def place_bulk_orders(orders: list[dict[str, Any]], auth: BitpinAuth | None) -> dict[str, Any]:
    if not orders:
        return {"orders": []}
    if len(orders) > 10:
        return {"status": "failed", "message": "bitpin bulk order limit is 10"}
    symbols = {api_symbol(str(order.get("symbol", ""))) for order in orders if order.get("symbol")}
    if len(symbols) != 1:
        return {"status": "failed", "message": "bitpin bulk orders must share one market symbol"}
    payload = [_bulk_order_payload(order) for order in orders]
    try:
        out = _auth_post(auth, "odr/orders/bulk/", payload)
        return out if isinstance(out, dict) else {"status": "failed", "message": "missing bitpin credentials"}
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}


def cancel_order(order_id: str, auth: BitpinAuth | None) -> dict[str, Any]:
    try:
        out = _auth_delete(auth, f"odr/orders/{order_id}/")
        return out if isinstance(out, dict) else {"status": "failed", "message": "missing bitpin credentials"}
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}


def cancel_bulk_orders(
    order_ids: list[str] | None,
    auth: BitpinAuth | None,
    *,
    identifiers: list[str] | None = None,
) -> dict[str, Any]:
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
        out = _auth_delete(auth, "odr/orders/bulk/", payload=payload)
        return out if isinstance(out, dict) else {"status": "failed", "message": "missing bitpin credentials"}
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


def _invalidate_access_token(auth: BitpinAuth | None) -> None:
    if isinstance(auth, BitpinApiKeyAuth):
        with _AUTH_LOCK:
            _ACCESS_TOKEN_CACHE.pop((auth.api_key, auth.api_secret), None)


def _raise_if_auth_throttled(key: tuple[str, str]) -> None:
    wait_s = _AUTH_THROTTLE_UNTIL.get(key, 0.0) - time.monotonic()
    if wait_s > 0.0:
        raise BitpinAuthCooldown(
            f"Bitpin authenticate throttled; retry after {int(wait_s) + 1} seconds"
        )


def _remember_auth_throttle(key: tuple[str, str], exc: BitpinAuthError) -> None:
    wait_s = _throttle_wait_seconds(exc)
    if wait_s is not None:
        _AUTH_THROTTLE_UNTIL[key] = time.monotonic() + wait_s


def _throttle_wait_seconds(exc: BitpinAuthError) -> float | None:
    detail = str(exc.detail).lower()
    if exc.status_code == 406 and "wrong ip" in detail:
        return 300.0
    if exc.status_code != 429:
        return None
    match = re.search(r"Expected available in\s+(\d+)\s+seconds", str(exc.detail))
    if match:
        return max(1.0, float(match.group(1)))
    return 30.0


def _auth_get(auth: BitpinAuth | None, path: str, params: dict[str, Any] | None = None) -> Any:
    return _auth_request(auth, lambda token: _get_auth(path, token, params))


def _auth_post(auth: BitpinAuth | None, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    return _auth_request(auth, lambda token: _post_auth(path, token, payload))


def _auth_delete(
    auth: BitpinAuth | None,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return _auth_request(auth, lambda token: _delete_auth(path, token, payload=payload))


def _auth_request(auth: BitpinAuth | None, request: Any) -> Any:
    token = _access_token(auth)
    if not token:
        return None
    try:
        return request(token)
    except BitpinAuthError as exc:
        if not isinstance(auth, BitpinApiKeyAuth) or not _is_token_not_valid(exc):
            raise
        token = refresh_access_token(auth).access_token
        try:
            return request(token)
        except BitpinAuthError as retry_exc:
            if _is_token_not_valid(retry_exc):
                _invalidate_access_token(auth)
            raise


def _is_token_not_valid(exc: BitpinAuthError) -> bool:
    if exc.status_code != 401:
        return False
    return "token_not_valid" in str(exc.detail) or "token is invalid or expired" in str(exc.detail).lower()


def _get_public(path: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(f"{BASE_URL}/v1/{path}", params=params or {}, timeout=15)
    response.raise_for_status()
    return response.json()


def _post_public(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{BASE_URL}/api/v1/{path}", json=payload, timeout=15)
    if not response.ok:
        raise BitpinAuthError(response.status_code, f"/api/v1/{path}", _response_detail(response))
    out = response.json()
    return out if isinstance(out, dict) else {}


def _get_auth(path: str, access_token: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        f"{BASE_URL}/api/v1/{path}",
        params=params or {},
        headers=_auth_headers(access_token),
        timeout=15,
    )
    if not response.ok:
        raise BitpinAuthError(response.status_code, f"/api/v1/{path}", _response_detail(response))
    return response.json()


def _post_auth(path: str, access_token: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{BASE_URL}/api/v1/{path}",
        json=payload,
        headers=_auth_headers(access_token),
        timeout=15,
    )
    if not response.ok:
        raise BitpinAuthError(response.status_code, f"/api/v1/{path}", _response_detail(response))
    out = response.json()
    return out if isinstance(out, dict) else {}


def _delete_auth(path: str, access_token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.delete(
        f"{BASE_URL}/api/v1/{path}",
        json=payload,
        headers=_auth_headers(access_token),
        timeout=15,
    )
    if not response.ok:
        raise BitpinAuthError(response.status_code, f"/api/v1/{path}", _response_detail(response))
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
