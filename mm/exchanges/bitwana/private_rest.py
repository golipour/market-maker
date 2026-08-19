"""Bitwana private REST helpers (JWT access + refresh)."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import requests

from mm.core.wallet import wallet_legs_from_available_and_locked
from mm.exchanges.instrument_wire import build_instrument_wire, merge_wallet_balances_raw

BASE_URL = os.environ.get("BITWANA_API_BASE_URL", "https://bit.yshm.ir").rstrip("/")


def api_symbol(symbol: str) -> str:
    return build_instrument_wire("bitwana", symbol).api_market_symbol


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float(default)


def _unwrap(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if "content" in payload:
        code = payload.get("statusCode")
        if code not in (None, 200, 201):
            message = payload.get("message") or payload.get("descCode") or "bitwana error"
            raise RuntimeError(f"Bitwana HTTP {code}: {message}")
        return payload.get("content")
    return payload


def _as_rows(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if not payload:
            return []
        inner = payload.get("content")
        if isinstance(inner, list):
            return inner
        items = payload.get("items")
        if isinstance(items, list):
            return items
        raise RuntimeError("Bitwana list fetch failed: expected an array")
    raise RuntimeError("Bitwana list fetch failed: expected an array")


def _asset_symbol(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("symbol") or "").upper().strip()
    return str(value or "").upper().strip()


@dataclass
class BitwanaJwtAuth:
    """Access JWT plus optional refresh token used on 401."""

    access_token: str
    refresh_token: str = ""
    base_url: str = BASE_URL
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def authorization_header(self) -> str:
        token = str(self.access_token or "").strip()
        if not token:
            return ""
        if token.lower().startswith("bearer "):
            return token
        return f"Bearer {token}"

    def refresh(self) -> None:
        refresh = str(self.refresh_token or "").strip()
        if not refresh:
            raise RuntimeError("Bitwana refresh token is missing")
        with self._lock:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/api/1/users/authenticate/refresh-token",
                json={"refreshToken": refresh},
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=15,
            )
            if not response.ok:
                raise RuntimeError(f"Bitwana token refresh failed: HTTP {response.status_code}")
            content = _unwrap(response.json())
            if not isinstance(content, dict):
                raise RuntimeError("Bitwana token refresh failed: empty response")
            access = str(content.get("accessToken") or "").strip()
            if not access:
                raise RuntimeError("Bitwana token refresh failed: missing accessToken")
            self.access_token = access
            next_refresh = str(content.get("refreshToken") or "").strip()
            if next_refresh:
                self.refresh_token = next_refresh


def coerce_auth(api_token: Any) -> BitwanaJwtAuth | None:
    if api_token is None:
        return None
    if isinstance(api_token, BitwanaJwtAuth):
        return api_token
    if isinstance(api_token, str) and api_token.strip():
        return BitwanaJwtAuth(access_token=api_token.strip())
    return None


def _headers(auth: BitwanaJwtAuth) -> dict[str, str]:
    header = auth.authorization_header()
    out = {"Accept": "application/json"}
    if header:
        out["Authorization"] = header
    return out


def _request(
    method: str,
    path: str,
    auth: BitwanaJwtAuth | None,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    retry: bool = True,
) -> Any:
    if auth is None:
        raise RuntimeError("missing bitwana auth")
    headers = _headers(auth)
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    response = requests.request(
        method,
        f"{auth.base_url.rstrip('/')}{path}",
        params=params or {},
        json=json_body,
        headers=headers,
        timeout=15,
    )
    if response.status_code == 401 and retry and auth.refresh_token:
        auth.refresh()
        return _request(
            method,
            path,
            auth,
            params=params,
            json_body=json_body,
            retry=False,
        )
    if not response.ok:
        detail: Any
        try:
            detail = response.json()
            if isinstance(detail, dict):
                detail = detail.get("message") or detail.get("descCode") or detail
        except Exception:
            detail = (response.text or "")[:400]
        raise RuntimeError(f"Bitwana HTTP {response.status_code} {path}: {detail}")
    if not response.content:
        return None
    return _unwrap(response.json())


def get_balances(api_token: Any) -> dict[str, dict[str, float]]:
    auth = coerce_auth(api_token)
    if auth is None:
        return {}
    rows = _as_rows(_request("GET", "/api/1/wallets/self", auth))
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset = _asset_symbol(row.get("currencyName") or row.get("assetId"))
        if not asset:
            continue
        total = _safe_float(row.get("totalBalance"))
        free = _safe_float(row.get("availableBalance"))
        locked = _safe_float(row.get("blockedBalance"))
        result[asset] = wallet_legs_from_available_and_locked(free, locked, total=total)
    if not result:
        raise RuntimeError("Bitwana balance fetch failed: empty wallet response")
    return merge_wallet_balances_raw("bitwana", result)


def list_open_orders(symbol: str, api_token: Any) -> list[dict]:
    auth = coerce_auth(api_token)
    if auth is None:
        return []
    market = api_symbol(symbol)
    try:
        payload = _request("GET", "/api/1/orders/open/self", auth)
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
        payload = _request("GET", f"/api/1/orders/self/marketname/{market}/100/0", auth)
    rows = _as_rows(payload)
    open_status = {"PENDING", "OPEN", "PARTIALLY_DONE"}
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").upper()
        if status and status not in open_status:
            continue
        row_market = _asset_symbol(row.get("marketName")) or str(row.get("marketName") or "")
        if row_market and row_market.replace("-", "_") != market:
            continue
        size = _safe_float(row.get("size"))
        remaining = _safe_float(row.get("remainedSize"), size)
        out.append(
            {
                **row,
                "price": row.get("unitPrice"),
                "amount": size,
                "remainingAmount": remaining,
                "filledAmount": max(0.0, size - remaining),
                "side": row.get("side"),
            }
        )
    return out


def list_filled_orders(symbol: str, api_token: Any) -> list[dict]:
    auth = coerce_auth(api_token)
    if auth is None:
        return []
    market = api_symbol(symbol)
    payload = _request(
        "GET",
        "/api/1/trades/self/100/0",
        auth,
        params={"market": market},
    )
    rows = _as_rows(payload)
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        qty = _safe_float(row.get("amount"))
        if qty <= 0.0:
            continue
        out.append(
            {
                **row,
                "price": row.get("unitPrice"),
                "executedQty": qty,
                "filledAmount": qty,
                "quantity": qty,
                "side": row.get("side"),
            }
        )
    return out


def place_limit_order(
    symbol: str,
    side: str,
    amount: str | float,
    price: str | float | int,
    api_token: Any,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    auth = coerce_auth(api_token)
    if auth is None:
        return {"status": "failed", "message": "missing api token"}
    payload: dict[str, Any] = {
        "marketName": api_symbol(symbol),
        "type": "LIMIT",
        "side": str(side).upper(),
        "size": _safe_float(amount),
        "unitPrice": _safe_float(price),
    }
    if client_order_id:
        payload["externalOrder"] = True
    try:
        content = _request("POST", "/api/1/orders", auth, json_body=payload)
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}
    if isinstance(content, dict):
        return {"success": True, **content}
    return {"status": "failed", "message": "bad response"}


def place_market_order(
    symbol: str,
    side: str,
    amount: str | float,
    api_token: Any,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    auth = coerce_auth(api_token)
    if auth is None:
        return {"status": "failed", "message": "missing api token"}
    payload: dict[str, Any] = {
        "marketName": api_symbol(symbol),
        "type": "MARKET",
        "side": str(side).upper(),
        "size": _safe_float(amount),
    }
    if client_order_id:
        payload["externalOrder"] = True
    try:
        content = _request("POST", "/api/1/orders", auth, json_body=payload)
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}
    if isinstance(content, dict):
        return {"success": True, **content}
    return {"status": "failed", "message": "bad response"}


def cancel_order(order_id: str | int, api_token: Any) -> dict[str, Any]:
    auth = coerce_auth(api_token)
    if auth is None:
        return {"status": "failed", "message": "missing api token"}
    try:
        oid = int(str(order_id).strip())
    except (TypeError, ValueError):
        return {"status": "failed", "message": "invalid order id"}
    try:
        content = _request("PUT", f"/api/1/orders/cancel/{oid}", auth)
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}
    if isinstance(content, dict):
        return {"success": True, **content}
    return {"success": True, "id": oid}
