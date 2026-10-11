"""Bitwana private REST helpers (JWT access + refresh)."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import requests

from mm.core.wallet import wallet_legs_from_available_and_locked
from mm.exchanges.instrument_wire import build_instrument_wire, merge_wallet_balances_raw
from mm.observability.logs import log_event

BASE_URL = os.environ.get("BITWANA_API_BASE_URL", "https://bit.yshm.ir").rstrip("/")

_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


def get_session(pool_size: int = 32) -> requests.Session:
    """Thread-safe pooled requests session with Keep-Alive for Bitwana REST."""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                s = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=pool_size,
                    pool_maxsize=pool_size,
                    max_retries=1,
                )
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _SESSION = s
    return _SESSION


def reset_session() -> None:
    """Close and reset pooled session (useful for tests or token reboots)."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            try:
                _SESSION.close()
            except Exception:
                pass
            _SESSION = None

# Bitwana wraps REST results as ``{statusCode, content}``. Cancel (and some
# async writes) return 202 Accepted; treating that as an error aborts replace
# syncs so new quotes never reach the book.
_SUCCESS_STATUS_CODES = {200, 201, 202, 204}


def api_symbol(symbol: str) -> str:
    return build_instrument_wire("bitwana", symbol).api_market_symbol


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float(default)


def _status_code(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unwrap(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if "content" in payload:
        code = _status_code(payload.get("statusCode"))
        if code is not None and code not in _SUCCESS_STATUS_CODES:
            message = payload.get("message") or payload.get("descCode") or "bitwana error"
            raise RuntimeError(f"Bitwana HTTP {code}: {message}")
        return payload.get("content")
    return payload


def _is_already_gone_cancel_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "http 404" in text
        or "not found" in text
        or "already cancel" in text
        or "already_cancel" in text
    )


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
    on_refresh: Any = field(default=None, repr=False, compare=False)

    def authorization_header(self) -> str:
        token = str(self.access_token or "").strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        return token

    def refresh(self) -> None:
        refresh = str(self.refresh_token or "").strip()
        if not refresh:
            raise RuntimeError("Bitwana refresh token is missing")
        with self._lock:
            import requests as _requests_module

            session = get_session()
            url = f"{self.base_url.rstrip('/')}/api/1/users/authenticate/refresh-token"
            payload = {"refreshToken": refresh}
            hdrs = {"Accept": "application/json", "Content-Type": "application/json"}
            if getattr(requests, "post", None) is not _requests_module.api.post:
                response = requests.post(url, json=payload, headers=hdrs, timeout=15)
            else:
                response = session.post(url, json=payload, headers=hdrs, timeout=15)
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
            rotated = bool(next_refresh) and next_refresh != refresh
            if next_refresh:
                self.refresh_token = next_refresh
            # The rotated pair only lives in this process. Nothing writes it back to
            # the registry, so a restart falls back to the stored (possibly expired)
            # token. Announce every rotation so the window is visible in logs instead
            # of surfacing later as an unexplained auth failure.
            if callable(self.on_refresh):
                try:
                    self.on_refresh(access_token=access, refresh_token=self.refresh_token, rotated=rotated)
                except Exception:  # pragma: no cover - never break auth on a notifier fault
                    pass


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
    import requests as _requests_module

    session = get_session()
    url = f"{auth.base_url.rstrip('/')}{path}"
    if getattr(requests, "request", None) is not _requests_module.api.request:
        response = requests.request(
            method,
            url,
            params=params or {},
            json=json_body,
            headers=headers,
            timeout=15,
        )
    else:
        response = session.request(
            method,
            url,
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
        if _is_already_gone_cancel_error(exc):
            return {"success": True, "id": oid, "already_gone": True, "message": str(exc)}
        return {"status": "failed", "message": str(exc)}
    if isinstance(content, dict):
        return {"success": True, **content}
    return {"success": True, "id": oid}
