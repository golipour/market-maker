"""Synchronous KuCoin private REST helpers (Classic Spot API)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests

from mm.core.models.instrument import split_symbol

BASE_URL = "https://api.kucoin.com"


@dataclass(frozen=True, slots=True)
class KucoinApiKeyAuth:
    api_key: str
    api_secret: str
    api_passphrase: str


class KucoinTransientError(RuntimeError):
    pass


def api_symbol(symbol: str) -> str:
    base, quote = split_symbol(symbol)
    return f"{base}-{quote}"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _require_auth(auth: KucoinApiKeyAuth | None) -> KucoinApiKeyAuth | None:
    if (
        isinstance(auth, KucoinApiKeyAuth)
        and auth.api_key
        and auth.api_secret
        and auth.api_passphrase
    ):
        return auth
    return None


def _sign_headers(method: str, endpoint: str, body: str, auth: KucoinApiKeyAuth) -> dict[str, str]:
    timestamp = str(int(time.time() * 1000))
    str_to_sign = timestamp + method.upper() + endpoint + body
    signature = base64.b64encode(
        hmac.new(auth.api_secret.encode("utf-8"), str_to_sign.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")
    passphrase = base64.b64encode(
        hmac.new(
            auth.api_secret.encode("utf-8"),
            auth.api_passphrase.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")
    return {
        "KC-API-KEY": auth.api_key,
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": timestamp,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json",
    }


def _unwrap_kucoin(payload: Any, *, path: str) -> Any:
    if not isinstance(payload, dict):
        raise RuntimeError(f"KuCoin {path}: bad response")
    code = str(payload.get("code", ""))
    if code != "200000":
        message = payload.get("msg") or payload
        raise RuntimeError(f"KuCoin {path}: {message}")
    return payload.get("data")


def _request(
    method: str,
    path: str,
    auth: KucoinApiKeyAuth | None,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    auth = _require_auth(auth)
    if not auth:
        raise RuntimeError("KuCoin private credentials missing")
    endpoint = path
    body = ""
    if method.upper() == "GET" and params:
        endpoint = f"{path}?{urlencode(params)}"
    elif json_body is not None:
        body = json.dumps(json_body, separators=(",", ":"))
    headers = _sign_headers(method, endpoint, body, auth)
    url = f"{BASE_URL}{path}"
    verb = method.upper()
    if verb == "GET":
        response = requests.get(url, params=params or {}, headers=headers, timeout=15)
    elif verb == "POST":
        response = requests.post(url, data=body, headers=headers, timeout=15)
    elif verb == "DELETE":
        response = requests.delete(url, params=params or {}, headers=headers, timeout=15)
    else:
        raise ValueError(f"unsupported HTTP method: {method}")
    if not response.ok:
        try:
            detail: Any = response.json()
        except Exception:
            detail = (response.text or "")[:4000]
        message = f"KuCoin HTTP {response.status_code} {path}: {detail}"
        if response.status_code == 429 or response.status_code >= 500:
            raise KucoinTransientError(message)
        raise RuntimeError(message)
    if not response.content:
        return {}
    return _unwrap_kucoin(response.json(), path=path)


def _public_post(path: str, json_body: dict[str, Any] | None = None) -> Any:
    response = requests.post(
        f"{BASE_URL}{path}",
        json=json_body or {},
        timeout=15,
    )
    if not response.ok:
        response.raise_for_status()
    return _unwrap_kucoin(response.json(), path=path)


def get_balances(auth: KucoinApiKeyAuth | None) -> dict[str, dict[str, float]]:
    rows = _request("GET", "/api/v1/accounts", auth, params={"type": "trade"})
    if not isinstance(rows, list):
        raise RuntimeError("KuCoin balance fetch failed: response is not a list")
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset = str(row.get("currency") or "").upper().strip()
        if not asset:
            continue
        free = _safe_float(row.get("available"))
        locked = _safe_float(row.get("holds"))
        total = _safe_float(row.get("balance"), free + locked)
        prev = result.get(asset)
        if prev is None:
            result[asset] = {"free": free, "locked": locked, "total": total}
        else:
            result[asset] = {
                "free": prev["free"] + free,
                "locked": prev["locked"] + locked,
                "total": prev["total"] + total,
            }
    from mm.exchanges.instrument_wire import merge_wallet_balances_raw

    merged = merge_wallet_balances_raw("kucoin", result)
    if not merged:
        raise RuntimeError("KuCoin balance fetch failed: empty balance response")
    return merged


def _order_rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [row for row in items if isinstance(row, dict)]
    return []


def list_open_orders(symbol: str, auth: KucoinApiKeyAuth | None) -> list[dict]:
    auth = _require_auth(auth)
    if not auth:
        return []
    payload = _request(
        "GET",
        "/api/v1/orders",
        auth,
        params={"status": "active", "symbol": api_symbol(symbol)},
    )
    return _order_rows(payload)


def list_fills(symbol: str, auth: KucoinApiKeyAuth | None) -> list[dict]:
    auth = _require_auth(auth)
    if not auth:
        return []
    payload = _request(
        "GET",
        "/api/v1/fills",
        auth,
        params={"symbol": api_symbol(symbol)},
    )
    return _order_rows(payload)


def place_order(
    symbol: str,
    side: str,
    order_type: str,
    amount: str | float,
    auth: KucoinApiKeyAuth | None,
    *,
    price: str | float | int | None = None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    auth = _require_auth(auth)
    if not auth:
        return {"status": "failed", "message": "missing api key/secret/passphrase"}
    body: dict[str, Any] = {
        "symbol": api_symbol(symbol),
        "side": str(side).lower(),
        "type": str(order_type).lower(),
        "size": str(amount),
    }
    if price is not None:
        body["price"] = str(price)
    if client_order_id:
        body["clientOid"] = str(client_order_id)[:64]
    try:
        out = _request("POST", "/api/v1/orders", auth, json_body=body)
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}


def place_market_order(
    symbol: str,
    side: str,
    amount: str | float,
    auth: KucoinApiKeyAuth | None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    return place_order(symbol, side, "market", amount, auth, client_order_id=client_order_id)


def place_limit_order(
    symbol: str,
    side: str,
    amount: str | float,
    price: str | float | int,
    auth: KucoinApiKeyAuth | None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    return place_order(symbol, side, "limit", amount, auth, price=price, client_order_id=client_order_id)


def cancel_order(symbol: str, order_id: str, auth: KucoinApiKeyAuth | None) -> dict[str, Any]:
    auth = _require_auth(auth)
    if not auth:
        return {"status": "failed", "message": "missing api key/secret/passphrase"}
    try:
        out = _request("DELETE", f"/api/v1/orders/{order_id}", auth)
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}


def bullet_public_websocket_url() -> str | None:
    try:
        data = _public_post("/api/v1/bullet-public", {})
        if not isinstance(data, dict):
            return None
        token = data.get("token")
        servers = data.get("instanceServers") or []
        if not token or not servers:
            return None
        endpoint = str(servers[0].get("endpoint", "")).rstrip("/")
        if not endpoint:
            return None
        from urllib.parse import quote

        return f"{endpoint}/?token={quote(str(token), safe='')}"
    except Exception:
        return None


def bullet_private_websocket_url(auth: KucoinApiKeyAuth | None) -> str | None:
    auth = _require_auth(auth)
    if not auth:
        return None
    try:
        data = _request("POST", "/api/v1/bullet-private", auth, json_body={})
        if not isinstance(data, dict):
            return None
        token = data.get("token")
        servers = data.get("instanceServers") or []
        if not token or not servers:
            return None
        endpoint = str(servers[0].get("endpoint", "")).rstrip("/")
        if not endpoint:
            return None
        from urllib.parse import quote

        return f"{endpoint}/?token={quote(str(token), safe='')}"
    except Exception:
        return None
