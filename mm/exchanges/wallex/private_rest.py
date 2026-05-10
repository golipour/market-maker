"""Synchronous Wallex private REST helpers (no trade_stats)."""

from __future__ import annotations

from typing import Any

import requests

BASE_URL = "https://api.wallex.ir"


def api_symbol(symbol: str) -> str:
    s = symbol.replace("_", "").upper()
    if s.endswith("IRT"):
        return f"{s[:-3]}TMN"
    return s


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _get_auth(path: str, api_token: str, params: dict[str, Any] | None = None) -> Any:
    headers = {
        "X-API-Key": api_token,
        "Accept": "application/json",
    }
    r = requests.get(
        f"{BASE_URL}{path}",
        params=params or {},
        headers=headers,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _post_json_auth(path: str, api_token: str, payload: dict[str, Any]) -> Any:
    headers = {
        "X-API-Key": api_token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    r = requests.post(f"{BASE_URL}{path}", headers=headers, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def _delete_auth(path: str, api_token: str, params: dict[str, Any] | None = None) -> Any:
    headers = {
        "X-API-Key": api_token,
        "Accept": "application/json",
    }
    r = requests.delete(
        f"{BASE_URL}{path}",
        params=params or {},
        headers=headers,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def get_balances(api_token: str | None) -> dict[str, dict[str, float]]:
    """Return all wallet rows Wallex exposes (often includes many zero balances).

    Non-zero filtering for snapshots and strategy frames is applied in
    ``RestPollingPrivateStateAdapter.normalize_balances`` so all exchanges share one path.
    """
    if not api_token:
        return {}
    candidates = [
        "/v1/account/balances",
        "/v1/account/wallets",
        "/v1/account/profile",
        "/v1/account",
    ]
    for path in candidates:
        try:
            out = _get_auth(path, api_token)
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
            free = _safe_float(
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
            locked = _safe_float(row.get("locked", row.get("inOrder", row.get("hold", 0.0))))
            total = _safe_float(row.get("total", free + locked))
            result[asset] = {"free": free, "locked": locked, "total": total}
        if result:
            return result
    return {}


def list_open_orders(symbol: str, api_token: str | None) -> list[dict]:
    if not api_token:
        return []
    sym = api_symbol(symbol)
    try:
        out = _get_auth("/v1/account/openOrders", api_token, params={"symbol": sym})
    except Exception:
        return []
    if not isinstance(out, dict) or not out.get("success"):
        return []
    result = out.get("result") or {}
    orders = result.get("orders") or []
    if not isinstance(orders, list):
        return []
    return [o for o in orders if isinstance(o, dict)]


def place_limit_order(
    symbol: str,
    side: str,
    amount: str | float,
    price: str | float | int,
    api_token: str | None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    if not api_token:
        return {"status": "failed", "message": "missing api token"}
    payload: dict[str, Any] = {
        "symbol": api_symbol(symbol),
        "type": "LIMIT",
        "side": str(side).upper(),
        "price": str(price),
        "quantity": str(amount),
    }
    if client_order_id:
        payload["client_id"] = str(client_order_id)[:64]
    try:
        out = _post_json_auth("/v1/account/orders", api_token, payload)
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}


def cancel_order(client_order_id: str, api_token: str | None) -> dict[str, Any]:
    if not api_token:
        return {"status": "failed", "message": "missing api token"}
    try:
        out = _delete_auth("/v1/account/orders", api_token, params={"clientOrderId": str(client_order_id)})
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}
