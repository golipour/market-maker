"""Synchronous Wallex private REST helpers (no trade_stats)."""

from __future__ import annotations

from typing import Any

import requests

BASE_URL = "https://api.wallex.ir"


def api_symbol(symbol: str) -> str:
    """Wallex REST/WS market symbol (``USDT_IRT`` in config → ``USDTTMN`` on the wire)."""

    from mm.exchanges.instrument_wire import build_instrument_wire

    return build_instrument_wire("wallex", symbol).api_market_symbol


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
    if not r.ok:
        detail: Any
        try:
            detail = r.json()
        except Exception:
            detail = (r.text or "")[:4000]
        raise RuntimeError(f"Wallex HTTP {r.status_code} {path}: {detail}")
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
    errors: list[str] = []
    for path in candidates:
        try:
            out = _get_auth(path, api_token)
        except Exception as exc:
            errors.append(f"{path} {type(exc).__name__}: {exc}")
            continue
        result: dict[str, dict[str, float]] = {}
        if isinstance(out, dict) and out.get("success") is False:
            errors.append(f"{path} success=false")
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
            from mm.exchanges.instrument_wire import merge_wallet_balances_raw

            return merge_wallet_balances_raw("wallex", result)
        errors.append(f"{path} empty balance response")
    detail = "; ".join(errors[-4:]) or "empty balance response"
    raise RuntimeError(f"Wallex balance fetch failed: {detail}")


def list_open_orders(symbol: str, api_token: str | None) -> list[dict]:
    if not api_token:
        return []
    sym = api_symbol(symbol)
    try:
        out = _get_auth("/v1/account/openOrders", api_token, params={"symbol": sym})
    except Exception as exc:
        raise RuntimeError(f"Wallex open-order fetch failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(out, dict) or not out.get("success"):
        raise RuntimeError("Wallex open-order fetch failed: success=false")
    result = out.get("result") or {}
    orders = result.get("orders") or []
    if not isinstance(orders, list):
        raise RuntimeError("Wallex open-order fetch failed: orders is not a list")
    return [o for o in orders if isinstance(o, dict)]


def list_filled_orders(symbol: str, api_token: str | None) -> list[dict]:
    if not api_token:
        return []
    sym = api_symbol(symbol)
    try:
        out = _get_auth(
            "/v1/account/orders",
            api_token,
            params={"market": sym, "per_page": 100},
        )
    except Exception as exc:
        raise RuntimeError(f"Wallex order-history fetch failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(out, dict) or not out.get("success"):
        raise RuntimeError("Wallex order-history fetch failed: success=false")
    result = out.get("result") or {}
    rows: list[dict] = []
    if isinstance(result, dict):
        for value in result.values():
            if isinstance(value, list):
                rows.extend(row for row in value if isinstance(row, dict))
    elif isinstance(result, list):
        rows.extend(row for row in result if isinstance(row, dict))
    filled_rows = [
        row
        for row in rows
        if str(row.get("status", "")).upper() in {"FILLED", "PARTIALLY_FILLED", "PARTIAL_FILLED"}
        or _safe_float(row.get("executedQty", row.get("filledAmount", 0.0))) > 0.0
    ]
    enriched: list[dict] = []
    for row in filled_rows:
        fills = row.get("fills")
        if fills:
            enriched.append(row)
            continue
        client_order_id = str(row.get("clientOrderId") or row.get("client_order_id") or "").strip()
        if not client_order_id:
            enriched.append(row)
            continue
        try:
            detail = get_order(client_order_id, api_token)
        except Exception:
            enriched.append(row)
            continue
        enriched.append(detail if detail else row)
    return enriched


def get_order(client_order_id: str, api_token: str | None) -> dict:
    if not api_token:
        return {}
    order_id = str(client_order_id or "").strip()
    if not order_id:
        return {}
    out = _get_auth(f"/v1/account/orders/{order_id}", api_token)
    if not isinstance(out, dict) or not out.get("success"):
        return {}
    result = out.get("result") or {}
    return result if isinstance(result, dict) else {}


def place_market_order(
    symbol: str,
    side: str,
    amount: str | float,
    api_token: str | None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    if not api_token:
        return {"status": "failed", "message": "missing api token"}
    payload: dict[str, Any] = {
        "symbol": api_symbol(symbol),
        "type": "MARKET",
        "side": str(side).upper(),
        "quantity": str(amount),
    }
    if client_order_id:
        payload["client_id"] = str(client_order_id)[:64]
    try:
        out = _post_json_auth("/v1/account/orders", api_token, payload)
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}


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
