"""Synchronous Nobitex private REST helpers (no trade_stats)."""

from __future__ import annotations

import hashlib
from typing import Any

import requests

BASE_URL = "https://apiv2.nobitex.ir"


def _nobitex_client_order_id(value: str | None, *, max_len: int = 32) -> str | None:
    """Map arbitrary runtime ids to Nobitex-safe ``clientOrderId``.

    Nobitex documents examples like ``order1`` (alphanumeric). Hyphens (``mpc-controller``),
    colons, etc. trigger ``ParseError``. Per `apidocs.nobitex.ir`, the field is up to 32 chars
    and must be unique per user among open/active/inactive orders — a truncated SHA-256 hex
    of the full client key satisfies both.
    """

    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest[:max_len]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def split_symbol_pair(symbol: str) -> tuple[str, str]:
    """Lowercase ``(src, dst)`` currencies for Nobitex private order APIs (IRT pairs use ``rls`` dst)."""

    from mm.exchanges.instrument_wire import build_instrument_wire

    return build_instrument_wire("nobitex", symbol).private_order_currencies()


def merge_nobitex_wallet_balances_to_internal_toman(result: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    from mm.exchanges.instrument_wire import merge_wallet_balances_raw

    return merge_wallet_balances_raw("nobitex", result)


def _get_auth(path: str, api_token: str, params: dict[str, Any] | None = None) -> Any:
    headers = {"Authorization": f"Token {api_token}"}
    r = requests.get(f"{BASE_URL}{path}", headers=headers, params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()


def _post_auth(path: str, api_token: str, data: dict[str, Any] | None = None) -> Any:
    headers = {"Authorization": f"Token {api_token}"}
    r = requests.post(f"{BASE_URL}{path}", headers=headers, data=data or {}, timeout=15)
    if not r.ok:
        detail: Any
        try:
            detail = r.json()
        except Exception:
            detail = (r.text or "")[:4000]
        raise RuntimeError(f"Nobitex HTTP {r.status_code} {path}: {detail}")
    return r.json()


def get_balances(api_token: str | None) -> dict[str, dict[str, float]]:
    if not api_token:
        return {}
    result: dict[str, dict[str, float]] = {}
    errors: list[str] = []
    try:
        out = _get_auth("/users/wallets/list", api_token)
        if isinstance(out, dict) and out.get("status") == "ok":
            wallets = out.get("wallets") or []
            if isinstance(wallets, list):
                for w in wallets:
                    if not isinstance(w, dict):
                        continue
                    asset = str(w.get("currency", "")).upper()
                    if not asset:
                        continue
                    free = _safe_float(w.get("activeBalance", w.get("balance", 0)))
                    locked = _safe_float(w.get("blockedBalance", 0))
                    total = _safe_float(w.get("balance", free + locked))
                    result[asset] = {"free": free, "locked": locked, "total": total}
        else:
            errors.append(f"/users/wallets/list status={getattr(out, 'get', lambda *_: None)('status')!r}")
    except Exception as exc:
        errors.append(f"/users/wallets/list {type(exc).__name__}: {exc}")
    if result:
        return merge_nobitex_wallet_balances_to_internal_toman(result)
    try:
        out = _get_auth("/v2/wallets", api_token)
        if not isinstance(out, dict) or out.get("status") != "ok":
            raise RuntimeError(f"status={getattr(out, 'get', lambda *_: None)('status')!r}")
        wallets = out.get("wallets") or {}
        if not isinstance(wallets, dict):
            raise RuntimeError("wallets is not an object")
        for asset, w in wallets.items():
            if not isinstance(w, dict):
                continue
            free = _safe_float(w.get("balance", 0))
            locked = _safe_float(w.get("blocked", 0))
            total = free + locked
            result[str(asset).upper()] = {"free": free, "locked": locked, "total": total}
    except Exception as exc:
        errors.append(f"/v2/wallets {type(exc).__name__}: {exc}")
    if result:
        return merge_nobitex_wallet_balances_to_internal_toman(result)
    detail = "; ".join(errors[-3:]) or "empty balance response"
    raise RuntimeError(f"Nobitex balance fetch failed: {detail}")


def list_open_orders(symbol: str, api_token: str | None) -> list[dict]:
    if not api_token:
        return []
    src, dst = split_symbol_pair(symbol)
    try:
        out = _get_auth(
            "/market/orders/list",
            api_token,
            params={"status": "open", "srcCurrency": src, "dstCurrency": dst, "details": 2},
        )
    except Exception as exc:
        raise RuntimeError(f"Nobitex open-order fetch failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(out, dict) or out.get("status") != "ok":
        raise RuntimeError(f"Nobitex open-order fetch failed: status={getattr(out, 'get', lambda *_: None)('status')!r}")
    orders = out.get("orders") or []
    if not isinstance(orders, list):
        raise RuntimeError("Nobitex open-order fetch failed: orders is not a list")
    return [o for o in orders if isinstance(o, dict)]


def place_market_order(
    symbol: str,
    side: str,
    amount: str | float,
    api_token: str | None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    if not api_token:
        return {"status": "failed", "message": "missing api token"}
    from mm.exchanges.instrument_wire import build_instrument_wire

    wire = build_instrument_wire("nobitex", symbol)
    src, dst = wire.private_order_currencies()
    payload: dict[str, Any] = {
        "type": str(side).lower(),
        "execution": "market",
        "srcCurrency": src,
        "dstCurrency": dst,
        "amount": str(amount),
    }
    cid = _nobitex_client_order_id(client_order_id)
    if cid:
        payload["clientOrderId"] = cid
    try:
        out = _post_auth("/market/orders/add", api_token, data=payload)
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
    from mm.exchanges.instrument_wire import build_instrument_wire

    wire = build_instrument_wire("nobitex", symbol)
    src, dst = wire.private_order_currencies()
    api_price = wire.scale_limit_price_to_api(_safe_float(price, 0.0))
    payload: dict[str, Any] = {
        "type": str(side).lower(),
        "execution": "limit",
        "srcCurrency": src,
        "dstCurrency": dst,
        "amount": str(amount),
        "price": str(api_price),
    }
    cid = _nobitex_client_order_id(client_order_id)
    if cid:
        payload["clientOrderId"] = cid
    try:
        out = _post_auth("/market/orders/add", api_token, data=payload)
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}


def cancel_order(order_id: int, api_token: str | None) -> dict[str, Any]:
    if not api_token:
        return {"status": "failed", "message": "missing api token"}
    try:
        out = _post_auth(
            "/market/orders/update-status",
            api_token,
            data={"order": int(order_id), "status": "canceled"},
        )
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}
