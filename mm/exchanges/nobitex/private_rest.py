"""Synchronous Nobitex private REST helpers (no trade_stats)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import time
from dataclasses import dataclass
from typing import Any

import requests

BASE_URL = "https://apiv2.nobitex.ir"


@dataclass(frozen=True, slots=True)
class NobitexApiKeyAuth:
    """Nobitex signed API-key credentials.

    ``public_key`` is sent as ``Nobitex-Key``. ``private_key`` is an Ed25519
    private key, preferably PEM encoded; raw hex/base64 32-byte private keys are
    accepted for local tooling.
    """

    public_key: str
    private_key: str


NobitexAuth = str | NobitexApiKeyAuth


def nobitex_wire_client_order_id(value: str | None, *, max_len: int = 32) -> str | None:
    """Public alias for mapping runtime client order ids to Nobitex wire ``clientOrderId``."""

    return _nobitex_client_order_id(value, max_len=max_len)


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


def _get_auth(path: str, auth: NobitexAuth, params: dict[str, Any] | None = None) -> Any:
    r = _request_auth("GET", path, auth, params=params)
    r.raise_for_status()
    return r.json()


def _post_auth(path: str, auth: NobitexAuth, data: dict[str, Any] | None = None) -> Any:
    r = _request_auth("POST", path, auth, data=data or {})
    if not r.ok:
        detail: Any
        try:
            detail = r.json()
        except Exception:
            detail = (r.text or "")[:4000]
        raise RuntimeError(f"Nobitex HTTP {r.status_code} {path}: {detail}")
    return r.json()


def _request_auth(
    method: str,
    path: str,
    auth: NobitexAuth,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> requests.Response:
    method_upper = method.upper()
    url = f"{BASE_URL}{path}"
    session = requests.Session()
    if isinstance(auth, NobitexApiKeyAuth):
        request = requests.Request(
            method_upper,
            url,
            params=params or {},
            json=data if method_upper != "GET" else None,
            headers={"Content-Type": "application/json"},
        )
        prepared = session.prepare_request(request)
        body = _prepared_body_text(prepared)
        prepared.headers.update(_signed_headers(auth, method_upper, prepared.path_url, body))
    else:
        request = requests.Request(
            method_upper,
            url,
            params=params or {},
            data=data or {},
            headers={"Authorization": _token_authorization_header(auth)},
        )
        prepared = session.prepare_request(request)
    return session.send(prepared, timeout=15)


def _token_authorization_header(api_token: str) -> str:
    raw = str(api_token or "").strip()
    if raw.lower().startswith(("token ", "bearer ")):
        return raw
    return f"Token {raw}"


def _signed_headers(auth: NobitexApiKeyAuth, method: str, path_url: str, body: str) -> dict[str, str]:
    timestamp = str(int(time.time()))
    message = f"{timestamp}{method.upper()}{path_url}{body}".encode("utf-8")
    return {
        "Nobitex-Key": auth.public_key.strip(),
        "Nobitex-Signature": _ed25519_signature_b64(auth.private_key, message),
        "Nobitex-Timestamp": timestamp,
    }


def _prepared_body_text(prepared: requests.PreparedRequest) -> str:
    body = prepared.body or b""
    if isinstance(body, bytes):
        return body.decode("utf-8")
    return str(body)


def _ed25519_signature_b64(private_key: str, message: bytes) -> str:
    key = str(private_key or "").strip().replace("\\n", "\n")
    if not key:
        raise RuntimeError("Nobitex API-key auth requires api_private_key")
    raw_key = _raw_private_key_bytes(key)
    if raw_key is not None:
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError as exc:
            raise RuntimeError("Nobitex API-key auth requires the 'cryptography' package") from exc
        signature = ed25519.Ed25519PrivateKey.from_private_bytes(raw_key).sign(message)
        return base64.b64encode(signature).decode("ascii")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:
        raise RuntimeError("Nobitex API-key auth requires the 'cryptography' package") from exc
    loaded = serialization.load_pem_private_key(key.encode("utf-8"), password=None)
    if not isinstance(loaded, ed25519.Ed25519PrivateKey):
        raise RuntimeError("Nobitex API-key private key must be an Ed25519 private key")
    signature = loaded.sign(message)
    return base64.b64encode(signature).decode("ascii")


def _raw_private_key_bytes(value: str) -> bytes | None:
    compact = "".join(value.split())
    if "BEGIN" in compact:
        return None
    try:
        decoded = bytes.fromhex(compact)
        if len(decoded) == 32:
            return decoded
    except ValueError:
        pass
    try:
        decoded = base64.b64decode(compact, validate=True)
        if len(decoded) == 32:
            return decoded
    except (binascii.Error, ValueError):
        pass
    return None


def get_balances(api_token: NobitexAuth | None) -> dict[str, dict[str, float]]:
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
                    from mm.core.wallet import wallet_legs_from_available_and_locked

                    free = _safe_float(w.get("activeBalance", w.get("balance", 0)))
                    locked = _safe_float(w.get("blockedBalance", 0))
                    total = _safe_float(w.get("balance", 0.0))
                    result[asset] = wallet_legs_from_available_and_locked(
                        free,
                        locked,
                        total=total if total > 0.0 else None,
                    )
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
            from mm.core.wallet import wallet_legs_from_available_and_locked

            # v2 ``balance`` is available; ``blocked`` is locked in open orders.
            free = _safe_float(w.get("balance", 0))
            locked = _safe_float(w.get("blocked", 0))
            result[str(asset).upper()] = wallet_legs_from_available_and_locked(free, locked)
    except Exception as exc:
        errors.append(f"/v2/wallets {type(exc).__name__}: {exc}")
    if result:
        return merge_nobitex_wallet_balances_to_internal_toman(result)
    detail = "; ".join(errors[-3:]) or "empty balance response"
    raise RuntimeError(f"Nobitex balance fetch failed: {detail}")


def list_open_orders(symbol: str, api_token: NobitexAuth | None) -> list[dict]:
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


def list_filled_orders(symbol: str, api_token: NobitexAuth | None) -> list[dict]:
    if not api_token:
        return []
    src, dst = split_symbol_pair(symbol)
    try:
        out = _get_auth(
            "/market/orders/list",
            api_token,
            params={
                "status": "done",
                "srcCurrency": src,
                "dstCurrency": dst,
                "details": 2,
                "order": "-created_at",
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Nobitex filled-order fetch failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(out, dict) or out.get("status") != "ok":
        raise RuntimeError(f"Nobitex filled-order fetch failed: status={getattr(out, 'get', lambda *_: None)('status')!r}")
    orders = out.get("orders") or []
    if not isinstance(orders, list):
        raise RuntimeError("Nobitex filled-order fetch failed: orders is not a list")
    return [o for o in orders if isinstance(o, dict)]


def place_market_order(
    symbol: str,
    side: str,
    amount: str | float,
    api_token: NobitexAuth | None,
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
    cid = nobitex_wire_client_order_id(client_order_id)
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
    api_token: NobitexAuth | None,
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
    cid = nobitex_wire_client_order_id(client_order_id)
    if cid:
        payload["clientOrderId"] = cid
    try:
        out = _post_auth("/market/orders/add", api_token, data=payload)
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    return out if isinstance(out, dict) else {"status": "failed", "message": "bad response"}


def cancel_order(order_id: int, api_token: NobitexAuth | None) -> dict[str, Any]:
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
