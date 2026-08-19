"""Account credential storage, parsing, and runtime private-config merge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mm.exchanges.auth import exchange_auth_spec, validate_credentials_for_exchange

_LEGACY_ENV_KEYS = frozenset(
    {
        "api_token_env",
        "api_key_env",
        "api_private_key_env",
        "api_secret_env",
        "api_passphrase_env",
    }
)

_STORED_SECRET_KEYS = ("api_secret", "api_passphrase")
_MASK = "********"


@dataclass(frozen=True, slots=True)
class AccountCredentials:
    auth_mode: str
    api_key: str
    api_secret: str = ""
    api_passphrase: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "auth_mode": self.auth_mode,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "api_passphrase": self.api_passphrase,
        }

    def redacted_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "auth_mode": self.auth_mode,
            "api_key": self.api_key,
            "api_secret_set": bool(self.api_secret),
            "api_passphrase_set": bool(self.api_passphrase),
        }
        if self.api_secret:
            out["api_secret"] = _MASK
        if self.api_passphrase:
            out["api_passphrase"] = _MASK
        return out


def is_stored_credentials(ref: str) -> bool:
    ref = str(ref or "").strip()
    if not ref.startswith("{"):
        return False
    try:
        payload = json.loads(ref)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("v") == 2:
        return True
    if "auth_mode" in payload and any(
        key in payload for key in ("api_key", "api_secret", "api_passphrase")
    ):
        return True
    return False


def is_legacy_env_ref(ref: str) -> bool:
    ref = str(ref or "").strip()
    if not ref:
        return False
    if not ref.startswith("{"):
        return True
    try:
        payload = json.loads(ref)
    except json.JSONDecodeError:
        return True
    if not isinstance(payload, dict):
        return True
    return any(str(key) in _LEGACY_ENV_KEYS for key in payload)


def credentials_from_ref(credential_ref: str) -> AccountCredentials | None:
    ref = str(credential_ref or "").strip()
    if not ref or not is_stored_credentials(ref):
        return None
    payload = json.loads(ref)
    if not isinstance(payload, dict):
        raise ValueError("credential_ref JSON must be an object")
    auth_mode = str(payload.get("auth_mode") or "api_key").strip()
    api_key = str(payload.get("api_key") or "").strip()
    api_secret = str(payload.get("api_secret") or payload.get("api_private_key") or "").strip()
    api_passphrase = str(payload.get("api_passphrase") or "").strip()
    if not api_key:
        raise ValueError("stored credentials require api_key")
    return AccountCredentials(
        auth_mode=auth_mode,
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
    )


def credentials_to_ref(credentials: AccountCredentials) -> str:
    payload = {"v": 2, **credentials.as_dict()}
    return json.dumps(payload, separators=(",", ":"))


def merge_credentials_update(
    existing_ref: str,
    *,
    exchange: str,
    auth_mode: str,
    api_key: str,
    api_secret: str = "",
    api_passphrase: str = "",
) -> AccountCredentials:
    existing = credentials_from_ref(existing_ref)
    secret = str(api_secret or "").strip()
    passphrase = str(api_passphrase or "").strip()
    if secret in ("", _MASK) and existing is not None:
        secret = existing.api_secret
    if passphrase in ("", _MASK) and existing is not None:
        passphrase = existing.api_passphrase
    credentials = AccountCredentials(
        auth_mode=str(auth_mode or "").strip(),
        api_key=str(api_key or "").strip(),
        api_secret=secret,
        api_passphrase=passphrase,
    )
    validate_credentials_for_exchange(
        exchange,
        auth_mode=credentials.auth_mode,
        api_key=credentials.api_key,
        api_secret=credentials.api_secret,
        api_passphrase=credentials.api_passphrase,
    )
    return credentials


def _legacy_env_defaults(exchange: str, token_env: str) -> dict[str, str]:
    """Expand a single legacy token env name into signed-auth env keys when needed."""

    exchange_key = exchange.lower().strip()
    out: dict[str, str] = {"api_token_env": token_env}
    try:
        spec = exchange_auth_spec(exchange_key)
    except ValueError:
        return out
    prefix = exchange_key.upper()
    if exchange_key == "bitwana":
        out["api_secret_env"] = "BITWANA_REFRESH_TOKEN"
        return out
    if "signed_key" in spec.supported_modes or "signed_key_passphrase" in spec.supported_modes:
        out["api_secret_env"] = f"{prefix}_API_SECRET"
        out["api_private_key_env"] = f"{prefix}_API_PRIVATE_KEY"
    if "signed_key_passphrase" in spec.supported_modes:
        out["api_passphrase_env"] = f"{prefix}_API_PASSPHRASE"
    return out


def private_credentials_from_ref(credential_ref: str, *, exchange: str = "") -> dict[str, str]:
    """Map a legacy env reference to ``*_env`` private config keys."""

    ref = str(credential_ref or "").strip()
    if not ref:
        return {}
    if is_stored_credentials(ref):
        credentials = credentials_from_ref(ref)
        assert credentials is not None
        out: dict[str, str] = {
            "auth_mode": credentials.auth_mode,
            "api_key": credentials.api_key,
        }
        if credentials.api_secret:
            out["api_secret"] = credentials.api_secret
        if credentials.api_passphrase:
            out["api_passphrase"] = credentials.api_passphrase
        return out
    if ref.startswith("{"):
        payload = json.loads(ref)
        if not isinstance(payload, dict):
            raise ValueError("credential_ref JSON must be an object")
        out = {
            str(key): str(value).strip()
            for key, value in payload.items()
            if str(key) in _LEGACY_ENV_KEYS and str(value).strip()
        }
        if not out:
            raise ValueError("credential_ref JSON must include at least one *_env key")
        if "api_private_key_env" in out and "api_secret_env" not in out:
            out["api_secret_env"] = out["api_private_key_env"]
        return out
    if ref.startswith("env:"):
        env_name = ref[4:].strip()
        if not env_name:
            raise ValueError("credential_ref env: prefix requires a variable name")
        return _legacy_env_defaults(exchange, env_name) if exchange else {"api_token_env": env_name}
    if exchange:
        return _legacy_env_defaults(exchange, ref)
    return {"api_token_env": ref}


def merge_private_config(
    exchange_private: dict[str, Any] | None,
    *,
    exchange: str,
    credential_ref: str,
) -> dict[str, Any]:
    """Merge exchange private defaults with account credentials."""

    base = dict(exchange_private or {})
    credentials = private_credentials_from_ref(credential_ref, exchange=exchange)
    merged = {**base, **credentials}
    for legacy_key in (
        "api_token",
        "api_key",
        "api_secret",
        "api_private_key",
        "api_passphrase",
        "api_token_env",
        "api_key_env",
        "api_secret_env",
        "api_private_key_env",
        "api_passphrase_env",
    ):
        if legacy_key in base and legacy_key not in credentials:
            merged.pop(legacy_key, None)
    if is_stored_credentials(credential_ref):
        merged.pop("api_token_env", None)
        merged.pop("api_key_env", None)
        merged.pop("api_secret_env", None)
        merged.pop("api_private_key_env", None)
        merged.pop("api_passphrase_env", None)
    return merged


def account_credentials_for_api(credential_ref: str) -> dict[str, Any] | None:
    credentials = credentials_from_ref(credential_ref)
    if credentials is None:
        return None
    return credentials.redacted_dict()
