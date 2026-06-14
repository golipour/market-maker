"""Resolve account credential references into runtime private config fields."""

from __future__ import annotations

import json
from typing import Any


_ALLOWED_PRIVATE_ENV_KEYS = frozenset(
    {
        "api_token_env",
        "api_key_env",
        "api_private_key_env",
    }
)


def private_credentials_from_ref(credential_ref: str) -> dict[str, str]:
    """Map a stored credential reference to ``*_env`` private config keys.

    Supported forms:
    - plain env var name -> ``api_token_env``
    - ``env:VAR_NAME`` -> ``api_token_env``
    - JSON object with allowed ``*_env`` keys
    """

    ref = str(credential_ref or "").strip()
    if not ref:
        return {}
    if ref.startswith("{"):
        payload = json.loads(ref)
        if not isinstance(payload, dict):
            raise ValueError("credential_ref JSON must be an object")
        out = {
            str(key): str(value).strip()
            for key, value in payload.items()
            if str(key) in _ALLOWED_PRIVATE_ENV_KEYS and str(value).strip()
        }
        if not out:
            raise ValueError("credential_ref JSON must include at least one *_env key")
        return out
    if ref.startswith("env:"):
        env_name = ref[4:].strip()
        if not env_name:
            raise ValueError("credential_ref env: prefix requires a variable name")
        return {"api_token_env": env_name}
    return {"api_token_env": ref}


def merge_private_config(
    exchange_private: dict[str, Any] | None,
    *,
    credential_ref: str,
) -> dict[str, Any]:
    """Merge exchange private defaults with account credential references."""

    base = dict(exchange_private or {})
    credentials = private_credentials_from_ref(credential_ref)
    merged = {**base, **credentials}
    merged.pop("api_token", None)
    merged.pop("api_key", None)
    merged.pop("api_private_key", None)
    return merged
