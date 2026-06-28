from __future__ import annotations

import pytest

from mm.exchanges.auth import (
    exchange_auth_spec,
    resolve_exchange_auth,
    validate_credentials_for_exchange,
)
from mm.registry.credentials import (
    AccountCredentials,
    credentials_from_ref,
    credentials_to_ref,
    is_legacy_env_ref,
    is_stored_credentials,
    merge_credentials_update,
    merge_private_config,
    private_credentials_from_ref,
)


def test_private_credentials_from_ref_plain_env_name() -> None:
    assert private_credentials_from_ref("NOBITEX_API_TOKEN") == {
        "api_token_env": "NOBITEX_API_TOKEN"
    }


def test_private_credentials_from_ref_env_prefix() -> None:
    assert private_credentials_from_ref("env:WALLEX_API_TOKEN") == {
        "api_token_env": "WALLEX_API_TOKEN"
    }


def test_private_credentials_from_ref_json_object() -> None:
    assert private_credentials_from_ref(
        '{"api_token_env":"TABDEAL_API_TOKEN","api_private_key_env":"TABDEAL_API_PRIVATE_KEY"}'
    ) == {
        "api_token_env": "TABDEAL_API_TOKEN",
        "api_private_key_env": "TABDEAL_API_PRIVATE_KEY",
        "api_secret_env": "TABDEAL_API_PRIVATE_KEY",
    }


def test_stored_credentials_round_trip() -> None:
    creds = AccountCredentials(
        auth_mode="signed_key",
        api_key="public-key",
        api_secret="private-key",
    )
    ref = credentials_to_ref(creds)
    assert is_stored_credentials(ref)
    assert not is_legacy_env_ref(ref)
    parsed = credentials_from_ref(ref)
    assert parsed == creds


def test_merge_credentials_update_keeps_existing_secret() -> None:
    existing = credentials_to_ref(
        AccountCredentials(auth_mode="signed_key", api_key="pub", api_secret="sec")
    )
    merged = merge_credentials_update(
        existing,
        exchange="nobitex",
        auth_mode="signed_key",
        api_key="pub-new",
        api_secret="********",
    )
    assert merged.api_key == "pub-new"
    assert merged.api_secret == "sec"


def test_merge_private_config_inline_stored_credentials() -> None:
    ref = credentials_to_ref(
        AccountCredentials(
            auth_mode="signed_key",
            api_key="public",
            api_secret="secret",
        )
    )
    merged = merge_private_config(None, exchange="nobitex", credential_ref=ref)
    assert merged["auth_mode"] == "signed_key"
    assert merged["api_key"] == "public"
    assert merged["api_secret"] == "secret"
    assert "api_token_env" not in merged


def test_merge_private_config_strips_inline_secrets_from_base() -> None:
    merged = merge_private_config(
        {
            "mode": "rest",
            "api_token": "inline-secret",
            "poll_interval_ms": 5000,
        },
        exchange="nobitex",
        credential_ref="NOBITEX_API_TOKEN",
    )
    assert merged["mode"] == "rest"
    assert merged["api_token_env"] == "NOBITEX_API_TOKEN"
    assert "api_token" not in merged


def test_wallex_auth_spec() -> None:
    spec = exchange_auth_spec("wallex")
    assert spec.default_mode == "api_key"
    assert resolve_exchange_auth("wallex", auth_mode="api_key", api_key="abc") == "abc"


def test_nobitex_signed_auth_builds_auth_object() -> None:
    from mm.exchanges.nobitex.private_rest import NobitexApiKeyAuth

    auth = resolve_exchange_auth(
        "nobitex",
        auth_mode="signed_key",
        api_key="public",
        api_secret="private",
    )
    assert isinstance(auth, NobitexApiKeyAuth)
    assert auth.public_key == "public"
    assert auth.private_key == "private"


def test_validate_kucoin_requires_passphrase() -> None:
    with pytest.raises(ValueError, match="api_passphrase"):
        validate_credentials_for_exchange(
            "kucoin",
            auth_mode="signed_key_passphrase",
            api_key="k",
            api_secret="s",
            api_passphrase="",
        )
