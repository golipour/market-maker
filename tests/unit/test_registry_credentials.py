from __future__ import annotations

import pytest

from mm.registry.credentials import merge_private_config, private_credentials_from_ref


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
    }


def test_merge_private_config_strips_inline_secrets() -> None:
    merged = merge_private_config(
        {
            "mode": "rest",
            "api_token": "inline-secret",
            "poll_interval_ms": 5000,
        },
        credential_ref="NOBITEX_API_TOKEN",
    )
    assert merged["mode"] == "rest"
    assert merged["api_token_env"] == "NOBITEX_API_TOKEN"
    assert "api_token" not in merged
