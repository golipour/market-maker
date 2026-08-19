from __future__ import annotations

import pytest

from mm.exchanges.auth import resolve_exchange_auth
from mm.runtime.config import PrivateRuntimeConfig


def test_private_runtime_config_resolves_nobitex_signed_key_inline() -> None:
    config = PrivateRuntimeConfig(
        auth_mode="signed_key",
        api_key="public-key",
        api_secret="private-key",
    )
    from mm.exchanges.nobitex.private_rest import NobitexApiKeyAuth

    credentials = config.resolved_api_credentials("nobitex")
    assert isinstance(credentials, NobitexApiKeyAuth)
    assert credentials.public_key == "public-key"


def test_private_runtime_config_resolves_wallex_api_key_inline() -> None:
    config = PrivateRuntimeConfig(auth_mode="api_key", api_key="wallex-key")
    assert config.resolved_api_credentials("wallex") == "wallex-key"


def test_private_runtime_config_legacy_env_token(monkeypatch) -> None:
    monkeypatch.setenv("NOBITEX_API_TOKEN", "legacy-token")
    config = PrivateRuntimeConfig(api_token_env="NOBITEX_API_TOKEN")
    assert config.resolved_api_credentials("nobitex") == "legacy-token"


def test_private_runtime_config_legacy_env_signed_key(monkeypatch) -> None:
    monkeypatch.setenv("NOBITEX_API_TOKEN", "public-key")
    monkeypatch.setenv("NOBITEX_API_PRIVATE_KEY", "private-key")
    config = PrivateRuntimeConfig(
        api_token_env="NOBITEX_API_TOKEN",
        api_private_key_env="NOBITEX_API_PRIVATE_KEY",
    )
    from mm.exchanges.nobitex.private_rest import NobitexApiKeyAuth

    credentials = config.resolved_api_credentials("nobitex")
    assert isinstance(credentials, NobitexApiKeyAuth)


def test_private_runtime_config_api_secret_env_alias(monkeypatch) -> None:
    monkeypatch.setenv("TABDEAL_API_KEY", "key")
    monkeypatch.setenv("TABDEAL_API_SECRET", "secret")
    config = PrivateRuntimeConfig(api_key_env="TABDEAL_API_KEY", api_secret_env="TABDEAL_API_SECRET")
    from mm.exchanges.tabdeal.private_rest import TabdealApiKeyAuth

    credentials = config.resolved_api_credentials("tabdeal")
    assert isinstance(credentials, TabdealApiKeyAuth)


def test_private_runtime_config_resolves_bitwana_jwt(monkeypatch) -> None:
    monkeypatch.setenv("BITWANA_API_TOKEN", "access")
    monkeypatch.setenv("BITWANA_REFRESH_TOKEN", "refresh")
    config = PrivateRuntimeConfig(api_token_env="BITWANA_API_TOKEN")
    from mm.exchanges.bitwana.private_rest import BitwanaJwtAuth

    credentials = config.resolved_api_credentials("bitwana")
    assert isinstance(credentials, BitwanaJwtAuth)
    assert credentials.access_token == "access"
    assert credentials.refresh_token == "refresh"

