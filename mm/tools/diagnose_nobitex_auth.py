"""Print resolved Nobitex auth mode without exposing secrets."""

from __future__ import annotations

import argparse
import json

from mm.env_loader import load_mm_env
from mm.registry.credentials import credentials_from_ref, is_stored_credentials, private_credentials_from_ref
from mm.registry.store import build_registry_store
from mm.runtime.config import PrivateRuntimeConfig, runtime_config_from_mapping
from mm.registry.runtime_config import runtime_config_mapping_from_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-id", default="account-nobitex-main")
    parser.add_argument("--strategy-id", default="mpc-usdt-local")
    args = parser.parse_args()

    load_mm_env()
    store = build_registry_store()
    account = store.get_account(args.account_id)
    print(f"account={account.account_id} exchange={account.exchange}")
    print(f"stored_format={'v2' if is_stored_credentials(account.credential_ref) else 'legacy_env'}")
    if is_stored_credentials(account.credential_ref):
        creds = credentials_from_ref(account.credential_ref)
        print(f"stored_auth_mode={creds.auth_mode if creds else '?'} api_secret_set={bool(creds and creds.api_secret)}")
    else:
        print(f"legacy_ref={account.credential_ref!r}")
        print("legacy_env_keys=", json.dumps(private_credentials_from_ref(account.credential_ref, exchange=account.exchange)))

    strategy = store.get_strategy(args.strategy_id)
    preset = store.get_preset(strategy.preset_id)
    exchanges = {row.exchange: row for row in store.list_exchanges()}
    platform = store.get_platform_settings()
    mapping = runtime_config_mapping_from_registry(
        strategy,
        account=account,
        preset=preset,
        exchanges=exchanges,
        platform=platform,
    )
    runtime = runtime_config_from_mapping(mapping)
    private = runtime.local_private
    assert private is not None
    exchange = account.exchange
    auth_mode = private.resolved_auth_mode(exchange)
    creds = private.resolved_api_credentials(exchange)
    cred_kind = type(creds).__name__ if not isinstance(creds, str) else "token_string"
    print(f"resolved_auth_mode={auth_mode} credential_type={cred_kind}")
    print(f"api_key_set={bool(private.resolved_api_key(exchange) or private.resolved_api_token(exchange))}")
    print(f"api_secret_set={bool(private.resolved_api_secret(exchange))}")
    if cred_kind == "token_string":
        print("DIAGNOSIS: using legacy Token auth — expect 401 if api_key value is an Ed25519 public key")
    elif cred_kind == "NobitexApiKeyAuth":
        print("DIAGNOSIS: using Ed25519 signed_key auth — adapter path is correct")


if __name__ == "__main__":
    main()
