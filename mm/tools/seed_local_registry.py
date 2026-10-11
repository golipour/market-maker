"""Idempotent Postgres registry seed for local multi-strategy worker dev."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from mm.env_loader import load_mm_env
from mm.registry.memory import RegistryNotFoundError
from mm.registry.store import RegistryStore, build_registry_store, registry_database_url

LOCAL_WORKER_ID = "worker-1"
LOCAL_WORKER_ASSIGNMENT_ID = "worker-1-assignment"
LOCAL_STRATEGY_ID = "mpc-usdt-local"
LOCAL_PRESET_ID = "thin_local_liquidity_weighted_mpc"
LOCAL_ACCOUNT_ID = "account-nobitex-main"
LOCAL_EXCHANGE = "nobitex"
LOCAL_SYMBOL = "USDT_IRT"
LOCAL_ACTOR_ID = "mpc-controller"
BITWANA_ACCOUNT_ID = "account-bitwana-main"
BITWANA_EXCHANGE = "bitwana"

_RUNTIME_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "market_maker.runtime.example.json"
)


def _local_platform_settings() -> dict[str, Any]:
    redis_url = os.environ.get("MM_BUS_REDIS_URL", "").strip() or "redis://localhost:6379/0"
    return {
        "bus": {
            "kind": "redis_streams",
            "redis_url": redis_url,
            "stream_prefix": "mm",
            "consumer_group": "mm-worker",
            "consumer_name": "worker-1",
            "max_stream_length": 100000,
            "max_queue_size": 1024,
            "backpressure_policy": "block",
        },
        "persistence": {"enabled": False},
        "observability": {
            "service_name": "market-maker-worker",
            "metrics_enabled": True,
            "metrics_port": 8003,
        },
        "scheduler": {"max_concurrency": 8},
    }


def _local_exchange_config() -> dict[str, Any]:
    return {
        "market_data": {
            "mode": "rest",
            "market_data_source": "bus",
            "poll_interval_ms": 2000,
            "depth_limit": 100,
            "trades_limit": 200,
        },
        "private": {
            "mode": "rest",
            "poll_interval_ms": 10000,
            "fill_event_source": "exchange_fill_history",
        },
        "fees": {"maker_fee": 1.5e-05},
    }


def _bitwana_exchange_config() -> dict[str, Any]:
    return {
        "market_data": {
            "mode": "rest",
            "market_data_source": "bus",
            "poll_interval_ms": 2000,
            "depth_limit": 200,
            "trades_limit": 200,
        },
        "private": {
            "mode": "rest",
            "auth_mode": "signed_key",
            "poll_interval_ms": 10000,
            "fill_event_source": "exchange_fill_history",
        },
        "fees": {"maker_fee": 0.002},
    }


def _strip_doc_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_doc_keys(nested)
            for key, nested in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_strip_doc_keys(item) for item in value]
    return value


def _example_preset_config(preset_id: str = LOCAL_PRESET_ID) -> dict[str, Any]:
    payload = json.loads(_RUNTIME_EXAMPLE_PATH.read_text())
    presets = payload.get("strategy_presets")
    if not isinstance(presets, dict):
        raise ValueError("runtime example missing strategy_presets")
    preset = presets.get(preset_id)
    if not isinstance(preset, dict):
        raise ValueError(f"runtime example missing strategy_presets.{preset_id}")
    return _strip_doc_keys(copy.deepcopy(preset))


def _local_preset_config() -> dict[str, Any]:
    preset = _example_preset_config()
    controller = preset.get("controller")
    if isinstance(controller, dict):
        controller["dry_run"] = True
        controller.pop("actor_id", None)
    return preset


def seed_local_worker_registry(
    store: RegistryStore,
    *,
    worker_id: str = LOCAL_WORKER_ID,
    strategy_id: str = LOCAL_STRATEGY_ID,
) -> bool:
    """Upsert local dev registry rows. Returns True if rows were written."""
    platform = _local_platform_settings()
    store.upsert_platform_settings(
        bus=platform["bus"],
        persistence=platform["persistence"],
        observability=platform["observability"],
        scheduler=platform["scheduler"],
    )
    store.upsert_exchange(
        exchange=LOCAL_EXCHANGE,
        config=_local_exchange_config(),
    )
    store.upsert_account(
        account_id=LOCAL_ACCOUNT_ID,
        exchange=LOCAL_EXCHANGE,
        credential_ref="NOBITEX_API_TOKEN",
    )
    store.upsert_preset(
        preset_id=LOCAL_PRESET_ID,
        config=_local_preset_config(),
    )
    wallet_id = f"{strategy_id}-wallet"
    store.upsert_allocation(
        allocation_id=wallet_id,
        account_id=LOCAL_ACCOUNT_ID,
        quote_asset="IRT",
        base_asset="USDT",
        allocated_quote=50_000_000.0,
        allocated_base=10_000.0,
    )
    store.upsert_strategy(
        strategy_id=strategy_id,
        actor_id=LOCAL_ACTOR_ID,
        account_id=LOCAL_ACCOUNT_ID,
        preset_id=LOCAL_PRESET_ID,
        quoting_exchange=LOCAL_EXCHANGE,
        liquidity_exchange=LOCAL_EXCHANGE,
        symbol=LOCAL_SYMBOL,
        status="active",
        wallet_id=wallet_id,
        ledger_source="virtual",
    )
    store.upsert_market_source(
        source_id=f"{LOCAL_EXCHANGE}-{LOCAL_SYMBOL}",
        exchange=LOCAL_EXCHANGE,
        symbol=LOCAL_SYMBOL,
    )
    store.upsert_market_feed(
        feed_id=f"{LOCAL_EXCHANGE}-{LOCAL_SYMBOL}-10000",
        source_id=f"{LOCAL_EXCHANGE}-{LOCAL_SYMBOL}",
        interval_ms=10_000,
    )
    store.upsert_worker_assignment(
        assignment_id=LOCAL_WORKER_ASSIGNMENT_ID,
        worker_id=worker_id,
        strategy_ids=[strategy_id],
        lease_ttl_seconds=300,
        status="active",
    )
    seed_bitwana_venue(store)
    return True


def seed_bitwana_venue(store: RegistryStore) -> None:
    """Upsert Bitwana exchange, account, and USDT_IRT market feed."""

    store.upsert_exchange(
        exchange=BITWANA_EXCHANGE,
        config=_bitwana_exchange_config(),
    )
    try:
        store.get_account(BITWANA_ACCOUNT_ID)
    except RegistryNotFoundError:
        store.upsert_account(
            account_id=BITWANA_ACCOUNT_ID,
            exchange=BITWANA_EXCHANGE,
            credential_ref="BITWANA_API_TOKEN",
        )
    store.upsert_market_source(
        source_id=f"{BITWANA_EXCHANGE}-{LOCAL_SYMBOL}",
        exchange=BITWANA_EXCHANGE,
        symbol=LOCAL_SYMBOL,
    )
    store.upsert_market_feed(
        feed_id=f"{BITWANA_EXCHANGE}-{LOCAL_SYMBOL}-10000",
        source_id=f"{BITWANA_EXCHANGE}-{LOCAL_SYMBOL}",
        interval_ms=10_000,
    )


def ensure_local_worker_registry(
    *,
    database_url: str | None = None,
    worker_id: str = LOCAL_WORKER_ID,
) -> bool:
    """Ensure an active worker assignment exists; seed when missing.

    Returns True when seeding ran, False when an active assignment already existed.
    """
    url = (database_url or registry_database_url() or "").strip()
    if not url:
        raise RuntimeError(
            "MM_DATABASE_URL is not set; cannot seed registry for multi-strategy worker"
        )
    store = build_registry_store(database_url=url)
    seeded = False
    try:
        store.get_active_worker_assignment(worker_id)
    except RegistryNotFoundError:
        seed_local_worker_registry(store, worker_id=worker_id)
        seeded = True
    seed_bitwana_venue(store)
    return seeded


def _apply_bus_redis_url(store: RegistryStore, redis_url: str) -> None:
    platform = store.get_platform_settings()
    bus = dict(platform.bus)
    bus["redis_url"] = redis_url
    store.upsert_platform_settings(bus=bus)


def main() -> None:
    load_mm_env()
    seeded = ensure_local_worker_registry()
    redis_url = os.environ.get("MM_BUS_REDIS_URL", "").strip()
    if redis_url:
        store = build_registry_store()
        _apply_bus_redis_url(store, redis_url)
    print(f"seeded={seeded} redis_url={redis_url or 'unchanged'}")


if __name__ == "__main__":
    main()

