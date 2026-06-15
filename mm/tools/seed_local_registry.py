"""Idempotent Postgres registry seed for local multi-strategy worker dev."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

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

_RUNTIME_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "market_maker.runtime.example.json"
)


def _local_platform_settings() -> dict[str, Any]:
    return {
        "bus": {
            "kind": "redis_streams",
            "redis_url": "redis://localhost:6379/0",
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
        },
        "private": {
            "mode": "rest",
            "poll_interval_ms": 10000,
            "fill_event_source": "exchange_fill_history",
        },
        "fees": {"maker_fee": 1.5e-05},
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
    store.upsert_market_feed(
        feed_id=f"{LOCAL_EXCHANGE}-{LOCAL_SYMBOL}-10000",
        exchange=LOCAL_EXCHANGE,
        symbol=LOCAL_SYMBOL,
        interval_ms=10_000,
        mode="rest",
    )
    store.upsert_worker_assignment(
        assignment_id=LOCAL_WORKER_ASSIGNMENT_ID,
        worker_id=worker_id,
        strategy_ids=[strategy_id],
        lease_ttl_seconds=300,
        status="active",
    )
    return True


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
    try:
        store.get_active_worker_assignment(worker_id)
    except RegistryNotFoundError:
        seed_local_worker_registry(store, worker_id=worker_id)
        return True
    return False
