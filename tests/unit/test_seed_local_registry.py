from __future__ import annotations

import pytest

from mm.registry.memory import MemoryRegistryStore, RegistryNotFoundError
from mm.tools.seed_local_registry import (
    LOCAL_STRATEGY_ID,
    LOCAL_WORKER_ID,
    ensure_local_worker_registry,
    seed_local_worker_registry,
)


def test_seed_local_worker_registry_creates_active_assignment() -> None:
    store = MemoryRegistryStore()
    assert seed_local_worker_registry(store) is True

    assignment = store.get_active_worker_assignment(LOCAL_WORKER_ID)
    assert assignment.worker_id == LOCAL_WORKER_ID
    assert assignment.strategy_ids == (LOCAL_STRATEGY_ID,)

    bundle = store.load_runtime_bundle(LOCAL_STRATEGY_ID)
    assert bundle["strategy"].strategy_id == LOCAL_STRATEGY_ID
    assert bundle["exchanges"]["nobitex"].config["market_data"]["market_data_source"] == "bus"


def test_ensure_local_worker_registry_seeds_once(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryRegistryStore()
    monkeypatch.setattr(
        "mm.tools.seed_local_registry.build_registry_store",
        lambda **_: store,
    )
    monkeypatch.setattr(
        "mm.tools.seed_local_registry.registry_database_url",
        lambda: "postgresql://example/test",
    )

    assert ensure_local_worker_registry() is True
    assert ensure_local_worker_registry() is False
    store.get_active_worker_assignment(LOCAL_WORKER_ID)


def test_ensure_local_worker_registry_requires_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mm.tools.seed_local_registry.registry_database_url", lambda: "")
    with pytest.raises(RuntimeError, match="MM_DATABASE_URL"):
        ensure_local_worker_registry()
