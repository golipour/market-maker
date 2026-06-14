"""Registry store factory."""

from __future__ import annotations

import os
from typing import TypeAlias

from mm.registry.memory import MemoryRegistryStore
from mm.registry.postgres import PostgresRegistryStore

RegistryStore: TypeAlias = MemoryRegistryStore | PostgresRegistryStore


def registry_database_url() -> str | None:
    return os.environ.get("MM_DATABASE_URL", "").strip() or None


def build_registry_store(*, database_url: str | None = None) -> RegistryStore:
    url = (database_url or registry_database_url() or "").strip()
    if url:
        store = PostgresRegistryStore(url)
        store.ensure_schema()
        return store
    return MemoryRegistryStore()
