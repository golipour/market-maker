#!/usr/bin/env python3
"""Migrate registry Postgres JSON configs to canonical optimizer parameter names."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import psycopg2


def _load_migration_module():
    path = Path(__file__).resolve().parents[1] / "registry" / "optimizer_config_migration.py"
    spec = importlib.util.spec_from_file_location("optimizer_config_migration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load migration module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    database_url = os.environ.get(
        "MM_DATABASE_URL",
        "postgresql://meysam:Welcome123@localhost:5432/postgres",
    )
    migrate_registry_optimizer_config_names = _load_migration_module().migrate_registry_optimizer_config_names
    conn = psycopg2.connect(database_url)
    try:
        updated = migrate_registry_optimizer_config_names(conn)
        conn.commit()
    finally:
        conn.close()
    print(f"migrated {updated} registry config row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
