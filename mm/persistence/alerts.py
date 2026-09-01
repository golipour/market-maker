"""Postgres connection helper for the cluster health alert store.

Uses ``psycopg2`` to match the rest of the cluster (the event writer also uses
psycopg2). The monitor opens a short-lived connection per call; that is fine
at its 30s tick rate.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

try:
    import psycopg2
except ImportError:  # pragma: no cover - environment fallback
    psycopg2 = None  # type: ignore[assignment]


class PostgresConnectionError(RuntimeError):
    pass


def build_postgres_store(*, database_url: str) -> "AlertsPostgresStore":
    if not database_url:
        raise PostgresConnectionError("MM_DATABASE_URL is not configured")
    if psycopg2 is None:
        raise PostgresConnectionError("psycopg2 is not installed in this environment")
    return AlertsPostgresStore(database_url=database_url)


class AlertsPostgresStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    @contextlib.contextmanager
    def connection(self) -> Iterator[Any]:
        conn = psycopg2.connect(self.database_url)
        try:
            yield conn
        finally:
            conn.close()
