"""Postgres connection helper for the cluster health alert store.

Kept separate from :class:`mm.persistence.postgres.PostgresEventWriter` because
the writer is event-stream oriented and the monitor runs a tiny ``SELECT``/
``INSERT`` workload from a synchronous context. The helper opens a short-lived
``psycopg`` connection per call; that is fine for the monitor's tick rate.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

try:
    import psycopg
except ImportError:  # pragma: no cover - environment fallback
    psycopg = None  # type: ignore[assignment]


class PostgresConnectionError(RuntimeError):
    pass


def build_postgres_store(*, database_url: str) -> "AlertsPostgresStore":
    if not database_url:
        raise PostgresConnectionError("MM_DATABASE_URL is not configured")
    if psycopg is None:
        raise PostgresConnectionError("psycopg is not installed in this environment")
    return AlertsPostgresStore(database_url=database_url)


class AlertsPostgresStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    @contextlib.contextmanager
    def connection(self) -> Iterator[Any]:
        conn = psycopg.connect(self.database_url, autocommit=False)
        try:
            yield conn
        finally:
            conn.close()
