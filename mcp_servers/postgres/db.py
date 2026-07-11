"""Postgres connection helpers for the MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    sslmode: str | None = None

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        url = (
            os.getenv("MM_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
        )
        if url:
            return cls.from_url(url)
        host = os.getenv("PGHOST")
        port = os.getenv("PGPORT", "5432")
        database = os.getenv("PGDATABASE")
        user = os.getenv("PGUSER")
        password = os.getenv("PGPASSWORD", "")
        if not all([host, database, user]):
            raise ValueError(
                "Set MM_DATABASE_URL (or DATABASE_URL) or PGHOST/PGDATABASE/PGUSER"
            )
        return cls(
            host=str(host),
            port=int(port),
            database=str(database),
            user=str(user),
            password=str(password),
            sslmode=os.getenv("PGSSLMODE"),
        )

    @classmethod
    def from_url(cls, url: str) -> DatabaseConfig:
        parsed = urlparse(url)
        if parsed.scheme not in {"postgresql", "postgres"}:
            raise ValueError(f"unsupported database URL scheme: {parsed.scheme!r}")
        database = (parsed.path or "").lstrip("/")
        if not database:
            raise ValueError("database URL must include a database name")
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=database,
            user=parsed.username or "",
            password=parsed.password or "",
            sslmode=os.getenv("PGSSLMODE"),
        )

    def connect(self, *, connect_timeout_s: int = 10) -> psycopg.Connection[Any]:
        kwargs: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "dbname": self.database,
            "user": self.user,
            "password": self.password,
            "connect_timeout": connect_timeout_s,
            "row_factory": dict_row,
        }
        if self.sslmode:
            kwargs["sslmode"] = self.sslmode
        conn = psycopg.connect(**kwargs)
        conn.execute("SET default_transaction_read_only = on")
        conn.execute("SET statement_timeout = '30s'")
        return conn


class PostgresClient:
    def __init__(self, config: DatabaseConfig, *, connect_timeout_s: int = 10) -> None:
        self._config = config
        self._connect_timeout_s = connect_timeout_s

    def run_read_query(self, sql: str, *, limit: int = 500) -> list[dict[str, Any]]:
        normalized = sql.strip().rstrip(";")
        upper = normalized.lstrip().upper()
        if upper.startswith(("SELECT", "WITH")):
            wrapped = f"SELECT * FROM ({normalized}) AS _mcp_q LIMIT %s"
            params: tuple[Any, ...] = (limit,)
        else:
            wrapped = normalized
            params = ()
        with self._config.connect(connect_timeout_s=self._connect_timeout_s) as conn:
            with conn.cursor() as cur:
                cur.execute(wrapped, params)
                if cur.description is None:
                    return []
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def list_tables(self) -> list[dict[str, Any]]:
        sql = """
            SELECT
                schemaname AS schema,
                tablename AS table,
                pg_size_pretty(pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename))) AS size
            FROM pg_tables
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY schemaname, tablename
        """
        with self._config.connect(connect_timeout_s=self._connect_timeout_s) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return [dict(row) for row in cur.fetchall()]

    def describe_table(self, table: str, *, schema: str = "public") -> list[dict[str, Any]]:
        sql = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """
        with self._config.connect(connect_timeout_s=self._connect_timeout_s) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (schema, table))
                return [dict(row) for row in cur.fetchall()]
