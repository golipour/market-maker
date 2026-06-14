"""Admin audit persistence for registry and control-plane actions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def record_admin_audit(
    database_url: str,
    *,
    command_id: str,
    user_id: str,
    command: str,
    status: str,
    payload: dict[str, Any] | None = None,
) -> None:
    import psycopg2

    body = dict(payload or {})
    now = datetime.now(timezone.utc)
    with psycopg2.connect(database_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS admin_audit_log (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL,
                    command_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT '',
                    command TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload JSONB NOT NULL
                )
                """
            )
            cur.execute(
                """
                INSERT INTO admin_audit_log (ts, command_id, user_id, command, status, payload)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    now,
                    command_id,
                    user_id,
                    command,
                    status,
                    json.dumps(body, ensure_ascii=False, allow_nan=False),
                ),
            )
        conn.commit()
