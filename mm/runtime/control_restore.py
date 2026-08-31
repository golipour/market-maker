"""Restore last RuntimeControlState after a worker restart."""

from __future__ import annotations

import json
from typing import Any

from mm.observability import log_event


def boot_paused_actors(actor_ids: set[str], snapshot: dict[str, Any] | None) -> set[str]:
    """Pause unknown actors; restore the last pause set for known ones."""

    actors = {str(item).strip() for item in actor_ids if str(item).strip()}
    if not snapshot:
        return set(actors)
    previous_actors = {str(item).strip() for item in (snapshot.get("actors") or []) if str(item).strip()}
    previous_paused = {str(item).strip() for item in (snapshot.get("paused_actors") or []) if str(item).strip()}
    if not previous_actors:
        return set(actors)
    return (previous_paused & actors) | (actors - previous_actors)


def load_latest_runtime_control_state(database_url: str | None) -> dict[str, Any] | None:
    url = str(database_url or "").strip()
    if not url:
        return None
    try:
        import psycopg2

        conn = psycopg2.connect(url, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT payload
                    FROM mm_events
                    WHERE topic = 'admin:events'
                      AND event_type = 'RuntimeControlState'
                    ORDER BY ts DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:
        log_event(
            "control_state_restore_failed",
            {"error": type(exc).__name__, "message": str(exc)},
        )
        return None
    if not row:
        return None
    payload = row[0]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict) or payload.get("type") != "RuntimeControlState":
        return None
    return payload
