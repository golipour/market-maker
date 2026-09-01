"""Persistence helpers for cluster health alerts.

Stores a small rolling history of state changes in ``mm_alerts``. The cluster
health monitor writes one row whenever ``cluster_state`` is not ``ok`` and
the alert fingerprint changes. The UI and Grafana dashboard read this table
to render a per-actor health timeline.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

VALID_SEVERITIES = ("ok", "degraded", "critical")


def _canonical_fingerprint(parts: Iterable[tuple[str, Any]]) -> str:
    payload = json.dumps(list(parts), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ClusterAlert:
    severity: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    opened_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def fingerprint(self) -> str:
        actors = self.detail.get("actors") or []
        actor_keys = [
            (
                str(a.get("actor_id") or ""),
                str(a.get("severity") or ""),
                ";".join(sorted(a.get("reasons") or ())),
            )
            for a in actors
        ]
        return _canonical_fingerprint(
            [("severity", self.severity), ("actors", actor_keys)]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "summary": self.summary,
            "detail": self.detail,
            "fingerprint": self.fingerprint(),
            "opened_at_ms": self.opened_at_ms,
        }


def latest_alert_fingerprint(alert: ClusterAlert) -> str:
    return alert.fingerprint()


class AlertStore:
    """Thin wrapper around the postgres store for ``mm_alerts``."""

    def __init__(self, postgres_store: Any) -> None:
        self._store = postgres_store

    def ensure_schema(self) -> None:
        with self._store.connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mm_alerts (
                        id BIGSERIAL PRIMARY KEY,
                        severity TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        detail JSONB NOT NULL DEFAULT '{}'::jsonb,
                        fingerprint TEXT NOT NULL,
                        opened_at_ms BIGINT NOT NULL,
                        resolved_at_ms BIGINT
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS mm_alerts_open_idx ON mm_alerts (resolved_at_ms, severity)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS mm_alerts_fingerprint_idx ON mm_alerts (fingerprint)"
                )
            conn.commit()

    def insert(self, alert: ClusterAlert) -> int:
        with self._store.connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO mm_alerts (severity, summary, detail, fingerprint, opened_at_ms)
                    VALUES (%s, %s, %s::jsonb, %s, %s)
                    RETURNING id
                    """,
                    (
                        alert.severity,
                        alert.summary,
                        json.dumps(alert.detail, default=str),
                        alert.fingerprint(),
                        alert.opened_at_ms,
                    ),
                )
                row = cursor.fetchone()
            conn.commit()
        return int(row[0]) if row else 0

    def latest_open(self, *, severity: str | None = None) -> ClusterAlert | None:
        with self._store.connection() as conn:  # type: ignore[attr-defined]
            query = (
                "SELECT id, severity, summary, detail, fingerprint, opened_at_ms "
                "FROM mm_alerts WHERE resolved_at_ms IS NULL"
            )
            params: tuple[Any, ...] = ()
            if severity is not None:
                query += " AND severity = %s"
                params = (severity,)
            query += " ORDER BY opened_at_ms DESC LIMIT 1"
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_alert(row)

    def resolve_open(self, *, severity_at_or_below: str) -> int:
        threshold = _severity_rank(severity_at_or_below)
        if threshold < 0:
            return 0
        with self._store.connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE mm_alerts
                    SET resolved_at_ms = %s
                    WHERE resolved_at_ms IS NULL
                      AND (
                        CASE severity
                          WHEN 'critical' THEN 2
                          WHEN 'degraded' THEN 1
                          ELSE 0
                        END
                      ) <= %s
                    RETURNING id
                    """,
                    (int(time.time() * 1000), threshold),
                )
                rows = cursor.fetchall()
            conn.commit()
        return len(rows)

    def history(self, *, limit: int = 50) -> list[ClusterAlert]:
        with self._store.connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, severity, summary, detail, fingerprint, opened_at_ms
                    FROM mm_alerts
                    ORDER BY opened_at_ms DESC
                    LIMIT %s
                    """,
                    (int(limit),),
                )
                rows = cursor.fetchall()
        return [_row_to_alert(row) for row in rows]


def _severity_rank(severity: str) -> int:
    return {"ok": 0, "degraded": 1, "critical": 2}.get(severity, -1)


def _row_to_alert(row: Any) -> ClusterAlert:
    detail_raw = row[3]
    if isinstance(detail_raw, (bytes, bytearray)):
        detail_raw = detail_raw.decode("utf-8")
    if isinstance(detail_raw, str):
        try:
            detail = json.loads(detail_raw)
        except json.JSONDecodeError:
            detail = {}
    else:
        detail = detail_raw or {}
    return ClusterAlert(
        severity=str(row[1]),
        summary=str(row[2]),
        detail=detail,
        opened_at_ms=int(row[5]),
    )
