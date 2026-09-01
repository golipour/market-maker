"""Standalone cluster health monitor.

This service runs as a separate Kubernetes Deployment. Every tick it:

1. Calls ``GET /api/admin/runtime/overview`` on the admin API.
2. Classifies the cluster as ``healthy`` / ``degraded`` / ``critical``.
3. Persists one row per state change in ``mm_alerts``.
4. Posts the alert to a webhook (``MM_ALERT_WEBHOOK`` env var) if configured.

The point is to make the cluster tell you when it is lying. If ``Running`` and
``data_status == lost`` co-exist for more than one tick, that is a page.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from mm.alerts.store import AlertStore, ClusterAlert, latest_alert_fingerprint
from mm.observability.heartbeat import classify_data_status

log = logging.getLogger("mm.cluster_health")

DEFAULT_TICK_SECONDS = 30
ADMIN_REQUEST_TIMEOUT = 8
DEFAULT_WEBHOOK_TIMEOUT = 5


@dataclass(slots=True)
class AdminClient:
    base_url: str
    token: str

    def fetch_overview(self) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/api/admin/runtime/overview"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=ADMIN_REQUEST_TIMEOUT) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def post_webhook(self, webhook_url: str, payload: dict[str, Any]) -> bool:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_WEBHOOK_TIMEOUT) as resp:  # noqa: S310
                return 200 <= resp.status < 300
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            log.warning("webhook_post_failed", extra={"error": str(exc)})
            return False


def _state_for(strategy: dict[str, Any]) -> dict[str, Any]:
    trading = str(strategy.get("trading_status") or "")
    data_status = str(strategy.get("data_status") or "unknown")
    actor_id = str(strategy.get("actor_id") or strategy.get("strategy_id") or "")
    health = strategy.get("data_health") or {}
    last_error = health.get("last_error") if isinstance(health, dict) else None

    severity = "ok"
    reasons: list[str] = []

    if trading == "running" and data_status in ("lost", "unattached"):
        severity = "critical"
        reasons.append(
            f"trading=Running but data_status={data_status} for actor {actor_id}"
        )
    elif data_status == "stale":
        severity = max(severity, "degraded", key=_severity_rank)
        reasons.append(f"data_status=stale for actor {actor_id}")
    elif health and not health.get("tradeable", True):
        severity = max(severity, "degraded", key=_severity_rank)
        reasons.append(
            f"worker reports actor {actor_id} not tradeable: "
            f"capability={health.get('capability_status')!r}"
        )

    if isinstance(last_error, dict) and last_error.get("kind"):
        severity = max(severity, "degraded", key=_severity_rank)
        reasons.append(
            f"last_error={last_error.get('kind')} for actor {actor_id}"
        )

    return {
        "actor_id": actor_id,
        "strategy_id": strategy.get("strategy_id"),
        "trading": trading,
        "data_status": data_status,
        "last_heartbeat_age_ms": health.get("last_heartbeat_age_ms") if isinstance(health, dict) else None,
        "interval_ms": health.get("interval_ms") if isinstance(health, dict) else None,
        "tradeable": health.get("tradeable") if isinstance(health, dict) else None,
        "last_error": last_error,
        "severity": severity,
        "reasons": reasons,
    }


def _severity_rank(severity: str) -> int:
    return {"ok": 0, "degraded": 1, "critical": 2}.get(severity, 0)


def cluster_state(overview: dict[str, Any]) -> dict[str, Any]:
    strategies: list[dict[str, Any]] = overview.get("strategies") or []
    per_actor = [_state_for(s) for s in strategies]
    overall = "ok"
    if any(p["severity"] == "critical" for p in per_actor):
        overall = "critical"
    elif any(p["severity"] == "degraded" for p in per_actor):
        overall = "degraded"
    return {
        "overall": overall,
        "actors": per_actor,
        "checked_at_ms": int(time.time() * 1000),
    }


def _build_alert(state: dict[str, Any]) -> ClusterAlert | None:
    if state["overall"] == "ok":
        return None
    return ClusterAlert(
        severity=state["overall"],
        summary=_summarise(state),
        detail=state,
    )


def _summarise(state: dict[str, Any]) -> str:
    crit = [a for a in state["actors"] if a["severity"] == "critical"]
    if crit:
        return f"{len(crit)} actor(s) Running but data lost: " + ", ".join(
            a["actor_id"] for a in crit
        )
    return f"{state['overall']}: " + "; ".join(
        "; ".join(a["reasons"]) for a in state["actors"] if a["reasons"]
    )


async def run_monitor(
    *,
    admin: AdminClient,
    store: AlertStore,
    tick_seconds: int = DEFAULT_TICK_SECONDS,
    webhook_url: str | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    stop_event = stop_event or asyncio.Event()
    while not stop_event.is_set():
        try:
            overview = await asyncio.to_thread(admin.fetch_overview)
            state = cluster_state(overview)
        except Exception as exc:  # noqa: BLE001
            log.exception("monitor_fetch_failed", extra={"error": str(exc)})
            await asyncio.wait_for(stop_event.wait(), timeout=tick_seconds)
            continue

        alert = _build_alert(state)
        if alert is None:
            try:
                store.resolve_open(severity_at_or_below="critical")
            except Exception as exc:  # noqa: BLE001
                log.exception("monitor_resolve_failed", extra={"error": str(exc)})
        else:
            fingerprint = latest_alert_fingerprint(alert)
            previous = store.latest_open(severity=alert.severity)
            if previous is None or previous.fingerprint != fingerprint:
                try:
                    store.insert(alert)
                except Exception as exc:  # noqa: BLE001
                    log.exception("monitor_insert_failed", extra={"error": str(exc)})
                if webhook_url:
                    ok = await asyncio.to_thread(
                        admin.post_webhook, webhook_url, alert.as_dict()
                    )
                    if not ok:
                        log.warning("webhook_post_failed", extra={"fingerprint": fingerprint})
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=tick_seconds)
        except asyncio.TimeoutError:
            pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mm-cluster-health")
    parser.add_argument("--admin-url", default=os.environ.get("MM_ADMIN_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--admin-token", default=os.environ.get("MM_ADMIN_TOKEN", ""))
    parser.add_argument("--database-url", default=os.environ.get("MM_DATABASE_URL", ""))
    parser.add_argument("--webhook-url", default=os.environ.get("MM_ALERT_WEBHOOK", ""))
    parser.add_argument(
        "--tick-seconds",
        type=int,
        default=int(os.environ.get("TICK_SECONDS", DEFAULT_TICK_SECONDS)),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if not args.admin_token:
        log.error("admin_token_required")
        return 2
    if not args.database_url:
        log.error("database_url_required")
        return 2
    from mm.persistence.alerts import build_postgres_store

    store = build_postgres_store(database_url=args.database_url)
    alert_store = AlertStore(store)
    alert_store.ensure_schema()
    admin = AdminClient(base_url=args.admin_url, token=args.admin_token)
    stop = asyncio.Event()
    try:
        asyncio.run(
            run_monitor(
                admin=admin,
                store=alert_store,
                tick_seconds=args.tick_seconds,
                webhook_url=args.webhook_url or None,
                stop_event=stop,
            )
        )
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
