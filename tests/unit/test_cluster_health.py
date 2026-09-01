from __future__ import annotations

import asyncio

import pytest

from mm.observability.cluster_health import (
    AdminClient,
    cluster_state,
    run_monitor,
)
from mm.alerts import AlertStore, ClusterAlert


class _MemoryAlertStore(AlertStore):
    """In-memory stand-in for the postgres-backed :class:`AlertStore`."""

    def __init__(self) -> None:
        self.rows: list[ClusterAlert] = []
        self.resolved: list[ClusterAlert] = []

    def insert(self, alert: ClusterAlert) -> int:  # type: ignore[override]
        self.rows.append(alert)
        return len(self.rows)

    def latest_open(self, *, severity: str | None = None):  # type: ignore[override]
        for row in reversed(self.rows):
            if row in self.resolved:
                continue
            if severity is None or row.severity == severity:
                return row
        return None

    def resolve_open(self, *, severity_at_or_below: str) -> int:  # type: ignore[override]
        rank = {"ok": 0, "degraded": 1, "critical": 2}.get(severity_at_or_below, -1)
        count = 0
        for row in self.rows:
            if row in self.resolved:
                continue
            r = {"ok": 0, "degraded": 1, "critical": 2}.get(row.severity, -1)
            if r <= rank:
                self.resolved.append(row)
                count += 1
        return count


def _overview(*, trading: str, data_status: str, tradeable: bool = True, last_error_kind: str | None = None) -> dict:
    health: dict = {
        "data_status": data_status,
        "last_heartbeat_ms": 0,
        "last_heartbeat_age_ms": 0,
        "interval_ms": 30_000,
        "capability_status": "resolved" if tradeable else "missing",
        "tradeable": tradeable,
        "last_error": None,
    }
    if last_error_kind:
        health["last_error"] = {
            "kind": last_error_kind,
            "message": "boom",
            "frame_id": "frame:1",
            "ts_ms": 0,
        }
    return {
        "checked_at_ms": 0,
        "control": {
            "connected": True,
            "paused_actors": [],
            "allocation_breached_actors": [],
            "paused_venue_roles": [],
            "dry_run": False,
            "kill_switch": False,
            "updated_at_ms": 0,
        },
        "workers": [],
        "strategies": [
            {
                "strategy_id": "s1",
                "actor_id": "bitwana-controller",
                "registry_status": "active",
                "worker_id": "worker-1",
                "worker_live": "alive",
                "trading_status": trading,
                "data_status": data_status,
                "data_health": health,
            }
        ],
    }


def test_cluster_state_critical_when_running_but_data_lost() -> None:
    state = cluster_state(_overview(trading="running", data_status="lost"))
    assert state["overall"] == "critical"
    assert any("Running but data_status=lost" in r for r in state["actors"][0]["reasons"])


def test_cluster_state_degraded_when_stale() -> None:
    state = cluster_state(_overview(trading="running", data_status="stale"))
    assert state["overall"] == "degraded"


def test_cluster_state_degraded_when_actor_untradeable() -> None:
    state = cluster_state(
        _overview(trading="running", data_status="fresh", tradeable=False)
    )
    assert state["overall"] == "degraded"
    assert any("not tradeable" in r for r in state["actors"][0]["reasons"])


def test_cluster_state_degraded_when_last_error_present() -> None:
    state = cluster_state(
        _overview(trading="running", data_status="fresh", last_error_kind="controller_error")
    )
    assert state["overall"] == "degraded"


def test_cluster_state_ok_when_all_fresh_and_tradeable() -> None:
    state = cluster_state(_overview(trading="running", data_status="fresh"))
    assert state["overall"] == "ok"


def test_alert_fingerprint_changes_only_on_state_change() -> None:
    alert_a = ClusterAlert(severity="critical", summary="x", detail={"actors": [{"actor_id": "a", "severity": "critical", "reasons": ["r1"]}]})
    alert_b = ClusterAlert(severity="critical", summary="x", detail={"actors": [{"actor_id": "a", "severity": "critical", "reasons": ["r1"]}]})
    alert_c = ClusterAlert(severity="critical", summary="x", detail={"actors": [{"actor_id": "a", "severity": "critical", "reasons": ["r2"]}]})
    assert alert_a.fingerprint() == alert_b.fingerprint()
    assert alert_a.fingerprint() != alert_c.fingerprint()


class _StubAdmin(AdminClient):
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def fetch_overview(self) -> dict:  # type: ignore[override]
        self.calls += 1
        return self.payload


@pytest.mark.asyncio
async def test_run_monitor_persists_alert_and_resolves_when_ok() -> None:
    payload_critical = _overview(trading="running", data_status="lost")
    payload_ok = _overview(trading="running", data_status="fresh")
    admin = _StubAdmin(payload_critical)
    store = _MemoryAlertStore()
    stop = asyncio.Event()

    async def flip_to_ok() -> None:
        # Allow the first tick to insert a critical alert, then flip to ok so
        # the second tick can resolve it.
        await asyncio.sleep(0.2)
        admin.payload = payload_ok

    flip = asyncio.create_task(flip_to_ok())
    monitor_task = asyncio.create_task(
        run_monitor(
            admin=admin,  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            tick_seconds=0.2,
            stop_event=stop,
        )
    )
    try:
        await asyncio.wait_for(flip, timeout=2.0)
        # Give the second tick enough time to run.
        for _ in range(20):
            if store.resolved:
                break
            await asyncio.sleep(0.05)
    finally:
        stop.set()
        try:
            await asyncio.wait_for(monitor_task, timeout=2.0)
        except asyncio.TimeoutError:
            monitor_task.cancel()
    assert admin.calls >= 2
    assert any(r.severity == "critical" for r in store.rows)
    assert store.resolved, "expected alert to be resolved when state returned to ok"
