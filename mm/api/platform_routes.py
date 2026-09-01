"""Platform console API: service health and Grafana embed configuration."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import APIRouter, Depends


def create_platform_router(*, require_auth: Any) -> APIRouter:
    router = APIRouter(prefix="/api/admin/platform", tags=["platform"])

    @router.get("/services")
    async def list_services(_: str = Depends(require_auth)) -> dict[str, Any]:
        targets = _service_targets()
        checks = await asyncio.gather(*(_probe_service(target) for target in targets))
        return {
            "checked_at_ms": int(time.time() * 1000),
            "services": checks,
        }

    @router.get("/grafana")
    async def grafana_config(_: str = Depends(require_auth)) -> dict[str, Any]:
        base_url = os.environ.get("MM_GRAFANA_URL", "http://localhost:3000").rstrip("/")
        tokens = _grafana_public_tokens()
        return {
            "base_url": base_url,
            "embed": bool(tokens) or _grafana_embed_enabled(),
            "dashboards": _grafana_dashboards(base_url, public_tokens=tokens),
            "solo_panels": _grafana_solo_panels(base_url),
        }

    return router


def _service_targets() -> list[dict[str, str]]:
    raw = os.environ.get("MM_PLATFORM_SERVICE_TARGETS", "").strip()
    if raw:
        payload = json.loads(raw)
        if isinstance(payload, list):
            return [
                {
                    "id": str(item.get("id", "")),
                    "name": str(item.get("name", item.get("id", "service"))),
                    "metrics_url": str(item.get("metrics_url", "")),
                    "health_url": str(item.get("health_url", "")),
                }
                for item in payload
                if isinstance(item, dict) and item.get("metrics_url")
            ]
    return [
        {
            "id": "admin",
            "name": "Control Plane (Admin)",
            "metrics_url": "http://127.0.0.1:8080/metrics",
            "health_url": "http://127.0.0.1:8080/api/admin/health",
        },
        {
            "id": "consumer",
            "name": "Consumer Runtime",
            "metrics_url": "http://127.0.0.1:8000/metrics",
            "health_url": "",
        },
        {
            "id": "ingestor",
            "name": "Market Data Ingestor",
            "metrics_url": "http://127.0.0.1:8001/metrics",
            "health_url": "",
        },
        {
            "id": "persistence",
            "name": "Persistence",
            "metrics_url": "http://127.0.0.1:8002/metrics",
            "health_url": "",
        },
        {
            "id": "worker",
            "name": "Strategy Worker",
            "metrics_url": "http://127.0.0.1:8003/metrics",
            "health_url": "",
        },
    ]


async def _probe_service(target: dict[str, str]) -> dict[str, Any]:
    metrics_ok, metrics_detail = await _probe_url(target["metrics_url"], expect_metrics=True)
    health_ok = None
    health_detail = ""
    health_url = target.get("health_url", "")
    if health_url:
        health_ok, health_detail = await _probe_url(health_url, expect_metrics=False)
    status = "up" if metrics_ok or health_ok else "down"
    if metrics_ok:
        status = "up"
    elif health_ok:
        status = "up"
    else:
        status = "down"
    return {
        "id": target["id"],
        "name": target["name"],
        "metrics_url": target["metrics_url"],
        "health_url": health_url or None,
        "status": status,
        "metrics_ok": metrics_ok,
        "health_ok": health_ok,
        "detail": metrics_detail or health_detail or "unreachable",
    }


async def _probe_url(url: str, *, expect_metrics: bool) -> tuple[bool, str]:
    if not url:
        return False, "no url configured"

    def _fetch() -> tuple[bool, str]:
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=2.0) as response:
                body = response.read(4096).decode("utf-8", errors="replace")
                if response.status >= 400:
                    return False, f"http {response.status}"
                if expect_metrics:
                    stripped = body.strip()
                    if stripped and "mm_" not in body and "# HELP" not in body:
                        return False, "unexpected metrics payload"
                return True, f"http {response.status}"
        except urllib.error.HTTPError as exc:
            return False, f"http {exc.code}"
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, ConnectionRefusedError):
                return False, "not running"
            return False, "unreachable"
        except Exception as exc:
            return False, type(exc).__name__

    return await asyncio.to_thread(_fetch)


def _grafana_embed_enabled() -> bool:
    raw = os.environ.get("MM_GRAFANA_EMBED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _grafana_public_tokens() -> dict[str, str]:
    raw = os.environ.get("MM_GRAFANA_PUBLIC_TOKENS", "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(uid).strip(): str(token).strip()
        for uid, token in payload.items()
        if str(uid).strip() and str(token).strip()
    }


def _grafana_dashboards(
    base_url: str,
    *,
    public_tokens: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    tokens = public_tokens if public_tokens is not None else _grafana_public_tokens()
    rows = [
        ("Market Data Quality", "mm-obs-v2-market-data"),
        ("Orderbook Depth", "mm-obs-v2-orderbook-depth"),
        ("Target Order Stems", "mm-obs-v2-target-order-stems"),
        ("Action Pipeline Side Counts", "mm-obs-v2-action-pipeline-side-counts"),
        ("Optimizer Diagnostics", "mm-obs-v2-optimizer"),
        ("Model Parameters", "mm-obs-v2-model-parameters"),
        ("Pipeline Time Profile", "mm-obs-v2-pipeline-time-profile"),
        ("Portfolio And PnL", "mm-obs-v2-portfolio-pnl"),
        ("Exchange Accounts", "mm-obs-v2-exchange-account"),
        ("Fill Spread PnL", "mm-obs-v2-fill-spread-pnl"),
        ("Cluster Health", "mm-obs-v2-cluster-health"),
    ]
    dashboards = []
    for title, uid in rows:
        token = tokens.get(uid, "")
        if token:
            embed_url = f"{base_url}/public-dashboards/{token}?theme=dark"
        else:
            embed_url = f"{base_url}/d/{uid}?orgId=1&theme=dark&kiosk=1"
        dashboards.append(
            {
                "title": title,
                "uid": uid,
                "embed_url": embed_url,
                "open_url": f"{base_url}/d/{uid}?orgId=1&from=now-6h&to=now",
            }
        )
    return dashboards


def _grafana_solo_panels(base_url: str) -> list[dict[str, Any]]:
    return []
