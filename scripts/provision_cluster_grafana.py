#!/usr/bin/env python3
"""Provision Market Maker datasources and dashboards on cluster Grafana.

Reads Grafana and Postgres credentials from kubectl (never prints them).
Remaps local docker-compose datasource UIDs to the cluster Grafana UIDs:
  prometheus-mm -> mimir
  loki-mm       -> loki
  postgres-mm   stays postgres-mm (created here)
"""

from __future__ import annotations

import base64
import http.cookiejar
import json
import subprocess
import sys
import urllib.error
import urllib.request
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "grafana" / "observability-v2"
KUBECONFIG = None  # inherit env
GRAFANA_URL = "http://192.168.41.83:30081"
POSTGRES_HOST = "postgres.market-maker.svc.cluster.local:5432"
FOLDER_TITLE = "Market Maker"
PUBLIC_DASHBOARD_UIDS = (
    "mm-obs-v2-market-data",
    "mm-obs-v2-orderbook-depth",
    "mm-obs-v2-target-order-stems",
    "mm-obs-v2-action-pipeline-side-counts",
    "mm-obs-v2-optimizer",
    "mm-obs-v2-model-parameters",
    "mm-obs-v2-pipeline-time-profile",
    "mm-obs-v2-portfolio-pnl",
    "mm-obs-v2-exchange-account",
    "mm-obs-v2-fill-spread-pnl",
)
UID_REMAP = {
    "prometheus-mm": "mimir",
    "loki-mm": "loki",
}
TYPE_REMAP = {
    "postgres": "grafana-postgresql-datasource",
}


def _kubectl_json(*args: str) -> dict:
    cmd = ["kubectl", *args, "-o", "json"]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _secret_data(namespace: str, name: str) -> dict[str, str]:
    payload = _kubectl_json("-n", namespace, "get", "secret", name)
    out: dict[str, str] = {}
    for key, raw in (payload.get("data") or {}).items():
        out[str(key)] = base64.b64decode(raw).decode("utf-8")
    return out


def _remap_uids(value):
    if isinstance(value, dict):
        uid = value.get("uid")
        typ = value.get("type")
        if isinstance(uid, str) and uid in UID_REMAP:
            value = {**value, "uid": UID_REMAP[uid]}
        if isinstance(typ, str) and typ in TYPE_REMAP:
            value = {**value, "type": TYPE_REMAP[typ]}
        return {key: _remap_uids(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_uids(item) for item in value]
    return value


def _grafana_admin() -> tuple[str, str]:
    env_user = os.environ.get("GRAFANA_USER", "").strip()
    env_password = os.environ.get("GRAFANA_PASSWORD", "").strip()
    if env_user and env_password:
        return env_user, env_password
    data = _secret_data("observability", "grafana")
    user = (data.get("admin-user") or "admin").strip()
    password = (data.get("admin-password") or "").strip()
    if not password:
        raise SystemExit("grafana admin-password missing")
    return user, password


def _default_all_multi_vars(dashboard: dict) -> dict:
    """Kiosk embeds keep empty multi-var currents, which filter every SQL panel to nothing."""
    templating = dashboard.get("templating")
    if not isinstance(templating, dict):
        return dashboard
    rows = templating.get("list")
    if not isinstance(rows, list):
        return dashboard
    all_current = {"selected": False, "text": "All", "value": "$__all"}
    patched = []
    for row in rows:
        if (
            isinstance(row, dict)
            and row.get("multi")
            and row.get("name") in {"exchange", "symbol", "actor_id", "run_id"}
        ):
            patched.append({**row, "current": all_current, "includeAll": True})
        else:
            patched.append(row)
    return {**dashboard, "templating": {**templating, "list": patched}}


class GrafanaClient:
    def __init__(self, base_url: str, user: str, password: str):
        self.base_url = base_url.rstrip("/")
        self._user = user
        self._password = password
        token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
        self._auth = f"Basic {token}"
        self._jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self._jar))
        self._use_session = False

    def login(self) -> bool:
        status, payload = self.request(
            "POST",
            "/login",
            {"user": self._user, "password": self._password},
            with_auth=False,
        )
        self._use_session = status in {200, 204}
        return self._use_session

    def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        with_auth: bool = True,
    ) -> tuple[int, dict | list | None]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if with_auth and not self._use_session:
            headers["Authorization"] = self._auth
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with self._opener.open(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw) if raw else None
                return response.status, parsed
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = {"message": raw[:300]}
            return exc.code, parsed

    def get(self, path: str) -> tuple[int, dict | list | None]:
        return self.request("GET", path)

    def post(self, path: str, body: dict) -> tuple[int, dict | list | None]:
        return self.request("POST", path, body)

    def put(self, path: str, body: dict) -> tuple[int, dict | list | None]:
        return self.request("PUT", path, body)

    def patch(self, path: str, body: dict) -> tuple[int, dict | list | None]:
        return self.request("PATCH", path, body)


def _ensure_folder(client: GrafanaClient) -> str:
    status, payload = client.get("/api/folders")
    if status != 200 or not isinstance(payload, list):
        raise SystemExit(f"list folders failed: http {status}")
    for row in payload:
        if isinstance(row, dict) and row.get("title") == FOLDER_TITLE:
            uid = str(row.get("uid") or "")
            print(f"folder exists uid={uid}")
            return uid
    status, created = client.post("/api/folders", {"title": FOLDER_TITLE})
    if status not in {200, 201} or not isinstance(created, dict):
        raise SystemExit(f"create folder failed: http {status} {created}")
    uid = str(created.get("uid") or "")
    print(f"folder created uid={uid}")
    return uid


def _upsert_postgres(client: GrafanaClient, *, user: str, password: str) -> None:
    body = {
        "name": "Market Maker Postgres",
        "type": "grafana-postgresql-datasource",
        "uid": "postgres-mm",
        "access": "proxy",
        "url": POSTGRES_HOST,
        "user": user,
        "isDefault": False,
        "secureJsonData": {"password": password},
        "jsonData": {
            "database": "postgres",
            "sslmode": "disable",
            "postgresVersion": 1700,
            "timescaledb": False,
            "maxOpenConns": 8,
            "maxIdleConns": 2,
            "connMaxLifetime": 300,
        },
    }
    status, existing = client.get("/api/datasources/uid/postgres-mm")
    if status == 200 and isinstance(existing, dict) and existing.get("id") is not None:
        body["id"] = existing["id"]
        status, payload = client.put("/api/datasources/uid/postgres-mm", body)
        if status in {200, 201}:
            action = "updated"
        elif status == 403:
            print("datasource postgres-mm provisioned via configmap (already active)")
            return
        else:
            raise SystemExit(f"postgres datasource failed: http {status} {payload}")
    else:
        status, payload = client.post("/api/datasources", body)
        action = "created"
    if status not in {200, 201}:
        raise SystemExit(f"postgres datasource failed: http {status} {payload}")
    print(f"datasource postgres-mm {action}")


def _import_dashboards(client: GrafanaClient, folder_uid: str) -> None:
    files = sorted(DASHBOARD_DIR.glob("mm-obs-v2-*.json"))
    if not files:
        raise SystemExit(f"no dashboards in {DASHBOARD_DIR}")
    imported = 0
    for path in files:
        dashboard = json.loads(path.read_text(encoding="utf-8"))
        dashboard = _default_all_multi_vars(_remap_uids(dashboard))
        dashboard["id"] = None
        dashboard["uid"] = path.stem
        status, payload = client.post(
            "/api/dashboards/db",
            {
                "dashboard": dashboard,
                "folderUid": folder_uid,
                "overwrite": True,
                "message": "market-maker cluster provision",
            },
        )
        if status not in {200, 201}:
            raise SystemExit(f"import {path.name} failed: http {status} {payload}")
        imported += 1
        print(f"dashboard {path.stem}")
    print(f"imported {imported} dashboards")


def _apply_dashboard_configmaps() -> None:
    files = sorted(DASHBOARD_DIR.glob("mm-obs-v2-*.json"))
    if not files:
        raise SystemExit(f"no dashboards in {DASHBOARD_DIR}")
    for path in files:
        dashboard = _default_all_multi_vars(
            _remap_uids(json.loads(path.read_text(encoding="utf-8")))
        )
        dashboard["id"] = None
        dashboard["uid"] = path.stem
        name = f"grafana-{path.stem}"[:63]
        manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": "market-maker",
                "labels": {
                    "grafana_dashboard": "1",
                    "app.kubernetes.io/part-of": "market-maker",
                    "app.kubernetes.io/component": "grafana-dashboard",
                },
                "annotations": {
                    "grafana_folder": FOLDER_TITLE,
                },
            },
            "data": {f"{path.stem}.json": json.dumps(dashboard, separators=(",", ":"))},
        }
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=json.dumps(manifest),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(f"apply {name} failed: {result.stderr.strip()}")
        print(result.stdout.strip() or name)
    print(f"applied {len(files)} dashboard configmaps")


def _apply_datasource_secret(*, user: str, password: str) -> None:
    """Provision postgres-mm datasource via sidecar Secret in market-maker namespace.

    Grafana sidecar discovers any Secret labeled `grafana_datasource: "1"`
    across all namespaces and reloads automatically within ~1 minute without
    modifying the observability ConfigMap or restarting Grafana.
    """
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "grafana-datasource-market-maker",
            "namespace": "market-maker",
            "labels": {
                "grafana_datasource": "1",
                "app.kubernetes.io/part-of": "market-maker",
            },
        },
        "stringData": {
            "market-maker.yaml": (
                "apiVersion: 1\n"
                "datasources:\n"
                "  - name: PostgreSQL\n"
                "    uid: postgres-mm\n"
                "    type: grafana-postgresql-datasource\n"
                "    access: proxy\n"
                f"    url: {POSTGRES_HOST}\n"
                f"    user: {user}\n"
                "    jsonData:\n"
                "      database: postgres\n"
                "      sslmode: disable\n"
                "      postgresVersion: 1700\n"
                "    secureJsonData:\n"
                f"      password: {password}\n"
            )
        },
    }
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(manifest),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"apply grafana-datasource-market-maker secret failed: {result.stderr.strip()}")
    print(result.stdout.strip() or "secret/grafana-datasource-market-maker applied")


def _ensure_public_dashboards(client: GrafanaClient) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for uid in PUBLIC_DASHBOARD_UIDS:
        status, existing = client.get(f"/api/dashboards/uid/{uid}/public-dashboards")
        token = ""
        public_uid = ""
        enabled = False
        if status == 200 and isinstance(existing, dict):
            token = str(existing.get("accessToken") or "")
            public_uid = str(existing.get("uid") or "")
            enabled = bool(existing.get("isEnabled"))
        if token and enabled:
            tokens[uid] = token
            print(f"public dashboard {uid}")
            continue
        body = {
            "isEnabled": True,
            "share": "public",
            "timeSelectionEnabled": True,
            "annotationsEnabled": False,
        }
        if public_uid:
            status, created = client.patch(
                f"/api/dashboards/uid/{uid}/public-dashboards/{public_uid}",
                body,
            )
        else:
            status, created = client.post(
                f"/api/dashboards/uid/{uid}/public-dashboards",
                body,
            )
        if status not in {200, 201} or not isinstance(created, dict):
            print(f"public dashboard {uid} failed http {status}")
            continue
        token = str(created.get("accessToken") or token)
        if token:
            tokens[uid] = token
            print(f"public dashboard {uid} enabled")
    return tokens


def _write_public_token_configmap(tokens: dict[str, str]) -> None:
    manifest = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "market-maker-grafana-public",
            "namespace": "market-maker",
            "labels": {"app.kubernetes.io/part-of": "market-maker"},
        },
        "data": {"tokens.json": json.dumps(tokens, separators=(",", ":"))},
    }
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(manifest),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"apply grafana public tokens failed: {result.stderr.strip()}")
    print(result.stdout.strip() or "configmap/market-maker-grafana-public")


def _health(client: GrafanaClient) -> None:
    status, payload = client.get("/api/health")
    if status != 200:
        raise SystemExit(f"grafana health failed: http {status} {payload}")
    print("grafana api ok")


def main() -> int:
    _apply_dashboard_configmaps()

    mm = _secret_data("market-maker", "market-maker-secrets")
    pg_user = (mm.get("POSTGRES_USER") or "mm").strip()
    pg_password = (mm.get("POSTGRES_PASSWORD") or "").strip()
    if not pg_password:
        raise SystemExit("POSTGRES_PASSWORD missing")

    _apply_datasource_secret(user=pg_user, password=pg_password)

    api_user, api_password = _grafana_admin()
    client = GrafanaClient(GRAFANA_URL, api_user, api_password)
    _health(client)
    if not client.login():
        print("grafana session login failed; continuing with basic auth")
    _upsert_postgres(client, user=pg_user, password=pg_password)
    folder_uid = _ensure_folder(client)
    _import_dashboards(client, folder_uid)
    status, ds = client.get("/api/datasources/uid/postgres-mm")
    ds_type = ds.get("type") if isinstance(ds, dict) else None
    print(f"datasource postgres-mm http {status} type={ds_type}")
    tokens = _ensure_public_dashboards(client)
    if tokens:
        _write_public_token_configmap(tokens)
        print(f"published {len(tokens)} public dashboards")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
