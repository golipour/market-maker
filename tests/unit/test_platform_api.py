from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mm.api.admin import create_admin_app, _platform_ui_dist
from mm.api.platform_routes import (
    _grafana_dashboards,
    _grafana_solo_panels,
    _probe_url,
    _service_targets,
)
from mm.bus.memory import InMemoryBus


@pytest.fixture
def platform_client() -> TestClient:
    return TestClient(create_admin_app(bus=InMemoryBus()))


def test_platform_services_returns_probe_list(platform_client: TestClient) -> None:
    with patch("mm.api.platform_routes._probe_url", return_value=(True, "http 200")):
        response = platform_client.get("/api/admin/platform/services")
    assert response.status_code == 200
    body = response.json()
    assert "checked_at_ms" in body
    assert len(body["services"]) >= 5
    assert body["services"][0]["id"] == "admin"


def test_platform_grafana_config(platform_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MM_GRAFANA_URL", "http://grafana.test:3000")
    response = platform_client.get("/api/admin/platform/grafana")
    assert response.status_code == 200
    body = response.json()
    assert body["base_url"] == "http://grafana.test:3000"
    uids = {d["uid"] for d in body["dashboards"]}
    assert uids == {
        "mm-obs-v2-market-data",
        "mm-obs-v2-orderbook-depth",
        "mm-obs-v2-target-order-stems",
        "mm-obs-v2-action-pipeline-side-counts",
        "mm-obs-v2-optimizer",
        "mm-obs-v2-model-parameters",
        "mm-obs-v2-pipeline-time-profile",
        "mm-obs-v2-portfolio-pnl",
        "mm-obs-v2-fill-spread-pnl",
    }
    assert [d["title"] for d in body["dashboards"]] == [
        "Market Data Quality",
        "Orderbook Depth",
        "Target Order Stems",
        "Action Pipeline Side Counts",
        "Optimizer Diagnostics",
        "Model Parameters",
        "Pipeline Time Profile",
        "Portfolio And PnL",
        "Fill Spread PnL",
    ]
    assert body["solo_panels"] == []


def test_grafana_embed_urls_use_kiosk() -> None:
    dashboards = _grafana_dashboards("http://localhost:3000")
    market_data = next(d for d in dashboards if d["uid"] == "mm-obs-v2-market-data")
    assert "kiosk=1" in market_data["embed_url"]
    assert _grafana_solo_panels("http://localhost:3000") == []


def test_service_targets_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {
            "id": "custom",
            "name": "Custom",
            "metrics_url": "http://127.0.0.1:9000/metrics",
            "health_url": "http://127.0.0.1:9000/health",
        }
    ]
    monkeypatch.setenv("MM_PLATFORM_SERVICE_TARGETS", json.dumps(payload))
    targets = _service_targets()
    assert len(targets) == 1
    assert targets[0]["id"] == "custom"


def test_root_redirects_to_ui_when_built(platform_client: TestClient, tmp_path, monkeypatch) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>ui</html>", encoding="utf-8")
    monkeypatch.setattr("mm.api.admin._platform_ui_dist", lambda: dist)
    app = create_admin_app(bus=InMemoryBus())
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"
    ui = client.get("/ui/")
    assert ui.status_code == 200
    assert "ui" in ui.text


def test_probe_url_accepts_empty_metrics_body() -> None:
    import asyncio
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"\n")

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ok, detail = asyncio.run(
            _probe_url(f"http://127.0.0.1:{port}/metrics", expect_metrics=True)
        )
        assert ok is True
        assert detail == "http 200"
    finally:
        server.shutdown()


def test_admin_metrics_include_control_plane_gauge(platform_client: TestClient) -> None:
    response = platform_client.get("/metrics")
    assert response.status_code == 200
    assert "mm_control_plane_up" in response.text


def test_platform_ui_dist_path_exists_after_build() -> None:
    # Document expected build output location; passes once `npm run build` has been run.
    path = _platform_ui_dist()
    assert path.name == "dist"
    assert path.parent.name == "ui"
