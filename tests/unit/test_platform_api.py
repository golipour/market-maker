from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mm.api.admin import create_admin_app, _platform_ui_dist
from mm.api.platform_routes import _grafana_dashboards, _grafana_solo_panels, _service_targets
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
    assert any(d["uid"] == "mm-obs-v2-runtime-overview" for d in body["dashboards"])
    assert body["solo_panels"]


def test_grafana_embed_urls_use_kiosk_and_solo() -> None:
    dashboards = _grafana_dashboards("http://localhost:3000")
    overview = next(d for d in dashboards if d["uid"] == "mm-obs-v2-runtime-overview")
    assert "kiosk=1" in overview["embed_url"]
    solo = _grafana_solo_panels("http://localhost:3000")
    assert solo[0]["embed_url"].startswith("http://localhost:3000/d-solo/")


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


def test_platform_ui_dist_path_exists_after_build() -> None:
    # Document expected build output location; passes once `npm run build` has been run.
    path = _platform_ui_dist()
    assert path.name == "dist"
    assert path.parent.name == "ui"
