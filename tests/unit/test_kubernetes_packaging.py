from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
K8S_DIR = ROOT / "deploy" / "kubernetes"
K8S_CONFIG_DIR = ROOT / "mm" / "configs" / "k8s"


def _load_manifest_documents(path: Path) -> list[dict]:
    documents: list[dict] = []
    for chunk in path.read_text().split("---"):
        chunk = chunk.strip()
        if not chunk:
            continue
        kind = _yaml_scalar(chunk, "kind")
        metadata_name = _yaml_scalar(chunk, "name")
        if kind:
            documents.append({"kind": kind, "metadata": {"name": metadata_name}, "raw": chunk})
    return documents


def _yaml_scalar(text: str, field: str) -> str | None:
    prefix = f"{field}:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip().strip('"').strip("'")
    return None


@pytest.mark.parametrize(
    "filename",
    [
        "namespace.yaml",
        "configmap.yaml",
        "redis.yaml",
        "ingestor.yaml",
        "persistence.yaml",
        "worker.yaml",
        "admin.yaml",
        "kustomization.yaml",
        "secret.example.yaml",
    ],
)
def test_kubernetes_manifest_files_exist(filename: str) -> None:
    assert (K8S_DIR / filename).is_file()


def test_kubernetes_manifests_define_core_workloads() -> None:
    kinds = set()
    names = set()
    for path in K8S_DIR.glob("*.yaml"):
        if path.name in {"kustomization.yaml", "secret.example.yaml"}:
            continue
        for doc in _load_manifest_documents(path):
            kinds.add(doc["kind"])
            if doc["metadata"]["name"]:
                names.add(doc["metadata"]["name"])
    assert "Namespace" in kinds
    assert "Deployment" in kinds
    assert "Service" in kinds
    assert "ConfigMap" in kinds
    assert "market-maker-ingestor" in names
    assert "market-maker-worker" in names
    assert "market-maker-persistence" in names
    assert "market-maker-admin" in names
    assert "redis" in names


def test_k8s_service_configs_are_valid_json() -> None:
    for name in ("ingestor.json", "persistence.json", "worker.json"):
        payload = json.loads((K8S_CONFIG_DIR / name).read_text())
        assert isinstance(payload, dict)
        assert payload["bus"]["redis_url"] == "redis://redis:6379/0"


def test_configmap_embeds_service_config_keys() -> None:
    configmap = (K8S_DIR / "configmap.yaml").read_text()
    for key in ("ingestor.json", "persistence.json", "worker.json"):
        assert f"{key}: |" in configmap
        assert "redis://redis:6379/0" in configmap
    assert '"bitwana"' in configmap
    secret_example = (K8S_DIR / "secret.example.yaml").read_text()
    assert "BITWANA_API_TOKEN" in secret_example
    assert "BITWANA_REFRESH_TOKEN" in secret_example


def test_dockerfile_declares_python_runtime() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "python:3.11-slim" in dockerfile
    assert "requirements.txt" in dockerfile
    assert "mm/" in dockerfile
    assert "platform/ui/dist" in dockerfile


def test_admin_entrypoint_module_imports() -> None:
    from mm.api import __main__ as admin_main

    assert callable(admin_main.main)
