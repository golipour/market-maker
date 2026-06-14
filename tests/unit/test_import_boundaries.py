"""Guarantee lightweight packages do not pull in the full runtime graph."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_core_strategy_import_does_not_load_runtime() -> None:
    code = """
import sys
for k in list(sys.modules):
    if k == "mm" or k.startswith("mm."):
        del sys.modules[k]
import mm.core.models.strategy  # noqa: F401
bad = [m for m in sys.modules if m.startswith("mm.runtime")]
assert not bad, bad
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_metric_labels_run_id_requires_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MM_RUN_ID", "run-test-1")
    monkeypatch.delenv("MM_METRICS_LABEL_RUN_ID", raising=False)

    import importlib

    import mm.observability.context as ctx

    importlib.reload(ctx)
    try:
        assert "run_id" not in ctx.metric_labels()
        monkeypatch.setenv("MM_METRICS_LABEL_RUN_ID", "1")
        importlib.reload(ctx)
        assert ctx.metric_labels().get("run_id") == "run-test-1"
    finally:
        monkeypatch.delenv("MM_RUN_ID", raising=False)
        monkeypatch.delenv("MM_METRICS_LABEL_RUN_ID", raising=False)
        importlib.reload(ctx)


def test_metric_labels_worker_and_strategy_id_require_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MM_WORKER_ID", "worker-a")
    monkeypatch.setenv("MM_STRATEGY_ID", "strategy-btc")
    monkeypatch.delenv("MM_METRICS_LABEL_WORKER_ID", raising=False)
    monkeypatch.delenv("MM_METRICS_LABEL_STRATEGY_ID", raising=False)

    import importlib

    import mm.observability.context as ctx

    importlib.reload(ctx)
    try:
        assert "worker_id" not in ctx.metric_labels()
        assert "strategy_id" not in ctx.metric_labels()
        monkeypatch.setenv("MM_METRICS_LABEL_WORKER_ID", "1")
        monkeypatch.setenv("MM_METRICS_LABEL_STRATEGY_ID", "1")
        importlib.reload(ctx)
        assert ctx.metric_labels().get("worker_id") == "worker-a"
        assert ctx.metric_labels().get("strategy_id") == "strategy-btc"
        assert ctx.get_context().worker_id == "worker-a"
        assert ctx.get_context().strategy_id == "strategy-btc"
    finally:
        monkeypatch.delenv("MM_WORKER_ID", raising=False)
        monkeypatch.delenv("MM_STRATEGY_ID", raising=False)
        monkeypatch.delenv("MM_METRICS_LABEL_WORKER_ID", raising=False)
        monkeypatch.delenv("MM_METRICS_LABEL_STRATEGY_ID", raising=False)
        importlib.reload(ctx)
