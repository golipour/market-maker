from __future__ import annotations

import json
from pathlib import Path


def test_model_params_dashboard_has_conditional_price_panels() -> None:
    dashboard = json.loads(
        Path("grafana/dashboards/mm_strategy_model_params_pg.json").read_text(encoding="utf-8")
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    expected = {
        "Conditional price model params",
        "Local conditional price — gate",
        "Local conditional price — components",
        "Global conditional price — gate",
        "Global conditional price — components",
    }
    assert expected.issubset(panels)

    local_gate_sql = panels["Local conditional price — gate"]["targets"][0]["rawSql"]
    global_gate_sql = panels["Global conditional price — gate"]["targets"][0]["rawSql"]
    local_component_sql = panels["Local conditional price — components"]["targets"][0]["rawSql"]

    assert "market_prob" in local_gate_sql
    assert "market_resp" in local_gate_sql
    assert "gate_beta[" in local_gate_sql
    assert "local_price_model" in local_gate_sql
    assert "global_price_model" in global_gate_sql
    assert "gate_bins" not in local_gate_sql
    assert "gate_bin_probs" not in local_gate_sql

    assert "market_beta" in local_component_sql
    assert "market_sigma" in local_component_sql
    assert "refresh_sigma" in local_component_sql
    assert "refresh_mean" in local_component_sql
