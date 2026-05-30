from mm.tools.generate_observability_v2_dashboards import (
    _runtime_fill_rate_sql,
    controller_health,
    liquidity_sourcing,
    runtime_overview,
)


def test_runtime_fill_rate_sql_keeps_private_fills_with_missing_symbol_column() -> None:
    sql = _runtime_fill_rate_sql("count")

    assert "COALESCE(e.exchange, e.payload->>'exchange', substring(e.topic from '^private:state:([^:]+):fills$')" in sql
    assert "COALESCE(e.symbol, e.payload->>'symbol') IN (${symbol:sqlstring})" in sql
    assert "COALESCE(e.symbol, e.payload->>'symbol') IS NULL" in sql
    assert "COALESCE(e.symbol, e.payload->>'symbol') = ''" in sql


def test_runtime_overview_does_not_emit_synthetic_zero_series() -> None:
    dashboard = runtime_overview()
    sql_targets = [
        target["rawSql"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", ())
        if "rawSql" in target
    ]

    for sql in sql_targets:
        assert "0.0 AS value" not in sql
        assert " AS metric, 0.0" not in sql


def test_frame_readiness_legends_include_readiness_field() -> None:
    dashboards = [controller_health(), liquidity_sourcing()]
    readiness_targets = [
        target
        for dashboard in dashboards
        for panel in dashboard["panels"]
        for target in panel.get("targets", ())
        if target.get("expr", "").startswith("mm_frame_sync_ready")
    ]

    assert readiness_targets
    for target in readiness_targets:
        assert target["legendFormat"] == "{{field}} {{exchange}} {{symbol}}"
