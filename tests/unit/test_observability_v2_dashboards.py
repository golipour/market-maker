from mm.tools.generate_observability_v2_dashboards import (
    _runtime_fill_rate_sql,
    controller_health,
    liquidity_sourcing,
    market_data_quality,
    model_quality,
    risk_execution,
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
    queries = [
        target.get("rawSql") or target.get("expr") or ""
        for panel in dashboard["panels"]
        for target in panel.get("targets", ())
    ]

    for query in queries:
        assert "0.0 AS value" not in query
        assert " AS metric, 0.0" not in query
        assert "vector(0)" not in query


def test_runtime_overview_uses_per_step_counts_not_per_second_rates() -> None:
    dashboard = runtime_overview()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}
    per_step_titles = {
        "Frames / step",
        "Decisions / step",
        "Sync requests / step",
        "Placed orders / step",
        "Failed orders / step",
        "Fill events / step",
        "Filled qty / step",
    }

    assert per_step_titles.issubset(panels_by_title)
    for title in per_step_titles:
        sql = panels_by_title[title]["targets"][0]["rawSql"]
        value_lines = [line for line in sql.splitlines() if " AS value" in line]
        assert value_lines
        for line in value_lines:
            assert "/ GREATEST(EXTRACT(EPOCH FROM '${__interval}'::interval), 1.0)" not in line


def test_rate_dashboards_use_exact_step_queries() -> None:
    dashboards = [
        runtime_overview(),
        market_data_quality(),
        controller_health(),
        model_quality(),
        risk_execution(),
    ]

    for dashboard in dashboards:
        for panel in dashboard["panels"]:
            for target in panel.get("targets", ()):
                query = target.get("rawSql") or target.get("expr") or ""
                assert "rate(" not in query
                assert "__rate_interval" not in query

    exact_titles = {
        "Frames / step",
        "Decisions / step",
        "Sync requests / step",
        "Placed orders / step",
        "Failed orders / step",
        "Market events / step",
        "Stage latency / step",
        "Warnings / step",
        "Model NLL / step",
        "Risk rejects / step",
        "Execution gate blocks / step",
        "Sync completed / failed / step",
        "Router failures / step",
        "Validation rejects / step",
    }
    panels_by_title = {
        panel["title"]: panel
        for dashboard in dashboards
        for panel in dashboard["panels"]
    }
    for title in exact_titles:
        sql = panels_by_title[title]["targets"][0]["rawSql"]
        assert 'to_timestamp((' in sql or 'to_timestamp(step.t_end_ms / 1000.0)' in sql
        assert "_time_bucket" not in sql
        assert "floor(EXTRACT(EPOCH" not in sql


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
