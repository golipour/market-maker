from mm.tools.generate_observability_v2_dashboards import (
    _runtime_fill_rate_sql,
    action_pipeline,
    controller_health,
    dashboards,
    liquidity_sourcing,
    market_data_quality,
    model_quality,
    risk_execution,
    runtime_overview,
)


def test_runtime_fill_rate_sql_keeps_private_fills_with_missing_symbol_column() -> None:
    sql = _runtime_fill_rate_sql("count")

    assert "COALESCE(e.exchange, e.payload->>'exchange', substring(e.topic from '^private:state:([^:]+):fills$')" in sql
    assert "e.topic LIKE 'private:state:%:fills'" in sql
    assert "COALESCE(e.symbol, e.payload->>'symbol')" in sql
    assert "COALESCE(e.symbol, e.payload->>'symbol') IN (${symbol:sqlstring})" not in sql


def test_event_rate_panels_emit_zero_filled_series() -> None:
    dashboards = [
        runtime_overview(),
        market_data_quality(),
        controller_health(),
        risk_execution(),
    ]
    zero_filled_titles = {
        "Frames / step",
        "Decisions / step",
        "Sync requests / step",
        "Placed orders / step",
        "Cancelled orders / step",
        "Failed orders / step",
        "Fill events / step (private feed)",
        "Filled qty / step (private feed)",
        "Execution live orders / step (latest snapshot)",
        "Execution live qty / step (latest snapshot)",
        "Market events / step",
        "Warnings / step",
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

    assert zero_filled_titles.issubset(panels_by_title)
    for title in zero_filled_titles:
        sql = panels_by_title[title]["targets"][0]["rawSql"]
        assert "generate_series" in sql
        assert "COALESCE(d.value, 0.0) AS value" in sql
        assert "CROSS JOIN metrics" in sql


def test_zero_fill_metric_grids_use_market_maker_scope_not_wallet_symbols() -> None:
    dashboard = runtime_overview()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}
    scoped_titles = {
        "Frames / step",
        "Decisions / step",
        "Placed orders / step",
        "Fill events / step (private feed)",
        "Execution live orders / step (latest snapshot)",
        "Execution live qty / step (latest snapshot)",
    }

    for title in scoped_titles:
        sql = panels_by_title[title]["targets"][0]["rawSql"]
        assert "event_type IN ('StrategyDecision', 'ControllerDiagnostics', 'JointTargetBook')" in sql
        assert "CROSS JOIN unnest(ARRAY[${symbol:sqlstring}]" not in sql

    templating = {item["name"]: item["query"] for item in dashboard["templating"]["list"]}
    assert "event_type IN ('StrategyDecision', 'ControllerDiagnostics', 'JointTargetBook')" in templating["symbol"]
    assert "private:state" not in templating["symbol"]


def test_postgres_queries_use_empty_safe_multiselect_filters() -> None:
    combined_sql = "\n".join(
        target["rawSql"]
        for dashboard in dashboards()
        for panel in dashboard["panels"]
        for target in panel.get("targets", ())
        if target.get("datasource", {}).get("type") == "postgres"
    )

    assert "IN (${exchange:sqlstring})" not in combined_sql
    assert "IN (${symbol:sqlstring})" not in combined_sql
    assert "IN (${actor_id:sqlstring})" not in combined_sql
    assert "= ANY(ARRAY[${exchange:sqlstring}]::text[])" in combined_sql
    assert "= ANY(ARRAY[${symbol:sqlstring}]::text[])" in combined_sql
    assert "= ANY(ARRAY[${actor_id:sqlstring}]::text[])" in combined_sql


def test_action_pipeline_splits_local_and_global_target_book_charts() -> None:
    dashboard = action_pipeline()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    assert panels_by_title["Local desired target orders"]["gridPos"]["x"] == 0
    assert panels_by_title["Global desired target orders"]["gridPos"]["x"] == 12
    assert panels_by_title["Local desired target qty"]["gridPos"]["x"] == 0
    assert panels_by_title["Global desired target qty"]["gridPos"]["x"] == 12
    assert panels_by_title["Local execution contract funnel"]["gridPos"]["x"] == 0
    assert panels_by_title["Global execution contract funnel"]["gridPos"]["x"] == 12
    assert panels_by_title["Local candidate/order rejections"]["gridPos"]["x"] == 0
    assert panels_by_title["Global candidate/order rejections"]["gridPos"]["x"] == 12
    assert panels_by_title["Local gate blocks / frame"]["gridPos"]["x"] == 0
    assert panels_by_title["Global gate blocks / frame"]["gridPos"]["x"] == 12

    local_sql = panels_by_title["Local desired target orders"]["targets"][0]["rawSql"]
    global_sql = panels_by_title["Global desired target orders"]["targets"][0]["rawSql"]
    local_qty_sql = panels_by_title["Local desired target qty"]["targets"][0]["rawSql"]
    global_qty_sql = panels_by_title["Global desired target qty"]["targets"][0]["rawSql"]
    local_funnel_sql = panels_by_title["Local execution contract funnel"]["targets"][0]["rawSql"]
    global_funnel_sql = panels_by_title["Global execution contract funnel"]["targets"][0]["rawSql"]
    local_reject_sql = panels_by_title["Local candidate/order rejections"]["targets"][0]["rawSql"]
    global_reject_sql = panels_by_title["Global candidate/order rejections"]["targets"][0]["rawSql"]
    local_gate_sql = panels_by_title["Local gate blocks / frame"]["targets"][0]["rawSql"]
    global_gate_sql = panels_by_title["Global gate blocks / frame"]["targets"][0]["rawSql"]

    assert "VALUES ('local', payload->'local')" in local_sql
    assert "VALUES ('global', payload->'global')" in global_sql
    assert "VALUES ('local', e.payload->'local')" in local_qty_sql
    assert "VALUES ('global', e.payload->'global')" in global_qty_sql
    assert "payload->'global'" not in local_sql
    assert "payload->'local'" not in global_sql
    for sql in (local_sql, global_sql, local_qty_sql, global_qty_sql):
        assert "${interval_ms}" in sql
        assert "floor(EXTRACT(EPOCH" in sql
    assert "e.payload->'local'->'orders'" in local_funnel_sql
    assert "COALESCE(gate.value->>'role', '') = 'local'" in local_funnel_sql
    assert "e.payload->'global'->'orders'" in global_funnel_sql
    assert "COALESCE(gate.value->>'role', '') = 'global'" in global_funnel_sql
    assert "NOT (reason.key LIKE 'global_%'" in local_reject_sql
    assert "reason.key LIKE 'global_%'" in global_reject_sql
    assert "blocked.reason AS metric" not in local_reject_sql
    assert "blocked.reason AS metric" not in global_reject_sql
    assert "blocked.reason AS metric" in local_gate_sql
    assert "blocked.reason AS metric" in global_gate_sql
    assert "COALESCE(gate.value->>'role', '') = 'local'" in local_gate_sql
    assert "COALESCE(gate.value->>'role', '') = 'global'" in global_gate_sql
    for sql in (local_funnel_sql, global_funnel_sql, local_reject_sql, global_reject_sql, local_gate_sql, global_gate_sql):
        assert "${interval_ms}" in sql
        assert "floor(EXTRACT(EPOCH" in sql


def test_runtime_overview_uses_per_step_counts_not_per_second_rates() -> None:
    dashboard = runtime_overview()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}
    per_step_titles = {
        "Frames / step",
        "Decisions / step",
        "Sync requests / step",
        "Placed orders / step",
        "Cancelled orders / step",
        "Failed orders / step",
        "Fill events / step (private feed)",
        "Filled qty / step (private feed)",
    }

    assert per_step_titles.issubset(panels_by_title)
    for title in per_step_titles:
        sql = panels_by_title[title]["targets"][0]["rawSql"]
        value_lines = [line for line in sql.splitlines() if " AS value" in line]
        assert value_lines
        for line in value_lines:
            assert "/ GREATEST(EXTRACT(EPOCH FROM '${__interval}'::interval), 1.0)" not in line


def test_rate_dashboards_use_approximate_step_queries() -> None:
    dashboards = [
        runtime_overview(),
        market_data_quality(),
        controller_health(),
        model_quality(),
        risk_execution(),
        action_pipeline(),
    ]

    for dashboard in dashboards:
        for panel in dashboard["panels"]:
            for target in panel.get("targets", ()):
                query = target.get("rawSql") or target.get("expr") or ""
                assert "rate(" not in query
                assert "__rate_interval" not in query

    approximate_titles = {
        "Frames / step",
        "Decisions / step",
        "Sync requests / step",
        "Placed orders / step",
        "Cancelled orders / step",
        "Failed orders / step",
        "Fill events / step (private feed)",
        "Execution live orders / step (latest snapshot)",
        "Execution live qty / step (latest snapshot)",
        "Market events / step",
        "Stage latency / step",
        "Warnings / step",
        "Model NLL / step",
        "Risk rejects / step",
        "Execution gate blocks / step",
        "Sync completed / failed / step",
        "Router failures / step",
        "Validation rejects / step",
        "Local desired target orders",
        "Global desired target orders",
        "Local desired target qty",
        "Global desired target qty",
        "Local execution contract funnel",
        "Global execution contract funnel",
        "Local candidate/order rejections",
        "Global candidate/order rejections",
        "Local gate blocks / frame",
        "Global gate blocks / frame",
    }
    panels_by_title = {
        panel["title"]: panel
        for dashboard in dashboards
        for panel in dashboard["panels"]
    }
    for title in approximate_titles:
        sql = panels_by_title[title]["targets"][0]["rawSql"]
        assert "${interval_ms}" in sql
        assert "floor(EXTRACT(EPOCH" in sql
        assert "${__interval}" not in sql


def test_runtime_live_order_panels_use_latest_snapshot_semantics() -> None:
    dashboard = runtime_overview()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    count_sql = panels_by_title["Execution live orders / step (latest snapshot)"]["targets"][0]["rawSql"]
    qty_sql = panels_by_title["Execution live qty / step (latest snapshot)"]["targets"][0]["rawSql"]
    latest_sql = panels_by_title["Latest private vs execution order counts"]["targets"][0]["rawSql"]
    fills_sql = panels_by_title["Recent private fills"]["targets"][0]["rawSql"]

    for sql in (count_sql, qty_sql):
        assert "latest_per_bucket" in sql
        assert "SELECT DISTINCT ON (bucket, exchange)" in sql
        assert "generate_series" in sql
        assert "COALESCE(d.value, 0.0) AS value" in sql
        assert "CROSS JOIN metrics" in sql
        assert "AVG(value)" not in sql
        assert "SUM(1.0)" not in sql

    assert "latest_execution" in latest_sql
    assert "latest_private" in latest_sql
    assert "COUNT(open_order.key)" in latest_sql
    assert "private:state:%:fills" in fills_sql
    assert "ORDER BY e.ts DESC" in fills_sql


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
