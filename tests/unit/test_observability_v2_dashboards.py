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
        action_pipeline(),
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
        "Order flow / step",
        "Sync plan counts / step",
        "Orders sent to exchange / step",
        "Orders accepted by exchange / step",
        "Canceled order ids / step",
        "Rejected order commands / step",
        "Fill events / step",
        "Venue gate status / step",
        "Risk reject reasons / step",
        "Gate block reasons / step",
        "Router failures / step",
        "Target orders / step",
        "Target quantity / step",
        "Frame order flow / step",
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


def test_action_pipeline_exposes_order_flow_charts() -> None:
    dashboard = action_pipeline()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    expected_titles = {
        "Target orders / step",
        "Target quantity / step",
        "Target ladder prices / step",
        "Target ladder qty / step",
        "Frame order flow / step",
        "Local orders removed before target / step",
        "Global risk rejects / step",
        "Local gate blocks / frame",
        "Global gate blocks / frame",
    }
    assert expected_titles.issubset(panels_by_title)
    assert "Latest target orders" not in panels_by_title
    assert "Latest frame order counts" not in panels_by_title
    for title in expected_titles:
        assert panels_by_title[title]["type"] == "timeseries"

    assert panels_by_title["Target orders / step"]["gridPos"]["x"] == 0
    assert panels_by_title["Target quantity / step"]["gridPos"]["x"] == 12
    assert panels_by_title["Target ladder prices / step"]["gridPos"]["x"] == 0
    assert panels_by_title["Target ladder qty / step"]["gridPos"]["x"] == 12
    assert panels_by_title["Frame order flow / step"]["gridPos"]["w"] == 24
    assert panels_by_title["Local orders removed before target / step"]["gridPos"]["x"] == 0
    assert panels_by_title["Global risk rejects / step"]["gridPos"]["x"] == 12
    assert panels_by_title["Local gate blocks / frame"]["gridPos"]["x"] == 0
    assert panels_by_title["Global gate blocks / frame"]["gridPos"]["x"] == 12

    target_count_sql = panels_by_title["Target orders / step"]["targets"][0]["rawSql"]
    target_qty_sql = panels_by_title["Target quantity / step"]["targets"][0]["rawSql"]
    ladder_price_sql = panels_by_title["Target ladder prices / step"]["targets"][0]["rawSql"]
    ladder_qty_sql = panels_by_title["Target ladder qty / step"]["targets"][0]["rawSql"]
    frame_flow_sql = panels_by_title["Frame order flow / step"]["targets"][0]["rawSql"]
    local_drop_sql = panels_by_title["Local orders removed before target / step"]["targets"][0]["rawSql"]
    global_reject_sql = panels_by_title["Global risk rejects / step"]["targets"][0]["rawSql"]
    local_gate_sql = panels_by_title["Local gate blocks / frame"]["targets"][0]["rawSql"]
    global_gate_sql = panels_by_title["Global gate blocks / frame"]["targets"][0]["rawSql"]

    assert "VALUES ('local', e.payload->'local'), ('global', e.payload->'global')" in target_count_sql
    assert "VALUES ('local', e.payload->'local'), ('global', e.payload->'global')" in target_qty_sql
    for sql in (ladder_price_sql, ladder_qty_sql):
        assert "WITH expanded AS" in sql
        assert "WITH ORDINALITY" in sql
        assert "SELECT DISTINCT ON" in sql
        assert "book.role" in sql
    assert "new order candidates" in frame_flow_sql
    assert "exchange accepted" in frame_flow_sql
    for sql in (target_count_sql, target_qty_sql):
        assert "${interval_ms}" in sql
        assert "floor(EXTRACT(EPOCH" in sql
        assert "generate_series" in sql
        assert "COALESCE(d.value, 0.0) AS value" in sql
    assert "GREATEST(drop_stage.before_count - drop_stage.after_count, 0.0)" in local_drop_sql
    assert "'venue limits'::text" in local_drop_sql
    assert "'order shaping'::text" in local_drop_sql
    assert "jsonb_each_text" not in local_drop_sql
    assert "reason.key LIKE 'global_%'" in global_reject_sql
    assert "blocked.reason AS metric" not in local_drop_sql
    assert "blocked.reason AS metric" not in global_reject_sql
    assert "blocked.reason AS metric" in local_gate_sql
    assert "blocked.reason AS metric" in global_gate_sql
    assert "COALESCE(gate.value->>'role', '') = 'local'" in local_gate_sql
    assert "COALESCE(gate.value->>'role', '') = 'global'" in global_gate_sql
    for sql in (local_drop_sql, global_reject_sql, local_gate_sql, global_gate_sql):
        assert "${interval_ms}" in sql
        assert "floor(EXTRACT(EPOCH" in sql


def test_risk_execution_exposes_order_flow_charts() -> None:
    dashboard = risk_execution()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    expected_titles = {
        "Order flow / step",
        "Sync plan counts / step",
        "Orders sent to exchange / step",
        "Orders accepted by exchange / step",
        "Canceled order ids / step",
        "Rejected order commands / step",
        "Fill events / step",
        "Venue gate status / step",
        "Risk reject reasons / step",
        "Gate block reasons / step",
        "Router failures / step",
    }
    assert expected_titles.issubset(panels_by_title)
    for title in expected_titles:
        assert panels_by_title[title]["type"] == "timeseries"

    order_flow_sql = panels_by_title["Order flow / step"]["targets"][0]["rawSql"]
    sync_sql = panels_by_title["Sync plan counts / step"]["targets"][0]["rawSql"]
    requested_sql = panels_by_title["Orders sent to exchange / step"]["targets"][0]["rawSql"]
    accepted_sql = panels_by_title["Orders accepted by exchange / step"]["targets"][0]["rawSql"]
    canceled_sql = panels_by_title["Canceled order ids / step"]["targets"][0]["rawSql"]
    rejected_sql = panels_by_title["Rejected order commands / step"]["targets"][0]["rawSql"]
    fills_sql = panels_by_title["Fill events / step"]["targets"][0]["rawSql"]
    venue_gate_sql = panels_by_title["Venue gate status / step"]["targets"][0]["rawSql"]

    assert "target orders" in order_flow_sql
    assert "placed orders" in order_flow_sql
    assert "canceled order ids" in order_flow_sql
    assert "rejected order commands" in order_flow_sql
    assert "fill events" in order_flow_sql
    assert "ExecutionSyncRequested" in sync_sql
    assert "ExecutionSyncCompleted" in sync_sql
    assert "target_count" in sync_sql
    assert "accepted_count" in sync_sql
    assert "PlaceOrdersRequested" in requested_sql
    assert "PlaceOrdersAccepted" in accepted_sql
    assert "CancelOrdersAccepted" in canceled_sql
    assert "OrderValidationRejected" in rejected_sql
    assert "PlaceOrdersFailed" in rejected_sql
    assert "private:state:%:fills" in fills_sql
    assert "has orders" in venue_gate_sql
    assert "can sync" in venue_gate_sql
    assert "live" in venue_gate_sql


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
        "Order flow / step",
        "Sync plan counts / step",
        "Orders sent to exchange / step",
        "Orders accepted by exchange / step",
        "Canceled order ids / step",
        "Rejected order commands / step",
        "Fill events / step",
        "Venue gate status / step",
        "Risk reject reasons / step",
        "Gate block reasons / step",
        "Router failures / step",
        "Target orders / step",
        "Target quantity / step",
        "Target ladder prices / step",
        "Target ladder qty / step",
        "Frame order flow / step",
        "Local orders removed before target / step",
        "Global risk rejects / step",
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
