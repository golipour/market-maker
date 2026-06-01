from mm.tools.generate_observability_v2_dashboards import (
    _runtime_fill_rate_sql,
    action_pipeline,
    action_pipeline_side_counts,
    controller_health,
    dashboards,
    fill_spread_pnl,
    liquidity_sourcing,
    market_data_quality,
    model_quality,
    orderbook_depth,
    portfolio_pnl,
    risk_execution,
    runtime_overview,
    target_order_stems,
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
        target_order_stems(),
        action_pipeline_side_counts(),
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
        "Bid/ask exchange accepted placement counts / step",
        "Bid/ask exchange accepted placement volume / step",
        "Bid/ask planner placement counts / step",
        "Bid/ask planner placement volume / step",
        "Bid target survival counts / step",
        "Ask target survival counts / step",
        "Bid placement counts / step",
        "Ask placement counts / step",
        "Bid live/cancel counts / step",
        "Ask live/cancel counts / step",
        "Bid removed by reason / step",
        "Ask removed by reason / step",
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


def test_orderbook_depth_includes_execution_ladder_stems() -> None:
    dashboard = orderbook_depth()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    expected = {
        "Marginal ladder": (27, "ladder"),
        "After venue rounding": (37, "rules_valid"),
        "Exchange-accepted placements": (47, "router_accepted"),
    }
    for title, (y, stage) in expected.items():
        panel = panels_by_title[title]
        assert panel["type"] == "xychart"
        assert panel["gridPos"] == {"h": 10, "w": 24, "x": 0, "y": y}

        sql = panel["targets"][0]["rawSql"]
        assert f"ARRAY['action_pipeline', '{stage}', 'orders']" in sql
        assert "ControllerDiagnostics" in sql
        assert "'${target_frame_mode}' = 'latest'" in sql
        assert "OR e.event_id = ${target_frame_event_id:sqlstring}" in sql

    vars_by_name = {var["name"]: var for var in dashboard["templating"]["list"]}
    assert vars_by_name["target_frame_mode"]["current"]["value"] == "latest"
    assert vars_by_name["target_frame_event_id"]["label"] == "Target frame"


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
    assert "'strategy cap'::text" in local_drop_sql
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


def test_action_pipeline_side_counts_dashboard_splits_bid_and_ask() -> None:
    dashboard = action_pipeline_side_counts()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    summary_expected = {
        "Bid/ask exchange accepted placement counts / step": (0, 3, "router_accepted", "accepted placement"),
        "Bid/ask exchange accepted placement volume / step": (0, 11, "router_accepted", "accepted placement"),
        "Bid/ask planner placement counts / step": (0, 19, "planned_place", "placement"),
        "Bid/ask planner placement volume / step": (0, 27, "planned_place", "placement"),
    }

    for title, (x, y, stage, label) in summary_expected.items():
        panel = panels_by_title[title]
        assert panel["type"] == "timeseries"
        assert panel["gridPos"]["x"] == x
        assert panel["gridPos"]["y"] == y
        assert panel["gridPos"]["w"] == 24
        sql = panel["targets"][0]["rawSql"]
        assert stage in sql
        assert f"bid {label}" in sql
        assert f"ask {label}" in sql
        assert "generate_series" in sql
        assert "COALESCE(d.value, 0.0) AS value" in sql

    accepted_volume_sql = panels_by_title["Bid/ask exchange accepted placement volume / step"]["targets"][0]["rawSql"]
    assert "o.value->>'qty'" in accepted_volume_sql

    volume_sql = panels_by_title["Bid/ask planner placement volume / step"]["targets"][0]["rawSql"]
    assert "o.value->>'qty'" in volume_sql

    expected = {
        "Bid target survival counts / step": (0, 35, "bid"),
        "Ask target survival counts / step": (12, 35, "ask"),
        "Bid placement counts / step": (0, 43, "bid"),
        "Ask placement counts / step": (12, 43, "ask"),
        "Bid live/cancel counts / step": (0, 51, "bid"),
        "Ask live/cancel counts / step": (12, 51, "ask"),
        "Bid removed by reason / step": (0, 59, "bid"),
        "Ask removed by reason / step": (12, 59, "ask"),
    }

    for title, (x, y, side) in expected.items():
        panel = panels_by_title[title]
        assert panel["type"] == "timeseries"
        assert panel["gridPos"]["x"] == x
        assert panel["gridPos"]["y"] == y
        assert panel["gridPos"]["w"] == 12
        sql = panel["targets"][0]["rawSql"]
        assert "ControllerDiagnostics" in sql
        assert f"= '{side}'" in sql
        assert "generate_series" in sql
        assert "COALESCE(d.value, 0.0) AS value" in sql

    survival_sql = panels_by_title["Bid target survival counts / step"]["targets"][0]["rawSql"]
    assert "('raw ladder', 'ladder')" in survival_sql
    assert "('strategy capped', 'capped')" in survival_sql
    assert "('risk approved', 'risk_approved_target')" in survival_sql
    assert "('venue valid', 'rules_valid')" in survival_sql

    placement_sql = panels_by_title["Ask placement counts / step"]["targets"][0]["rawSql"]
    assert "('planner placed', 'planned_place')" in placement_sql
    assert "('exchange accepted', 'router_accepted')" in placement_sql

    live_sql = panels_by_title["Bid live/cancel counts / step"]["targets"][0]["rawSql"]
    assert "('kept live', 'kept')" in live_sql
    assert "('planner canceled', 'planned_cancel')" in live_sql

    removed_sql = panels_by_title["Ask removed by reason / step"]["targets"][0]["rawSql"]
    assert "ARRAY['action_pipeline', stages.stage, 'rejections']" in removed_sql
    assert "risk_approved_target" in removed_sql
    assert "rules_valid" in removed_sql
    assert "router_accepted" in removed_sql


def test_portfolio_pnl_includes_per_venue_value_breakdowns() -> None:
    dashboard = portfolio_pnl()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    expected = {
        "Local cash / inventory value / wealth": (0, 19, "local"),
        "Global cash / inventory value / wealth": (12, 19, "global"),
    }
    for title, (x, y, role) in expected.items():
        panel = panels_by_title[title]
        assert panel["type"] == "timeseries"
        assert panel["gridPos"]["x"] == x
        assert panel["gridPos"]["y"] == y
        sql = panel["targets"][0]["rawSql"]
        assert f"ARRAY['state','{role}_cash']" in sql
        assert f"ARRAY['state','{role}_inventory']" in sql
        assert f"ARRAY['state','{role}_sell_vwap']" in sql
        assert f"'{role}_cash' AS metric" in sql
        assert f"'{role}_inventory_value'" in sql
        assert f"'{role}_wealth'" in sql
        assert "COALESCE(reported_wealth, cash + inventory * mark_price) AS value" in sql

    decomposition_expected = {
        "Local step PnL decomposition": (0, 27, "local"),
        "Global step PnL decomposition": (12, 27, "global"),
        "Total step PnL decomposition": (0, 35, "total"),
    }
    for title, (x, y, scope) in decomposition_expected.items():
        panel = panels_by_title[title]
        assert panel["type"] == "timeseries"
        assert panel["gridPos"]["x"] == x
        assert panel["gridPos"]["y"] == y
        sql = panel["targets"][0]["rawSql"]
        assert "LAG(cash) OVER" in sql
        assert "LAG(inventory) OVER" in sql
        assert "LAG(mark_price) OVER" in sql
        assert "'net_spread_capture' AS metric" in sql
        assert "'inventory_drift'" in sql
        assert "'adverse_selection'" in sql
        assert "(delta_cash + delta_inventory * mark_price_prev) AS value" in sql
        assert "(inventory_prev * delta_price) AS value" in sql
        assert "(delta_inventory * delta_price) AS value" in sql
        if scope == "total":
            assert "ARRAY['state','local_cash']" in sql
            assert "ARRAY['state','global_cash']" in sql
            assert "ARRAY['optimizer','config','reference_price']" in sql
        else:
            assert f"ARRAY['state','{scope}_cash']" in sql
            assert f"ARRAY['state','{scope}_inventory']" in sql
            assert f"ARRAY['state','{scope}_sell_vwap']" in sql

    assert panels_by_title["Step economics"]["gridPos"]["y"] == 43


def test_target_order_stems_dashboard_aligns_price_qty_stage_charts() -> None:
    dashboard = target_order_stems()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    stem_titles = {
        "01 Strategy raw ladder",
        "02 Strategy cap -> target orderbook",
        "03 Risk-approved target",
        "04 Planner new place orders",
        "05 Venue rules valid after rounding",
        "06 Exchange accepted placed orders",
        "07 Planner kept live orders",
        "08 Planner cancel orders",
        "09 Removed by local risk / collateral",
        "10 Removed by venue adapter: invalid price/qty",
        "11 Removed by venue adapter: min qty",
        "12 Removed by venue adapter: min notional",
    }
    assert stem_titles.issubset(panels_by_title)

    expected_y = {
        "01 Strategy raw ladder": 3,
        "02 Strategy cap -> target orderbook": 10,
        "03 Risk-approved target": 17,
        "04 Planner new place orders": 24,
        "05 Venue rules valid after rounding": 31,
        "06 Exchange accepted placed orders": 38,
        "07 Planner kept live orders": 45,
        "08 Planner cancel orders": 52,
        "09 Removed by local risk / collateral": 59,
        "10 Removed by venue adapter: invalid price/qty": 66,
        "11 Removed by venue adapter: min qty": 73,
        "12 Removed by venue adapter: min notional": 80,
    }
    for title in stem_titles:
        assert panels_by_title[title]["gridPos"]["x"] == 0
        assert panels_by_title[title]["gridPos"]["w"] == 24
        assert panels_by_title[title]["gridPos"]["h"] == 7
        assert panels_by_title[title]["gridPos"]["y"] == expected_y[title]

    for title in stem_titles:
        panel = panels_by_title[title]
        assert panel["type"] == "xychart"
        assert panel["fieldConfig"]["defaults"]["custom"]["show"] == "points"
        assert panel["fieldConfig"]["defaults"]["custom"]["lineWidth"] == 0
        assert panel["transformations"][0]["options"]["fields"] == ["series"]
        price_override = next(
            override
            for override in panel["fieldConfig"]["overrides"]
            if override["matcher"]["options"] == "price"
        )
        assert {"id": "decimals", "value": 6} in price_override["properties"]
        axis_override = next(
            override
            for override in panel["fieldConfig"]["overrides"]
            if override["matcher"]["options"] == "^axis range$"
        )
        assert {
            "id": "custom.hideFrom",
            "value": {"legend": True, "tooltip": True, "viz": False},
        } in axis_override["properties"]
        assert "black" in str(axis_override["properties"])
        overrides = str(panel["fieldConfig"]["overrides"])
        assert ".*bid$" in overrides
        assert ".*ask$" in overrides
        assert "green" in overrides
        assert "red" in overrides

    raw_sql = panels_by_title["01 Strategy raw ladder"]["targets"][0]["rawSql"]
    target_sql = panels_by_title["02 Strategy cap -> target orderbook"]["targets"][0]["rawSql"]
    risk_sql = panels_by_title["03 Risk-approved target"]["targets"][0]["rawSql"]
    place_sql = panels_by_title["04 Planner new place orders"]["targets"][0]["rawSql"]
    rules_sql = panels_by_title["05 Venue rules valid after rounding"]["targets"][0]["rawSql"]
    accepted_sql = panels_by_title["06 Exchange accepted placed orders"]["targets"][0]["rawSql"]
    kept_sql = panels_by_title["07 Planner kept live orders"]["targets"][0]["rawSql"]
    cancel_sql = panels_by_title["08 Planner cancel orders"]["targets"][0]["rawSql"]
    risk_removed_sql = panels_by_title["09 Removed by local risk / collateral"]["targets"][0]["rawSql"]
    invalid_sql = panels_by_title["10 Removed by venue adapter: invalid price/qty"]["targets"][0]["rawSql"]
    min_qty_sql = panels_by_title["11 Removed by venue adapter: min qty"]["targets"][0]["rawSql"]
    min_notional_sql = panels_by_title["12 Removed by venue adapter: min notional"]["targets"][0]["rawSql"]

    mode_var = next(var for var in dashboard["templating"]["list"] if var["name"] == "target_frame_mode")
    assert mode_var["label"] == "Target frame mode"
    assert mode_var["current"]["value"] == "latest"
    assert [option["value"] for option in mode_var["options"]] == ["latest", "pinned"]
    frame_var = next(var for var in dashboard["templating"]["list"] if var["name"] == "target_frame_event_id")
    assert frame_var["label"] == "Target frame"
    assert "e.event_id AS __value" in frame_var["query"]
    assert "ControllerDiagnostics" in frame_var["query"]

    assert "ARRAY['action_pipeline', 'ladder', 'orders']" in raw_sql
    assert "ARRAY['action_pipeline', 'capped', 'orders']" in target_sql
    assert "ARRAY['action_pipeline', 'risk_approved_target', 'orders']" in risk_sql
    assert "ARRAY['action_pipeline', 'risk_approved_target', 'rejections']" in risk_removed_sql
    assert "COALESCE(NULLIF(o.value->>'reason', ''), 'unknown')" in risk_removed_sql
    assert "ARRAY['action_pipeline', 'kept', 'orders']" in kept_sql
    assert "ARRAY['action_pipeline', 'planned_cancel', 'orders']" in cancel_sql
    assert "ARRAY['action_pipeline', 'planned_place', 'orders']" in place_sql
    assert "ARRAY['action_pipeline', 'rules_valid', 'orders']" in rules_sql
    assert "ARRAY['action_pipeline', 'router_accepted', 'orders']" in accepted_sql
    assert "ARRAY['action_pipeline', 'rules_valid', 'rejections']" in invalid_sql
    assert "rules_invalid_price_or_qty" in invalid_sql
    assert "ARRAY['action_pipeline', 'rules_valid', 'rejections']" in min_qty_sql
    assert "rules_min_qty" in min_qty_sql
    assert "ARRAY['action_pipeline', 'rules_valid', 'rejections']" in min_notional_sql
    assert "rules_min_notional" in min_notional_sql

    for sql in (
        raw_sql,
        target_sql,
        risk_sql,
        risk_removed_sql,
        place_sql,
        rules_sql,
        accepted_sql,
        kept_sql,
        cancel_sql,
        invalid_sql,
        min_qty_sql,
        min_notional_sql,
    ):
        assert "ControllerDiagnostics" in sql
        assert "'${target_frame_mode}' = 'latest'" in sql
        assert "OR e.event_id = ${target_frame_event_id:sqlstring}" in sql
        assert "axis_frame AS" in sql
        assert "'axis range' AS series" in sql
        assert "FROM axis_rows" in sql
        assert "price::text" not in sql


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
        target_order_stems(),
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


def test_fill_spread_pnl_dashboard_shows_interval_fill_summary() -> None:
    dashboard = fill_spread_pnl()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    expected_positions = {
        "Filled bid volume": (0, 5),
        "Filled ask volume": (5, 5),
        "Average bid fill price": (10, 5),
        "Average ask fill price": (15, 5),
        "Average spread PnL / unit": (20, 4),
        "Total fill fee": (0, 5),
    }

    for title, (x, w) in expected_positions.items():
        panel = panels_by_title[title]
        assert panel["type"] == "stat"
        expected_y = 8 if title == "Total fill fee" else 3
        assert panel["gridPos"] == {"h": 5, "w": w, "x": x, "y": expected_y}
        sql = panel["targets"][0]["rawSql"]
        assert "private:state:%:fills" in sql
        assert "$__timeFilter(e.ts)" in sql
        assert "COALESCE(e.exchange, e.payload->>'exchange'" in sql
        assert "COALESCE(e.symbol, e.payload->>'symbol', '')" in sql
        assert "SUM(price * qty) FILTER" in sql
        assert "e.payload->>'fee'" in sql

    bid_volume_sql = panels_by_title["Filled bid volume"]["targets"][0]["rawSql"]
    assert "COALESCE(bid_qty, 0.0) AS value" in bid_volume_sql

    ask_volume_sql = panels_by_title["Filled ask volume"]["targets"][0]["rawSql"]
    assert "COALESCE(ask_qty, 0.0) AS value" in ask_volume_sql

    bid_avg_sql = panels_by_title["Average bid fill price"]["targets"][0]["rawSql"]
    assert "bid_avg_price AS value" in bid_avg_sql

    ask_avg_sql = panels_by_title["Average ask fill price"]["targets"][0]["rawSql"]
    assert "ask_avg_price AS value" in ask_avg_sql

    spread_sql = panels_by_title["Average spread PnL / unit"]["targets"][0]["rawSql"]
    assert "ask_avg_price - bid_avg_price AS value" in spread_sql

    fee_sql = panels_by_title["Total fill fee"]["targets"][0]["rawSql"]
    assert "SUM(fee) AS total_fee" in fee_sql
    assert "COALESCE(total_fee, 0.0) AS value" in fee_sql


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
