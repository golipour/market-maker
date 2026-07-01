from mm.tools.generate_observability_v2_dashboards import (
    _optimizer_weighted_components_sql,
    _runtime_fill_rate_sql,
    action_pipeline,
    action_pipeline_side_counts,
    controller_health,
    dashboards,
    exchange_account,
    execution_errors,
    fill_spread_pnl,
    liquidity_sourcing,
    market_data_quality,
    model_parameters,
    model_quality,
    optimizer_diagnostics,
    orderbook_depth,
    pipeline_time_profile,
    portfolio_pnl,
    risk_execution,
    runtime_overview,
    target_order_stems,
)


def test_optimizer_weighted_components_sql_includes_final_objective_total() -> None:
    sql = _optimizer_weighted_components_sql()
    dashboard = optimizer_diagnostics()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    assert "UNION ALL" in sql
    assert "'objective_total' AS metric" in sql
    assert "ARRAY['optimizer','objective','final','value']" in sql
    assert "ARRAY['optimizer','objective','final','components']" in sql
    assert panels_by_title["Objective components (weighted J terms)"]["targets"][0]["rawSql"] == sql


def test_observability_v2_dashboard_defaults_are_lightweight() -> None:
    dashboard = market_data_quality()
    templating = {item["name"]: item for item in dashboard["templating"]["list"]}

    assert dashboard["time"] == {"from": "now-15m", "to": "now"}
    assert templating["exchange"]["current"] == {"selected": False, "text": [], "value": []}
    assert templating["symbol"]["current"] == {"selected": False, "text": [], "value": []}
    assert templating["actor_id"]["current"] == {
        "selected": False,
        "text": "All",
        "value": "$__all",
    }


def test_runtime_fill_rate_sql_keeps_private_fills_with_missing_symbol_column() -> None:
    sql = _runtime_fill_rate_sql("count")

    assert "COALESCE(e.exchange, e.payload->>'exchange', substring(e.topic from '^private:state:([^:]+):fills$')" in sql
    assert "e.topic LIKE 'private:state:%:fills'" in sql
    assert "COALESCE(e.symbol, e.payload->>'symbol')" in sql
    assert "COALESCE(e.symbol, e.payload->>'symbol') IN (${symbol:sqlstring})" not in sql


def test_market_data_quality_uses_per_exchange_market_events() -> None:
    dashboard = market_data_quality()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    assert "Local best bid / ask / VWAP" not in panels_by_title
    assert "Global best bid / ask / VWAP" not in panels_by_title

    price_sql = panels_by_title["Best bid / ask / VWAP by exchange"]["targets"][0]["rawSql"]
    assert "role_exchange" not in price_sql
    assert "mm_market_facts" in price_sql
    assert "event_type = 'MarketEvent'" in price_sql
    assert "exchange || ' ' || symbol || ' best_bid'" in price_sql
    assert "COALESCE(symbol, payload->>'symbol') || ' buy_vwap'" in price_sql
    assert "COALESCE(symbol, payload->>'symbol') || ' sell_vwap'" in price_sql
    assert "COALESCE(symbol, payload->>'symbol') || ' trade_vwap'" in price_sql
    assert "NULLIF(payload->>'vwap', '0')" in price_sql
    assert "interval_ms::text = NULLIF('${interval_ms}', '')" in price_sql
    assert "price_series:sqlstring" in price_sql

    volume_sql = panels_by_title["Buy / sell volume by exchange"]["targets"][0]["rawSql"]
    assert "role_exchange" not in volume_sql
    assert "mm_market_facts" in volume_sql
    assert "exchange || ' ' || symbol || ' buy_volume'" in volume_sql
    assert "interval_ms::text = NULLIF('${interval_ms}', '')" in volume_sql
    assert "volume_series:sqlstring" in volume_sql

    templating = {item["name"]: item for item in dashboard["templating"]["list"]}
    assert templating["price_series"]["multi"] is True
    assert "trade_vwap" in templating["price_series"]["query"]
    assert templating["price_series"]["current"]["value"] == [
        "best_bid",
        "best_ask",
        "buy_vwap",
        "sell_vwap",
    ]

    assert "Market events / step" not in panels_by_title
    assert "Raw feed metrics guide" in panels_by_title
    freshness = panels_by_title["Raw feed freshness ms"]
    assert freshness.get("description")


def test_event_rate_panels_emit_zero_filled_series() -> None:
    dashboards = [
        runtime_overview(),
        market_data_quality(),
        controller_health(),
        risk_execution(),
        execution_errors(),
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
        "Execution error flow / step",
        "Errors by reason / step",
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
    assert "execution_gates','venue_gates" in templating["exchange"]
    assert "global_market','exchange" in templating["exchange"]
    assert "event_type IN ('StrategyDecision', 'ControllerDiagnostics', 'JointTargetBook')" in templating["symbol"]
    assert "global_market','symbol" in templating["symbol"]
    assert "event_type = 'MarketEvent'" in templating["symbol"]
    assert "ORDER BY latest_ts DESC" in templating["symbol"]
    assert "private:state" not in templating["symbol"]


def test_orderbook_depth_includes_execution_ladder_stems() -> None:
    dashboard = orderbook_depth()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    depth_expected = {
        "Orderbook depth": (0, 4, "MarketEvent", "m.exchange = ctx.exchange"),
        "Clean orderbook depth": (0, 12, "cleared_orderbook", "ctx.diag_payload AS payload"),
        "Action ladder depth": (0, 20, "ARRAY['action_pipeline', 'ladder', 'orders']", "exchange:sqlstring"),
    }
    for title, (x, y, expected_sql, role_sql) in depth_expected.items():
        panel = panels_by_title[title]
        assert panel["type"] == "xychart"
        assert panel["gridPos"]["x"] == x
        assert panel["gridPos"]["y"] == y
        assert panel["gridPos"]["w"] == 24
        sql = panel["targets"][0]["rawSql"]
        assert expected_sql in sql
        assert role_sql in sql
        if "action ladder depth" in title.lower():
            assert "jsonb_typeof" in sql
            assert "orders_sample" in sql

    assert "Ladder grid vs orders" not in panels_by_title
    assert "Local orderbook depth" not in panels_by_title
    assert "Global marginal ladder" not in panels_by_title

    expected = {
        "Marginal ladder": (0, 28, "ladder"),
        "After venue rounding": (0, 38, "rules_valid"),
        "Exchange-accepted placements": (12, 38, "router_accepted"),
    }
    for title, (x, y, stage) in expected.items():
        panel = panels_by_title[title]
        assert panel["type"] == "xychart"
        assert panel["fieldConfig"]["defaults"]["custom"]["show"] == "bars"
        assert panel["gridPos"]["x"] == x
        assert panel["gridPos"]["y"] == y

        sql = panel["targets"][0]["rawSql"]
        assert f"ARRAY['action_pipeline', '{stage}', 'orders']" in sql
        assert "ControllerDiagnostics" in sql
        assert "role_exchanges" not in sql
        assert "exchange:sqlstring" in sql
        assert "'${target_frame_mode}' = 'latest'" in sql
        assert "OR e.event_id = ${target_frame_event_id:sqlstring}" in sql
        assert "jsonb_typeof" in sql
        assert "ELSE '[]'::jsonb" in sql
        assert "bucket_grid AS" in sql
        assert "generate_series" in sql
        assert "'axis range' AS series" not in sql
        if stage == "ladder":
            assert "action_pipeline', 'continuous', 'bid'" in sql

    vars_by_name = {var["name"]: var for var in dashboard["templating"]["list"]}
    assert vars_by_name["target_frame_mode"]["current"]["value"] == "latest"
    assert vars_by_name["target_frame_event_id"]["label"] == "Target frame"
    assert vars_by_name["spread_buckets"]["label"] == "Spread buckets"


def test_pipeline_time_profile_dashboard_has_end_to_end_latency_series() -> None:
    dashboard = pipeline_time_profile()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    assert dashboard["uid"] == "mm-obs-v2-pipeline-time-profile"
    assert dashboard["title"] == "MM Observability v2 / 02B Pipeline Time Profile"

    expected_titles = {
        "End-to-end pipeline latency / frame",
        "Cumulative pipeline offset vs orderbook timestamp",
        "Cumulative pipeline offset vs frame close (t_end_ms)",
        "Raw feed receive lag latest",
        "Optimizer substage latency / frame",
        "Slowest recent frames",
    }
    assert expected_titles.issubset(panels_by_title)

    pipeline_sql = panels_by_title["End-to-end pipeline latency / frame"]["targets"][0]["rawSql"]
    assert "mm_market_facts" in pipeline_sql
    assert "market.orderbook_age_at_frame_close" in pipeline_sql
    assert "market.frame_persist_lag" in pipeline_sql
    assert "strategy.clean" in pipeline_sql
    assert "strategy.decide_total" in pipeline_sql
    assert "strategy.optimizer_sample_draw" in pipeline_sql
    assert "strategy.optimizer_train_loop" in pipeline_sql
    assert "strategy.optimizer_total" in pipeline_sql
    assert "strategy.risk" in pipeline_sql
    assert "strategy.publish" in pipeline_sql
    assert "execution.cancel" in pipeline_sql
    assert "execution.place" in pipeline_sql
    assert "execution.sync" in pipeline_sql
    assert "controller.total" in pipeline_sql
    assert "e.payload #>> ARRAY['optimizer','timing'" in pipeline_sql

    orderbook_sql = panels_by_title["Cumulative pipeline offset vs orderbook timestamp"]["targets"][0]["rawSql"]
    assert "+orderbook.ref" in orderbook_sql
    assert "+frame_close" in orderbook_sql
    assert "+after_cancel" in orderbook_sql
    assert "+after_place" in orderbook_sql
    assert "local_orderbook_timestamp_ms" in orderbook_sql
    assert "controller_start_ms" in orderbook_sql
    assert "f.controller_start_ms + COALESCE(f.controller_total_ms, 0)" in orderbook_sql
    assert "f.controller_start_ms - f.controller_total_ms" not in orderbook_sql
    assert "LPAD(s.pipeline_order::text, 2, '0')" in orderbook_sql
    assert "ORDER BY 1, s.pipeline_order" in orderbook_sql
    assert 'AS "time" AS "time"' not in orderbook_sql
    orderbook_panel = panels_by_title["Cumulative pipeline offset vs orderbook timestamp"]
    assert orderbook_panel["options"]["legend"]["sortBy"] == "Name"
    assert orderbook_panel["options"]["legend"]["sortDesc"] is False

    t_end_sql = panels_by_title["Cumulative pipeline offset vs frame close (t_end_ms)"]["targets"][0]["rawSql"]
    assert "+frame_close.ref" in t_end_sql
    assert "+frame_assembled" in t_end_sql
    assert "+pipeline_end" in t_end_sql
    assert "controller_start_ms" in t_end_sql
    assert "f.controller_start_ms + COALESCE(f.controller_total_ms, 0)" in t_end_sql
    assert "f.controller_start_ms - f.controller_total_ms" not in t_end_sql
    assert "LPAD(s.pipeline_order::text, 2, '0')" in t_end_sql
    assert "ORDER BY 1, s.pipeline_order" in t_end_sql
    assert 'AS "time" AS "time"' not in t_end_sql
    t_end_panel = panels_by_title["Cumulative pipeline offset vs frame close (t_end_ms)"]
    assert t_end_panel["options"]["legend"]["sortBy"] == "Name"
    assert t_end_panel["options"]["legend"]["sortDesc"] is False

    raw_panel = panels_by_title["Raw feed receive lag latest"]
    assert raw_panel["targets"][0]["datasource"]["type"] == "prometheus"
    assert "mm_raw_market_ingest_lag_latest_ms" in raw_panel["targets"][0]["expr"]
    assert 'kind="orderbook"' in raw_panel["targets"][0]["expr"]

    optimizer_sql = panels_by_title["Optimizer substage latency / frame"]["targets"][0]["rawSql"]
    assert "sample_draw_sec" in optimizer_sql
    assert "train_sec" in optimizer_sql
    assert "total_sec" in optimizer_sql
    assert "decide_ms" in optimizer_sql

    slow_sql = panels_by_title["Slowest recent frames"]["targets"][0]["rawSql"]
    assert "controller_total_ms" in slow_sql
    assert "ORDER BY COALESCE" in slow_sql
    assert "LIMIT 50" in slow_sql

    assert pipeline_time_profile()["uid"] in {dashboard["uid"] for dashboard in dashboards()}


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


def test_model_parameters_dashboard_flattens_local_and_global_params() -> None:
    dashboard = model_parameters()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    expected = {
        "Formula Reminder",
        "Latest model snapshot",
        "Model NLL / step",
        "Local price gate",
        "Global price gate",
        "Local price coefficients and means",
        "Global price coefficients and means",
        "Local price uncertainty",
        "Global price uncertainty",
        "Local market-flow params",
        "Global market-flow params",
        "Local price scale/config",
        "Global price scale/config",
        "All local params",
        "All global params",
    }
    assert dashboard["uid"] == "mm-obs-v2-model-parameters"
    assert dashboard["title"] == "MM Observability v2 / 03B Model Parameters"
    assert expected.issubset(panels_by_title)
    formula = panels_by_title["Formula Reminder"]["options"]["content"]
    assert "q_market(t)" in formula
    assert "market_beta.c[0]" in formula
    assert "buy_vwap(depth)" in formula
    assert "sell_vwap(depth)" in formula

    local_all_sql = panels_by_title["All local params"]["targets"][0]["rawSql"]
    local_config_sql = panels_by_title["Local price scale/config"]["targets"][0]["rawSql"]
    global_gate_sql = panels_by_title["Global price gate"]["targets"][0]["rawSql"]

    assert "WITH RECURSIVE" in local_all_sql
    assert "local_price_model" in local_all_sql
    assert "local_market_flow_model" in local_all_sql
    assert "global_price_model" in global_gate_sql
    assert "gate_beta" in global_gate_sql
    assert "jsonb_array_elements" in local_all_sql
    assert "jsonb_each" in local_all_sql
    assert "active_in_mpc" in local_all_sql
    assert "StrategyDecision" in local_all_sql
    assert "reference_vwap_vol" in local_config_sql
    assert "shape_alpha" in local_config_sql
    assert "student_df" in local_config_sql
    assert "metric !~" in local_all_sql
    assert "reference_vwap_vol" in local_all_sql


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


def test_exchange_account_dashboard_uses_physical_snapshot_sources() -> None:
    dashboard = exchange_account()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}
    templating = {item["name"]: item for item in dashboard["templating"]["list"]}

    assert dashboard["uid"] == "mm-obs-v2-exchange-account"
    assert dashboard["time"]["from"] == "now-24h"
    assert templating["diagnostics_topic"]["hide"] == 2
    assert templating["interval_ms"]["hide"] == 2

    balances_sql = panels_by_title["Latest account balances"]["targets"][0]["rawSql"]
    assert "PrivateStateSnapshot" in balances_sql
    assert "private:state:%:snapshot" in balances_sql
    assert "$__timeFilter(e.ts)" not in balances_sql

    age_sql = panels_by_title["Private snapshot age"]["targets"][0]["rawSql"]
    assert "$__timeFilter(e.ts)" not in age_sql

    errors_sql = panels_by_title["Recent private poll errors"]["targets"][0]["rawSql"]
    assert "private:state:%:errors" in errors_sql
    assert "401" not in errors_sql

    step_sql = panels_by_title["Private poll errors / step"]["targets"][0]["rawSql"]
    assert "private:state:%:errors" in step_sql
    assert "$__timeFilter(e.ts)" in step_sql

    venue_sql = panels_by_title["Latest venue ledger"]["targets"][0]["rawSql"]
    assert "VenueLedgerSnapshot" in venue_sql
    assert "JointLedgerSnapshot" in venue_sql
    assert "ledger:snapshot" in venue_sql
    assert "available_base" in venue_sql
    assert "open_orders" in venue_sql
    assert "$__timeFilter(e.ts)" not in venue_sql

    inventory_sql = panels_by_title["Trading inventory (base)"]["targets"][0]["rawSql"]
    assert "available_base" in inventory_sql
    assert "locked_base" in inventory_sql
    assert "$__timeFilter(e.ts)" in inventory_sql


def test_diagnostics_and_target_book_filters_use_topic_patterns() -> None:
    combined_sql = "\n".join(
        target["rawSql"]
        for dashboard in dashboards()
        for panel in dashboard["panels"]
        for target in panel.get("targets", ())
        if target.get("datasource", {}).get("type") == "postgres"
    )
    assert "e.topic = '${diagnostics_topic}'" not in combined_sql
    assert "topic = '${target_books_topic}'" not in combined_sql
    assert "e.topic LIKE 'strategy:diagnostics:%'" in combined_sql
    assert "e.topic LIKE 'strategy:target_books:%'" in combined_sql


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
        assert f"ARRAY['state','{role}_quote_wallet','total']" in sql
        assert f"ARRAY['state','{role}_base_wallet','total']" in sql
        assert f"ARRAY['state','{role}_cash']" in sql
        assert f"ARRAY['state','{role}_inventory']" in sql
        assert f"ARRAY['state','{role}_sell_vwap']" in sql
        assert f"'{role}_cash' AS metric" in sql
        assert f"'{role}_inventory_value'" in sql
        assert f"'{role}_wealth'" in sql
        assert "COALESCE(reported_wealth, cash + inventory * mark_price) AS value" in sql

    decomposition_expected = {
        "Cumulative net spread capture PnL": (0, 27, "(delta_cash + delta_inventory * mark_price_prev)"),
        "Cumulative inventory drift PnL": (0, 35, "(inventory_prev * delta_price)"),
        "Cumulative adverse selection PnL": (0, 43, "(delta_inventory * delta_price)"),
    }
    for title, (x, y, component_expr) in decomposition_expected.items():
        panel = panels_by_title[title]
        assert panel["type"] == "timeseries"
        assert panel["gridPos"]["x"] == x
        assert panel["gridPos"]["y"] == y
        assert panel["gridPos"]["w"] == 24
        sql = panel["targets"][0]["rawSql"]
        assert "LAG(cash) OVER (PARTITION BY metric ORDER BY \"time\")" in sql
        assert "LAG(inventory) OVER (PARTITION BY metric ORDER BY \"time\")" in sql
        assert "LAG(mark_price) OVER (PARTITION BY metric ORDER BY \"time\")" in sql
        assert component_expr in sql
        assert "SUM(step_value) OVER" in sql
        assert "PARTITION BY metric" in sql
        assert "'local' AS metric" in sql
        assert "'global'" in sql
        assert "'total'" in sql
        assert "ARRAY['state','local_quote_wallet','total']" in sql
        assert "ARRAY['state','global_quote_wallet','total']" in sql
        assert "ARRAY['state','local_base_wallet','total']" in sql
        assert "ARRAY['state','global_base_wallet','total']" in sql
        assert "ARRAY['optimizer','config','reference_price']" in sql

    assert "Local interval deltas" not in panels_by_title

    assert "Step economics" not in panels_by_title


def test_target_order_stems_dashboard_aligns_price_qty_stage_charts() -> None:
    dashboard = target_order_stems()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    stages = [
        ("01 Strategy raw ladder", 3, "ladder", "orders", None),
        ("02 Strategy cap -> target orderbook", 10, "capped", "orders", None),
        ("03 Risk-approved target", 17, "risk_approved_target", "orders", None),
        ("04 Planner new place orders", 24, "planned_place", "orders", None),
        ("05 Venue rules valid after rounding", 31, "rules_valid", "orders", None),
        ("06 Exchange accepted placed orders", 38, "router_accepted", "orders", None),
        ("07 Planner kept live orders", 45, "kept", "orders", None),
        ("08 Planner cancel orders", 52, "planned_cancel", "orders", None),
        ("09 Removed by local risk / collateral", 59, "risk_approved_target", "rejections", None),
        ("10 Removed by venue adapter: invalid price/qty", 66, "rules_valid", "rejections", "rules_invalid_price_or_qty"),
        ("11 Removed by venue adapter: min qty", 73, "rules_valid", "rejections", "rules_min_qty"),
        ("12 Removed by venue adapter: min notional", 80, "rules_valid", "rejections", "rules_min_notional"),
    ]
    stem_titles: list[str] = []
    stage_by_title: dict[str, tuple[str, str, str | None, str]] = {}
    for stage_title, y, stage, field, reason in stages:
        local_title = f"Local {stage_title}"
        global_stage_title = stage_title.replace("09 Removed by local risk", "09 Removed by global risk")
        global_title = f"Global {global_stage_title}"
        expected = ((local_title, 0, "local"), (global_title, 12, "global"))
        for title, x, role in expected:
            stem_titles.append(title)
            stage_by_title[title] = (stage, field, reason, role)
            assert title in panels_by_title
            assert panels_by_title[title]["gridPos"]["x"] == x
            assert panels_by_title[title]["gridPos"]["w"] == 12
            assert panels_by_title[title]["gridPos"]["h"] == 7
            assert panels_by_title[title]["gridPos"]["y"] == y

    for title in stem_titles:
        panel = panels_by_title[title]
        assert panel["type"] == "xychart"
        assert panel["fieldConfig"]["defaults"]["custom"]["show"] == "bars"
        assert panel["transformations"][0]["options"]["fields"] == ["series"]
        spread_override = next(
            override
            for override in panel["fieldConfig"]["overrides"]
            if override["matcher"]["options"] == "spread"
        )
        assert {"id": "custom.axisCenteredZero", "value": True} in spread_override["properties"]
        overrides = str(panel["fieldConfig"]["overrides"])
        assert ".* bid)$" in overrides or "bid" in overrides
        assert "ask" in overrides
        assert "green" in overrides
        assert "red" in overrides

    mode_var = next(var for var in dashboard["templating"]["list"] if var["name"] == "target_frame_mode")
    assert mode_var["label"] == "Target frame mode"
    assert mode_var["current"]["value"] == "latest"
    assert [option["value"] for option in mode_var["options"]] == ["latest", "pinned"]
    frame_var = next(var for var in dashboard["templating"]["list"] if var["name"] == "target_frame_event_id")
    assert frame_var["label"] == "Target frame"
    assert "e.event_id AS __value" in frame_var["query"]
    assert "ControllerDiagnostics" in frame_var["query"]
    spread_var = next(var for var in dashboard["templating"]["list"] if var["name"] == "spread_max")
    assert spread_var["label"] == "Spread max"
    bucket_var = next(var for var in dashboard["templating"]["list"] if var["name"] == "spread_buckets")
    assert bucket_var["label"] == "Spread buckets"

    for title in stem_titles:
        stage, field, reason, role = stage_by_title[title]
        sql = panels_by_title[title]["targets"][0]["rawSql"]
        assert f"ARRAY['action_pipeline', '{stage}', '{field}']" in sql
        if title.endswith("09 Removed by local risk / collateral") or title.endswith("09 Removed by global risk / collateral"):
            assert "COALESCE(NULLIF(o.value->>'reason', ''), 'unknown')" in sql
        if reason is not None:
            assert reason in sql
        assert "role_exchanges" in sql
        assert f"WHERE role = '{role}'" in sql
        assert "ControllerDiagnostics" in sql
        assert "'${target_frame_mode}' = 'latest'" in sql
        assert "OR e.event_id = ${target_frame_event_id:sqlstring}" in sql
        assert "bucket_grid AS" in sql
        assert "generate_series" in sql
        assert "spread_max" in sql
        assert "'axis range' AS series" not in sql
        assert "FROM axis_rows" not in sql
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


def test_execution_errors_dashboard_surfaces_adapter_and_validation_failures() -> None:
    dashboard = execution_errors()
    panels_by_title = {panel["title"]: panel for panel in dashboard["panels"]}

    expected_titles = {
        "Execution error totals",
        "Errors by exchange",
        "Execution error flow / step",
        "Errors by reason / step",
        "Grouped exchange error messages",
        "Recent execution errors",
        "Runtime error logs",
    }
    assert expected_titles.issubset(panels_by_title)
    assert dashboard["uid"] == "mm-obs-v2-execution-errors"
    assert dashboard["title"] == "MM Observability v2 / 06B Execution Errors"

    flow_sql = panels_by_title["Execution error flow / step"]["targets"][0]["rawSql"]
    reason_sql = panels_by_title["Errors by reason / step"]["targets"][0]["rawSql"]
    grouped_sql = panels_by_title["Grouped exchange error messages"]["targets"][0]["rawSql"]
    recent_sql = panels_by_title["Recent execution errors"]["targets"][0]["rawSql"]
    logs_expr = panels_by_title["Runtime error logs"]["targets"][0]["expr"]

    for sql in (flow_sql, reason_sql, grouped_sql, recent_sql):
        assert "PlaceOrdersFailed" in sql
        assert "CancelOrdersFailed" in sql
        assert "OrderValidationRejected" in sql
        assert "ExecutionSyncFailed" in sql

    assert "PlaceOrdersRequested" in flow_sql
    assert "PlaceOrdersAccepted" in flow_sql
    assert "payload->>'message'" in grouped_sql
    assert "jsonb_array_elements" in recent_sql
    assert "PlaceOrdersFailed" in logs_expr


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
        "Total filled bid volume": (0, 3, 6),
        "Local filled bid volume": (6, 3, 6),
        "Global filled bid volume": (12, 3, 6),
        "Average bid fill price": (18, 3, 6),
        "Total filled ask volume": (0, 8, 6),
        "Local filled ask volume": (6, 8, 6),
        "Global filled ask volume": (12, 8, 6),
        "Average ask fill price": (18, 8, 6),
        "Spread PnL": (0, 13, 6),
        "Inventory PnL": (6, 13, 6),
        "Fee PnL": (12, 13, 6),
    }

    for title, (x, y, w) in expected_positions.items():
        panel = panels_by_title[title]
        assert panel["type"] == "stat"
        assert panel["gridPos"] == {"h": 5, "w": w, "x": x, "y": y}
        assert panel["fieldConfig"]["defaults"]["thresholds"] == {
            "mode": "absolute",
            "steps": [
                {"color": "red", "value": None},
                {"color": "gray", "value": 0},
                {"color": "green", "value": 1e-12},
            ],
        }
        sql = panel["targets"][0]["rawSql"]
        assert "$__timeFilter(e.ts)" in sql
        assert "private:state:%:fills" in sql
        assert "role_exchanges" in sql
        assert "COALESCE(e.exchange, e.payload->>'exchange'" in sql
        assert "COALESCE(e.symbol, e.payload->>'symbol', '')" in sql
        assert "SUM(price * qty) FILTER" in sql
        assert "e.payload->>'fee'" in sql
        assert "interval_mark" in sql

    total_bid_volume_sql = panels_by_title["Total filled bid volume"]["targets"][0]["rawSql"]
    assert "COALESCE(bid_qty, 0.0) AS value" in total_bid_volume_sql
    assert "WHERE role = " not in total_bid_volume_sql

    bid_volume_sql = panels_by_title["Local filled bid volume"]["targets"][0]["rawSql"]
    assert "COALESCE(bid_qty, 0.0) AS value" in bid_volume_sql
    assert "WHERE role = 'local'" in bid_volume_sql

    global_bid_volume_sql = panels_by_title["Global filled bid volume"]["targets"][0]["rawSql"]
    assert "COALESCE(bid_qty, 0.0) AS value" in global_bid_volume_sql
    assert "WHERE role = 'global'" in global_bid_volume_sql

    total_ask_volume_sql = panels_by_title["Total filled ask volume"]["targets"][0]["rawSql"]
    assert "COALESCE(ask_qty, 0.0) AS value" in total_ask_volume_sql
    assert "WHERE role = " not in total_ask_volume_sql

    ask_volume_sql = panels_by_title["Local filled ask volume"]["targets"][0]["rawSql"]
    assert "COALESCE(ask_qty, 0.0) AS value" in ask_volume_sql
    assert "WHERE role = 'local'" in ask_volume_sql

    global_ask_volume_sql = panels_by_title["Global filled ask volume"]["targets"][0]["rawSql"]
    assert "COALESCE(ask_qty, 0.0) AS value" in global_ask_volume_sql
    assert "WHERE role = 'global'" in global_ask_volume_sql

    bid_avg_sql = panels_by_title["Average bid fill price"]["targets"][0]["rawSql"]
    assert "bid_avg_price AS value" in bid_avg_sql
    assert "WHERE role = " not in bid_avg_sql

    ask_avg_sql = panels_by_title["Average ask fill price"]["targets"][0]["rawSql"]
    assert "ask_avg_price AS value" in ask_avg_sql
    assert "WHERE role = " not in ask_avg_sql

    spread_sql = panels_by_title["Spread PnL"]["targets"][0]["rawSql"]
    assert (
        "LEAST(COALESCE(bid_qty, 0.0), COALESCE(ask_qty, 0.0)) "
        "* (COALESCE(ask_avg_price, 0.0) - COALESCE(bid_avg_price, 0.0)) AS value"
    ) in spread_sql

    inventory_sql = panels_by_title["Inventory PnL"]["targets"][0]["rawSql"]
    assert "WHEN COALESCE(bid_qty, 0.0) >= COALESCE(ask_qty, 0.0) THEN" in inventory_sql
    assert "interval_mark.mark_price - COALESCE(bid_avg_price, 0.0)" in inventory_sql
    assert "COALESCE(ask_avg_price, 0.0) - interval_mark.mark_price" in inventory_sql

    fee_sql = panels_by_title["Fee PnL"]["targets"][0]["rawSql"]
    assert "SUM(fee) AS total_fee" in fee_sql
    assert "-COALESCE(total_fee, 0.0) AS value" in fee_sql
    assert "global_maker_fee" not in fee_sql
    assert "local_maker_fee" not in fee_sql
    assert "fee_config" not in fee_sql
    assert "e.payload->>'fee'" in fee_sql


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
