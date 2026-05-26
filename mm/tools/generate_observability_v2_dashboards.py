#!/usr/bin/env python3
"""Generate the Observability v2 Grafana dashboards.

The existing dashboards stay in the ``Market Maker`` folder. These dashboards
are provisioned by a separate provider into ``Market Maker - Observability v2``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "grafana" / "observability-v2"

PROM_DS = {"type": "prometheus", "uid": "prometheus-mm"}
PG_DS = {"type": "postgres", "uid": "postgres-mm"}
LOKI_DS = {"type": "loki", "uid": "loki-mm"}


def _target(
    datasource: dict[str, str],
    expr: str,
    *,
    ref: str = "A",
    fmt: str = "time_series",
    legend: str | None = None,
) -> dict[str, Any]:
    if datasource["type"] == "prometheus":
        target = {"datasource": datasource, "expr": expr, "refId": ref}
        if legend:
            target["legendFormat"] = legend
        return target
    if datasource["type"] == "loki":
        target = {"datasource": datasource, "expr": expr, "refId": ref}
        if legend:
            target["legendFormat"] = legend
        return target
    return {
        "datasource": datasource,
        "editorMode": "code",
        "format": fmt,
        "rawQuery": True,
        "rawSql": expr,
        "refId": ref,
    }


def _panel(
    id_: int,
    title: str,
    panel_type: str,
    datasource: dict[str, str],
    expr: str,
    *,
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    fmt: str = "time_series",
    unit: str = "short",
    legend: str | None = None,
) -> dict[str, Any]:
    panel: dict[str, Any] = {
        "datasource": datasource,
        "fieldConfig": {"defaults": {"unit": unit, "decimals": 3}, "overrides": []},
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": id_,
        "targets": [_target(datasource, expr, fmt=fmt, legend=legend)],
        "title": title,
        "type": panel_type,
    }
    if panel_type == "timeseries":
        panel["options"] = {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "none"},
        }
    elif panel_type == "stat":
        panel["options"] = {"colorMode": "value", "graphMode": "area", "justifyMode": "auto"}
    elif panel_type == "table":
        panel["options"] = {"cellHeight": "sm", "footer": {"countRows": False}, "showHeader": True}
    elif panel_type == "logs":
        panel["options"] = {"dedupStrategy": "none", "enableLogDetails": True, "showCommonLabels": False}
    return panel


def _text(id_: int, title: str, content: str, *, y: int) -> dict[str, Any]:
    return {
        "gridPos": {"h": 3, "w": 24, "x": 0, "y": y},
        "id": id_,
        "options": {"content": content, "mode": "markdown"},
        "title": title,
        "transparent": True,
        "type": "text",
    }


def _row(id_: int, title: str, *, y: int) -> dict[str, Any]:
    return {"collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "id": id_, "panels": [], "title": title, "type": "row"}


def _dashboard(uid: str, title: str, panels: list[dict[str, Any]], *, tags: list[str]) -> dict[str, Any]:
    return {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "liveNow": True,
        "panels": panels,
        "refresh": "10s",
        "schemaVersion": 39,
        "style": "dark",
        "tags": ["market-maker", "observability-v2", *tags],
        "templating": {"list": _templating()},
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": title,
        "uid": uid,
        "version": 1,
        "weekStart": "",
    }


def _templating() -> list[dict[str, Any]]:
    return [
        _query_var("exchange", "Exchange", "SELECT DISTINCT exchange FROM mm_events WHERE exchange IS NOT NULL ORDER BY 1", multi=True),
        _query_var("symbol", "Symbol", "SELECT DISTINCT symbol FROM mm_events WHERE symbol IS NOT NULL ORDER BY 1", multi=True),
        _query_var("actor_id", "Actor", "SELECT DISTINCT actor_id FROM mm_events WHERE actor_id IS NOT NULL ORDER BY 1", multi=True),
        _query_var("run_id", "Run", "SELECT DISTINCT run_id FROM mm_events WHERE run_id <> '' ORDER BY 1 DESC", multi=True),
        _query_var("diagnostics_topic", "Diagnostics topic", "SELECT DISTINCT topic FROM mm_events WHERE topic LIKE 'strategy:diagnostics:%' ORDER BY 1"),
        _query_var("target_books_topic", "Target books topic", "SELECT DISTINCT topic FROM mm_events WHERE topic LIKE 'strategy:target_books:%' ORDER BY 1"),
        _query_var(
            "interval_ms",
            "Interval ms",
            "SELECT DISTINCT COALESCE(payload->>'interval_ms', split_part(topic, ':', 5)) AS interval_ms FROM mm_events WHERE COALESCE(payload->>'interval_ms', split_part(topic, ':', 5)) <> '' ORDER BY 1",
        ),
    ]


def _query_var(name: str, label: str, query: str, *, multi: bool = False) -> dict[str, Any]:
    return {
        "current": {"selected": False, "text": "All", "value": "$__all"} if multi else {},
        "datasource": PG_DS,
        "definition": query,
        "hide": 0,
        "includeAll": multi,
        "label": label,
        "multi": multi,
        "name": name,
        "options": [],
        "query": query,
        "refresh": 1,
        "sort": 1,
        "type": "query",
    }


def _custom_var(name: str, label: str, query: str) -> dict[str, Any]:
    first = query.split(",", 1)[0]
    return {
        "current": {"selected": True, "text": first, "value": first},
        "hide": 0,
        "includeAll": False,
        "label": label,
        "multi": False,
        "name": name,
        "options": [{"selected": i == 0, "text": item, "value": item} for i, item in enumerate(query.split(","))],
        "query": query,
        "type": "custom",
    }


def runtime_overview() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Fast runtime health from Prometheus, plus latest controller warnings from PostgreSQL.", y=0),
        _panel(2, "Scrape up", "stat", PROM_DS, 'up{job="market_maker"}', x=0, y=3, w=6, h=4, legend="{{job}}"),
        _panel(3, "Frame rate", "timeseries", PROM_DS, 'sum by (exchange,symbol)(rate(mm_joint_strategy_frames_total[$__rate_interval]))', x=6, y=3, legend="{{exchange}} {{symbol}}"),
        _panel(4, "Decision rate", "timeseries", PROM_DS, 'sum by (actor_id,exchange,symbol)(rate(mm_controller_decisions_total[$__rate_interval]))', x=18, y=3, w=6, legend="{{exchange}} {{symbol}}"),
        _panel(5, "Controller stage p95", "timeseries", PROM_DS, 'histogram_quantile(0.95, sum by (le,stage)(rate(mm_controller_stage_latency_ms_bucket[$__rate_interval])))', x=0, y=11, legend="{{stage}}"),
        _panel(6, "Order sync p95", "timeseries", PROM_DS, 'histogram_quantile(0.95, sum by (le,exchange)(rate(mm_order_sync_latency_ms_bucket[$__rate_interval])))', x=12, y=11, legend="{{exchange}}"),
        _panel(7, "Execution completed / failed", "timeseries", PROM_DS, 'sum by (exchange,__name__)(rate({__name__=~"mm_execution_sync_completed_total|mm_execution_sync_failed_total"}[$__rate_interval]))', x=0, y=19, legend="{{exchange}} {{__name__}}"),
        _panel(8, "Live orders", "timeseries", PROM_DS, "mm_execution_live_orders", x=12, y=19, legend="{{exchange}} {{symbol}}"),
        _panel(9, "Persistence errors", "timeseries", PROM_DS, 'sum by (writer,error)(rate(mm_persistence_errors_total[$__rate_interval]))', x=0, y=27, legend="{{writer}} {{error}}"),
        _panel(10, "Latest controller warnings", "table", PG_DS, _latest_warnings_sql(), x=12, y=27, fmt="table"),
    ]
    return _dashboard("mm-obs-v2-runtime-overview", "MM Observability v2 / 00 Runtime Overview", panels, tags=["runtime"])


def market_data_quality() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Market data quality from persisted `MarketEvent` rows.", y=0),
        _panel(2, "Market event rate", "timeseries", PG_DS, _market_event_rate_sql(), x=0, y=3),
        _panel(3, "Market event age ms", "timeseries", PG_DS, _market_age_sql(), x=12, y=3),
        _panel(4, "Best bid / ask", "timeseries", PG_DS, _best_bid_ask_sql(), x=0, y=11),
        _panel(5, "Spread", "timeseries", PG_DS, _spread_sql(), x=12, y=11),
        _panel(6, "Buy / sell volume", "timeseries", PG_DS, _volume_sql(), x=0, y=19),
        _panel(7, "Latency skew", "timeseries", PROM_DS, "mm_latency_skew_ms", x=12, y=19, legend="{{exchange}} {{symbol}}"),
    ]
    return _dashboard("mm-obs-v2-market-data", "MM Observability v2 / 01 Market Data Quality", panels, tags=["market-data"])


def controller_health() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Controller frame freshness, readiness, warnings, and stage timings.", y=0),
        _panel(2, "Frame age", "timeseries", PROM_DS, 'mm_frame_age_ms{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"}', x=0, y=3, legend="{{exchange}} {{symbol}}"),
        _panel(3, "Frame readiness", "timeseries", PROM_DS, 'mm_frame_sync_ready{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"}', x=12, y=3, legend="{{exchange}} {{symbol}}"),
        _panel(4, "Stage latency p95", "timeseries", PROM_DS, 'histogram_quantile(0.95, sum by (le,stage)(rate(mm_controller_stage_latency_ms_bucket{actor_id=~"${actor_id:regex}"}[$__rate_interval])))', x=0, y=11, legend="{{stage}}"),
        _panel(5, "Warnings", "timeseries", PROM_DS, 'sum by (reason)(rate(mm_controller_warnings_total{actor_id=~"${actor_id:regex}"}[$__rate_interval]))', x=12, y=11, legend="{{reason}}"),
        _panel(6, "Recent diagnostics warnings", "table", PG_DS, _latest_warnings_sql(limit=100), x=0, y=19, w=24, fmt="table"),
    ]
    return _dashboard("mm-obs-v2-controller-health", "MM Observability v2 / 02 Frame And Controller Health", panels, tags=["controller"])


def model_quality() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Model NLL, residuals, and skip reasons emitted from `ControllerDiagnostics.models`.", y=0),
        _panel(2, "Model NLL p95", "timeseries", PROM_DS, 'histogram_quantile(0.95, sum by (le,model)(rate(mm_model_nll_bucket{actor_id=~"${actor_id:regex}"}[$__rate_interval])))', x=0, y=3, legend="{{model}}"),
        _panel(3, "Standardized residual", "timeseries", PROM_DS, 'mm_model_standardized_residual{actor_id=~"${actor_id:regex}"}', x=12, y=3, legend="{{model}}"),
        _panel(4, "Score skipped", "timeseries", PROM_DS, 'sum by (model,reason)(rate(mm_model_score_skipped_total{actor_id=~"${actor_id:regex}"}[$__rate_interval]))', x=0, y=11, legend="{{model}} {{reason}}"),
        _panel(5, "Latest model scores", "table", PG_DS, _latest_model_scores_sql(), x=0, y=19, w=24, fmt="table"),
    ]
    return _dashboard("mm-obs-v2-model-quality", "MM Observability v2 / 03 Strategy Model Quality", panels, tags=["models"])


def optimizer_diagnostics() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Optimizer objective, contribution fractions, timing, and gradient health.", y=0),
        _panel(2, "Objective value", "timeseries", PG_DS, _optimizer_objective_sql(), x=0, y=3),
        _panel(3, "Objective improvement", "timeseries", PG_DS, _optimizer_improvement_sql(), x=12, y=3),
        _panel(4, "Contribution fractions", "timeseries", PG_DS, _optimizer_contrib_sql(), x=0, y=11),
        _panel(5, "Optimizer timings", "timeseries", PG_DS, _optimizer_timing_sql(), x=12, y=11),
        _panel(6, "Nonfinite counts", "timeseries", PG_DS, _optimizer_gradients_sql(), x=0, y=19, w=24),
    ]
    return _dashboard("mm-obs-v2-optimizer", "MM Observability v2 / 04 Optimizer Diagnostics", panels, tags=["optimizer"])


def action_pipeline() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Target-book execution funnel. Early stages are optimizer/order shaping; target_book and later stages are the current risk, planner, rules, and router path.", y=0),
        _panel(2, "Order count funnel", "timeseries", PROM_DS, 'mm_action_stage_orders{actor_id=~"${actor_id:regex}",side="all"}', x=0, y=3, legend="{{exchange}} {{stage}}"),
        _panel(3, "Qty by stage", "timeseries", PROM_DS, 'mm_action_stage_qty{actor_id=~"${actor_id:regex}"}', x=12, y=3, legend="{{exchange}} {{stage}} {{side}}"),
        _panel(4, "Notional by stage", "timeseries", PROM_DS, 'mm_action_stage_notional{actor_id=~"${actor_id:regex}"}', x=0, y=11, legend="{{exchange}} {{stage}} {{side}}"),
        _panel(5, "Action rejections", "timeseries", PROM_DS, 'sum by (stage,reason)(rate(mm_action_rejected_total{actor_id=~"${actor_id:regex}"}[$__rate_interval]))', x=12, y=11, legend="{{stage}} {{reason}}"),
        _panel(6, "Pipeline stage rows", "table", PG_DS, _action_pipeline_table_sql(), x=0, y=19, w=24, fmt="table"),
    ]
    return _dashboard("mm-obs-v2-action-pipeline", "MM Observability v2 / 05 Action Pipeline Funnel", panels, tags=["actions"])


def risk_execution() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Risk decisions, execution gates, order sync lifecycle, and adapter health.", y=0),
        _panel(2, "Risk approved", "timeseries", PROM_DS, 'mm_risk_approved{actor_id=~"${actor_id:regex}"}', x=0, y=3, legend="{{exchange}}"),
        _panel(3, "Risk rejects", "timeseries", PROM_DS, 'sum by (reason)(rate(mm_risk_reject_total{actor_id=~"${actor_id:regex}"}[$__rate_interval]))', x=12, y=3, legend="{{reason}}"),
        _panel(4, "Execution gate blocks", "timeseries", PROM_DS, 'sum by (reason)(rate(mm_execution_gate_block_total{actor_id=~"${actor_id:regex}"}[$__rate_interval]))', x=0, y=11, legend="{{reason}}"),
        _panel(5, "Sync completed / failed", "timeseries", PROM_DS, 'sum by (exchange,__name__)(rate({__name__=~"mm_execution_sync_completed_total|mm_execution_sync_failed_total"}[$__rate_interval]))', x=12, y=11, legend="{{exchange}} {{__name__}}"),
        _panel(6, "Router failures", "timeseries", PROM_DS, 'sum by (exchange,operation,error)(rate(mm_order_router_failures_total[$__rate_interval]))', x=0, y=19, legend="{{exchange}} {{operation}} {{error}}"),
        _panel(7, "Validation rejects", "timeseries", PROM_DS, 'sum by (exchange,reason)(rate(mm_order_validation_rejected_total[$__rate_interval]))', x=12, y=19, legend="{{exchange}} {{reason}}"),
        _panel(8, "Recent execution events", "table", PG_DS, _recent_execution_events_sql(), x=0, y=27, w=24, fmt="table"),
    ]
    return _dashboard("mm-obs-v2-risk-execution", "MM Observability v2 / 06 Risk And Execution", panels, tags=["risk", "execution"])


def portfolio_pnl() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Inventory, cash, wealth, and step economics.", y=0),
        _panel(2, "Inventory", "timeseries", PG_DS, _portfolio_field_sql("local_inventory", "global_inventory"), x=0, y=3),
        _panel(3, "Cash", "timeseries", PG_DS, _portfolio_field_sql("local_cash", "global_cash"), x=12, y=3),
        _panel(4, "Wealth", "timeseries", PG_DS, _portfolio_field_sql("local_wealth", "global_wealth"), x=0, y=11),
        _panel(5, "Risk exposure", "timeseries", PROM_DS, 'mm_risk_exposure{actor_id=~"${actor_id:regex}"}', x=12, y=11, legend="{{exchange}} {{phase}} {{field}}"),
        _panel(6, "Step economics", "timeseries", PG_DS, _step_economics_sql(), x=0, y=19, w=24),
    ]
    return _dashboard("mm-obs-v2-portfolio-pnl", "MM Observability v2 / 07 Portfolio And PnL", panels, tags=["portfolio"])


def liquidity_sourcing() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Local/global readiness and joint target-book routing.", y=0),
        _panel(2, "Readiness flags", "timeseries", PROM_DS, 'mm_frame_sync_ready{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"}', x=0, y=3, legend="{{exchange}} {{symbol}}"),
        _panel(3, "Latency skew", "timeseries", PROM_DS, 'mm_latency_skew_ms{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"}', x=12, y=3, legend="{{exchange}} {{symbol}}"),
        _panel(4, "Venue gates", "table", PG_DS, _venue_gates_sql(), x=0, y=11, w=24, fmt="table"),
        _panel(5, "Target book order count", "timeseries", PG_DS, _target_book_count_sql(), x=0, y=19),
        _panel(6, "Target book qty", "timeseries", PG_DS, _target_book_qty_sql(), x=12, y=19),
    ]
    return _dashboard("mm-obs-v2-liquidity", "MM Observability v2 / 08 Liquidity Sourcing", panels, tags=["liquidity"])


def runtime_logs() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Structured runtime logs from Loki.", y=0),
        _panel(2, "Log volume by event", "timeseries", LOKI_DS, 'sum by (event) (count_over_time({job="market-maker-runtime"} | json [$__interval]))', x=0, y=3, w=24, legend="{{event}}"),
        _panel(3, "Warnings and errors", "logs", LOKI_DS, '{job="market-maker-runtime"} | json | level=~"warning|error"', x=0, y=11, w=24, h=10),
        _panel(4, "Execution failures", "logs", LOKI_DS, '{job="market-maker-runtime"} | json | event=~"execution_sync_failed|adapter_request_failed"', x=0, y=21, w=24, h=10),
    ]
    return _dashboard("mm-obs-v2-runtime-logs", "MM Observability v2 / 09 Runtime Logs", panels, tags=["logs"])


def incident_drilldown() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Frame-level drilldown. Narrow the time range and variables to inspect one incident.", y=0),
        _panel(2, "Latest diagnostics payload", "table", PG_DS, _latest_payload_sql("diagnostics"), x=0, y=3, w=24, fmt="table"),
        _panel(3, "Latest strategy decision", "table", PG_DS, _latest_payload_sql("decision"), x=0, y=11, w=12, fmt="table"),
        _panel(4, "Latest target book", "table", PG_DS, _latest_payload_sql("target_book"), x=12, y=11, w=12, fmt="table"),
        _panel(5, "Matching execution events", "table", PG_DS, _recent_execution_events_sql(limit=100), x=0, y=19, w=24, fmt="table"),
        _panel(6, "Matching logs", "logs", LOKI_DS, '{job="market-maker-runtime"} | json | exchange=~"$exchange" | symbol=~"$symbol"', x=0, y=27, w=24, h=10),
    ]
    return _dashboard("mm-obs-v2-incident", "MM Observability v2 / 10 Incident Drilldown", panels, tags=["incident"])


def _diagnostics_filter() -> str:
    return """e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.exchange IN (${exchange:sqlstring})
  AND e.symbol IN (${symbol:sqlstring})
  AND e.actor_id IN (${actor_id:sqlstring})
  AND ($__timeFilter(e.ts))"""


def _latest_warnings_sql(limit: int = 25) -> str:
    return f"""SELECT e.ts, e.actor_id, e.exchange, e.symbol, w.warning
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(e.payload->'warnings', '[]'::jsonb)) AS w(warning)
WHERE {_diagnostics_filter()}
ORDER BY e.ts DESC
LIMIT {limit}"""


def _market_event_rate_sql() -> str:
    return """SELECT $__timeGroupAlias(ts, $__interval),
       (exchange || ' ' || symbol) AS metric,
       COUNT(*)::double precision AS value
FROM mm_events
WHERE event_type = 'MarketEvent'
  AND exchange IN (${exchange:sqlstring})
  AND symbol IN (${symbol:sqlstring})
  AND $__timeFilter(ts)
GROUP BY 1, exchange, symbol
ORDER BY 1"""


def _market_age_sql() -> str:
    return """SELECT ts AS "time",
       (exchange || ' ' || symbol) AS metric,
       GREATEST(0, EXTRACT(EPOCH FROM now()) * 1000 - (payload->>'t_end_ms')::double precision) AS value
FROM mm_events
WHERE event_type = 'MarketEvent'
  AND exchange IN (${exchange:sqlstring})
  AND symbol IN (${symbol:sqlstring})
  AND $__timeFilter(ts)
ORDER BY 1"""


def _best_bid_ask_sql() -> str:
    return """SELECT ts AS "time", (exchange || ' best_bid') AS metric,
       (payload->'orderbook'->'bids'->0->>'price')::double precision AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange IN (${exchange:sqlstring}) AND symbol IN (${symbol:sqlstring}) AND $__timeFilter(ts)
UNION ALL
SELECT ts AS "time", (exchange || ' best_ask') AS metric,
       (payload->'orderbook'->'asks'->0->>'price')::double precision AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange IN (${exchange:sqlstring}) AND symbol IN (${symbol:sqlstring}) AND $__timeFilter(ts)
ORDER BY 1"""


def _spread_sql() -> str:
    return """SELECT ts AS "time", (exchange || ' spread') AS metric,
       ((payload->'orderbook'->'asks'->0->>'price')::double precision -
        (payload->'orderbook'->'bids'->0->>'price')::double precision) AS value
FROM mm_events
WHERE event_type = 'MarketEvent'
  AND exchange IN (${exchange:sqlstring})
  AND symbol IN (${symbol:sqlstring})
  AND $__timeFilter(ts)
ORDER BY 1"""


def _volume_sql() -> str:
    return """SELECT ts AS "time", (exchange || ' buy_volume') AS metric, (payload->>'buy_volume')::double precision AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange IN (${exchange:sqlstring}) AND symbol IN (${symbol:sqlstring}) AND $__timeFilter(ts)
UNION ALL
SELECT ts AS "time", (exchange || ' sell_volume') AS metric, (payload->>'sell_volume')::double precision AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange IN (${exchange:sqlstring}) AND symbol IN (${symbol:sqlstring}) AND $__timeFilter(ts)
ORDER BY 1"""


def _latest_model_scores_sql() -> str:
    return f"""SELECT e.ts, e.actor_id, e.exchange, e.symbol,
       m.key AS model,
       m.value->>'kind' AS kind,
       m.value->>'skip_reason' AS skip_reason,
       (m.value->>'nll')::double precision AS nll,
       (m.value->>'standardized_residual')::double precision AS standardized_residual,
       m.value->'params_after' AS params_after
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(COALESCE(e.payload->'models', '{{}}'::jsonb)) AS m(key, value)
WHERE {_diagnostics_filter()}
ORDER BY e.ts DESC
LIMIT 100"""


def _optimizer_objective_sql() -> str:
    return f"""SELECT e.ts AS "time", stage AS metric,
       (e.payload #>> ARRAY['optimizer','objective',stage,'value'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['initial','final','best']) AS stage
WHERE {_diagnostics_filter()}
  AND e.payload #>> ARRAY['optimizer','objective',stage,'value'] IS NOT NULL
ORDER BY 1"""


def _optimizer_improvement_sql() -> str:
    return f"""SELECT e.ts AS "time", kv.key AS metric, (kv.value)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['optimizer','objective','improvement'], '{{}}'::jsonb)) AS kv
WHERE {_diagnostics_filter()}
ORDER BY 1"""


def _optimizer_contrib_sql() -> str:
    return f"""SELECT e.ts AS "time", kv.key AS metric, (kv.value)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['optimizer','objective','contribution_fractions'], '{{}}'::jsonb)) AS kv
WHERE {_diagnostics_filter()}
ORDER BY 1"""


def _optimizer_timing_sql() -> str:
    return f"""SELECT e.ts AS "time", kv.key AS metric,
       CASE WHEN kv.key LIKE '%_sec' THEN (kv.value)::double precision * 1000.0 ELSE (kv.value)::double precision END AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['optimizer','timing'], '{{}}'::jsonb)) AS kv
WHERE {_diagnostics_filter()}
  AND kv.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
ORDER BY 1"""


def _optimizer_gradients_sql() -> str:
    return f"""SELECT e.ts AS "time", kv.key AS metric, (kv.value)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['optimizer','gradients'], '{{}}'::jsonb)) AS kv
WHERE {_diagnostics_filter()}
  AND kv.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
ORDER BY 1"""


def _action_pipeline_table_sql() -> str:
    return f"""SELECT e.ts, e.actor_id, e.exchange, e.symbol, stage,
       (e.payload #>> ARRAY['action_pipeline', stage, 'order_count'])::double precision AS order_count,
       (e.payload #>> ARRAY['action_pipeline', stage, 'buy_qty'])::double precision AS buy_qty,
       (e.payload #>> ARRAY['action_pipeline', stage, 'sell_qty'])::double precision AS sell_qty,
       e.payload #> ARRAY['action_pipeline', stage, 'rejection_counts'] AS rejection_counts
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['ladder','venue_quantized','capped','target_book','risk_approved_target','planned_cancel','planned_place','rules_valid','router_accepted','kept']) AS stage
WHERE {_diagnostics_filter()}
ORDER BY e.ts DESC, stage
LIMIT 300"""


def _recent_execution_events_sql(limit: int = 50) -> str:
    return f"""SELECT ts, topic, event_type, exchange, symbol, actor_id, payload
FROM mm_events
WHERE topic LIKE 'execution:%'
  AND exchange IN (${{exchange:sqlstring}})
  AND (symbol IN (${{symbol:sqlstring}}) OR symbol IS NULL)
  AND $__timeFilter(ts)
ORDER BY ts DESC
LIMIT {limit}"""


def _portfolio_field_sql(*fields: str) -> str:
    selects = []
    for field in fields:
        selects.append(
            f"""SELECT e.ts AS "time", '{field}' AS metric,
       (e.payload #>> ARRAY['state','{field}'])::double precision AS value
FROM mm_events e
WHERE {_diagnostics_filter()}
  AND e.payload #>> ARRAY['state','{field}'] IS NOT NULL"""
        )
    return "\nUNION ALL\n".join(selects) + "\nORDER BY 1"


def _step_economics_sql() -> str:
    return """SELECT ts AS "time", metric, value
FROM step_economics
CROSS JOIN LATERAL (
  VALUES
    ('spread_pnl', spread_pnl),
    ('inventory_pnl', inventory_pnl),
    ('cross_pnl', cross_pnl),
    ('fee_pnl', fee_pnl),
    ('wealth_pnl', wealth_pnl)
) AS v(metric, value)
WHERE exchange IN (${exchange:sqlstring})
  AND symbol IN (${symbol:sqlstring})
  AND $__timeFilter(ts)
ORDER BY 1"""


def _venue_gates_sql() -> str:
    return f"""SELECT e.ts, gates.key AS exchange, gates.value->>'role' AS role,
       gates.value->>'live' AS live,
       gates.value->>'has_orders' AS has_orders,
       gates.value->'blocked_reasons' AS blocked_reasons
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(COALESCE(e.payload #> ARRAY['execution_gates','venue_gates'], '{{}}'::jsonb)) AS gates
WHERE {_diagnostics_filter()}
ORDER BY e.ts DESC
LIMIT 100"""


def _target_book_count_sql() -> str:
    return """SELECT ts AS "time", venue AS metric, jsonb_array_length(COALESCE(book->'orders', '[]'::jsonb))::double precision AS value
FROM mm_events
CROSS JOIN LATERAL (
  VALUES ('local', payload->'local'), ('global', payload->'global')
) AS v(venue, book)
WHERE topic = '${target_books_topic}'
  AND event_type = 'JointTargetBook'
  AND exchange IN (${exchange:sqlstring})
  AND symbol IN (${symbol:sqlstring})
  AND book IS NOT NULL
  AND $__timeFilter(ts)
ORDER BY 1"""


def _target_book_qty_sql() -> str:
    return """SELECT e.ts AS "time", (venue || ' ' || (o->>'side')) AS metric,
       SUM((o->>'qty')::double precision) AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES ('local', e.payload->'local'), ('global', e.payload->'global')
) AS v(venue, book)
CROSS JOIN LATERAL jsonb_array_elements(COALESCE(book->'orders', '[]'::jsonb)) AS o
WHERE e.topic = '${target_books_topic}'
  AND e.event_type = 'JointTargetBook'
  AND e.exchange IN (${exchange:sqlstring})
  AND e.symbol IN (${symbol:sqlstring})
  AND $__timeFilter(e.ts)
GROUP BY 1, venue, (o->>'side')
ORDER BY 1"""


def _latest_payload_sql(kind: str) -> str:
    if kind == "diagnostics":
        where = f"{_diagnostics_filter()}"
    elif kind == "decision":
        where = """event_type = 'StrategyDecision'
  AND exchange IN (${exchange:sqlstring})
  AND symbol IN (${symbol:sqlstring})
  AND actor_id IN (${actor_id:sqlstring})
  AND $__timeFilter(ts)"""
    else:
        where = """topic = '${target_books_topic}'
  AND event_type = 'JointTargetBook'
  AND exchange IN (${exchange:sqlstring})
  AND symbol IN (${symbol:sqlstring})
  AND actor_id IN (${actor_id:sqlstring})
  AND $__timeFilter(ts)"""
    alias = "e" if kind == "diagnostics" else ""
    if alias:
        return f"SELECT e.ts, e.topic, e.event_type, e.payload FROM mm_events e WHERE {where} ORDER BY e.ts DESC LIMIT 1"
    return f"SELECT ts, topic, event_type, payload FROM mm_events WHERE {where} ORDER BY ts DESC LIMIT 1"


def dashboards() -> list[dict[str, Any]]:
    return [
        runtime_overview(),
        market_data_quality(),
        controller_health(),
        model_quality(),
        optimizer_diagnostics(),
        action_pipeline(),
        risk_execution(),
        portfolio_pnl(),
        liquidity_sourcing(),
        runtime_logs(),
        incident_drilldown(),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for dashboard in dashboards():
        path = OUT / f"{dashboard['uid']}.json"
        path.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
