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
    decimals: int = 3,
    legend: str | None = None,
) -> dict[str, Any]:
    panel: dict[str, Any] = {
        "datasource": datasource,
        "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals}, "overrides": []},
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


def _xy_depth_panel(id_: int, title: str, expr: str, *, y: int) -> dict[str, Any]:
    return {
        "datasource": PG_DS,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": True,
                    "axisColorMode": "text",
                    "axisLabel": "qty",
                    "axisPlacement": "auto",
                    "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                    "lineInterpolation": "linear",
                    "lineStyle": {"fill": "solid"},
                    "lineWidth": 2,
                    "pointSize": {"fixed": 4, "max": 20, "min": 1},
                    "scaleDistribution": {"type": "linear"},
                    "show": "lines",
                    "spanNulls": False,
                },
                "unit": "short",
                "decimals": 3,
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": "spread"},
                    "properties": [
                        {"id": "custom.axisLabel", "value": "price - mid"},
                        {"id": "custom.axisCenteredZero", "value": True},
                        {"id": "decimals", "value": 1},
                    ],
                },
                {
                    "matcher": {"id": "byName", "options": "bid"},
                    "properties": [
                        {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}},
                        {"id": "displayName", "value": "Bid"},
                    ],
                },
                {
                    "matcher": {"id": "byName", "options": "ask"},
                    "properties": [
                        {"id": "color", "value": {"fixedColor": "red", "mode": "fixed"}},
                        {"id": "displayName", "value": "Ask"},
                    ],
                },
            ],
        },
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": y},
        "id": id_,
        "options": {
            "legend": {"calcs": [], "displayMode": "list", "placement": "bottom", "showLegend": True},
            "mapping": "auto",
            "series": [],
            "tooltip": {"mode": "multi", "sort": "none"},
        },
        "targets": [_target(PG_DS, expr, fmt="table")],
        "title": title,
        "type": "xychart",
        "pluginVersion": "13.0.0",
        "transformations": [
            {
                "id": "partitionByValues",
                "options": {
                    "fields": ["side"],
                    "keepFields": False,
                    "naming": {"mode": "as_is", "asLabels": False},
                },
            }
        ],
    }


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


def _dashboard(
    uid: str,
    title: str,
    panels: list[dict[str, Any]],
    *,
    tags: list[str],
    extra_templating: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
        "templating": {"list": [*_templating(), *(extra_templating or [])]},
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
        _text(1, "Scope", "Raw orderbook/trade feed health before interval aggregation, plus local/global price-volume comparison.", y=0),
        _panel(2, "Raw feed rate / sec", "timeseries", PROM_DS, 'sum by (exchange,symbol,kind)(rate(mm_raw_market_events_total{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"}[$__rate_interval]))', x=0, y=3, w=6, unit="ops", legend="{{exchange}} {{kind}}"),
        _panel(3, "Raw feed freshness ms", "timeseries", PROM_DS, '(time() * 1000) - max by (exchange,symbol,kind)(mm_raw_market_event_timestamp_ms{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"})', x=6, y=3, w=6, unit="ms", legend="{{exchange}} {{kind}}"),
        _panel(4, "Raw receive lag latest ms", "timeseries", PROM_DS, 'max by (exchange,symbol,kind)(mm_raw_market_ingest_lag_latest_ms{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"})', x=12, y=3, w=6, unit="ms", legend="{{exchange}} {{kind}}"),
        _panel(5, "Raw event gap latest ms", "timeseries", PROM_DS, 'max by (exchange,symbol,kind)(mm_raw_market_event_gap_latest_ms{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"})', x=18, y=3, w=6, unit="ms", legend="{{exchange}} {{kind}}"),
        _panel(6, "Local best bid / ask / VWAP", "timeseries", PG_DS, _role_price_vwap_sql("local"), x=0, y=11),
        _panel(7, "Global best bid / ask / VWAP", "timeseries", PG_DS, _role_price_vwap_sql("global"), x=12, y=11),
        _panel(8, "Local buy / sell volume", "timeseries", PG_DS, _role_volume_sql("local"), x=0, y=19),
        _panel(9, "Global buy / sell volume", "timeseries", PG_DS, _role_volume_sql("global"), x=12, y=19),
    ]
    return _dashboard("mm-obs-v2-market-data", "MM Observability v2 / 01 Market Data Quality", panels, tags=["market-data"])


def orderbook_depth() -> dict[str, Any]:
    panels = [
        _text(
            1,
            "Scope",
            "Depth curves use `price - mid` on the x-axis, so zero is the selected frame mid. Bid and ask depth are shown together on each chart.",
            y=0,
        ),
        _xy_depth_panel(2, "Orderbook depth", _orderbook_depth_sql("raw"), y=3),
        _xy_depth_panel(3, "Clean orderbook depth", _orderbook_depth_sql("clean"), y=11),
        _xy_depth_panel(4, "Action ladder depth", _orderbook_depth_sql("ladder"), y=19),
    ]
    return _dashboard(
        "mm-obs-v2-orderbook-depth",
        "MM Observability v2 / 01B Orderbook Depth",
        panels,
        tags=["orderbook"],
        extra_templating=[_custom_var("spread_max", "Spread max", "2000,5000,10000,20000,500,200,100")],
    )


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
        _text(1, "Scope", "Model NLL and latest model scores emitted from `ControllerDiagnostics.models`.", y=0),
        _panel(2, "Model NLL p95", "timeseries", PROM_DS, 'histogram_quantile(0.95, sum by (le,model)(rate(mm_model_nll_bucket{actor_id=~"${actor_id:regex}"}[$__rate_interval])))', x=0, y=3, w=24, legend="{{model}}"),
        _panel(3, "Latest model scores", "table", PG_DS, _latest_model_scores_sql(), x=0, y=11, w=24, fmt="table"),
    ]
    return _dashboard("mm-obs-v2-model-quality", "MM Observability v2 / 03 Strategy Model Quality", panels, tags=["models"])


def optimizer_diagnostics() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Optimizer objective, contribution fractions, timing, and gradient health.", y=0),
        _panel(2, "Objective value", "timeseries", PG_DS, _optimizer_objective_sql(), x=0, y=3, w=24, unit="sci", decimals=4),
        _panel(3, "Contribution fractions", "timeseries", PG_DS, _optimizer_contrib_sql(), x=0, y=11),
        _panel(4, "Optimizer timings", "timeseries", PG_DS, _optimizer_timing_sql(), x=12, y=11),
        _panel(5, "Nonfinite counts", "timeseries", PG_DS, _optimizer_gradients_sql(), x=0, y=19, w=24),
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
        _text(1, "Scope", "Inventory, cash, total marked value, wealth, and step economics.", y=0),
        _panel(2, "Inventory", "timeseries", PG_DS, _portfolio_field_sql("local_inventory", "global_inventory"), x=0, y=3),
        _panel(3, "Cash", "timeseries", PG_DS, _portfolio_field_sql("local_cash", "global_cash"), x=12, y=3),
        _panel(4, "Wealth", "timeseries", PG_DS, _portfolio_field_sql("local_wealth", "global_wealth"), x=0, y=11),
        _panel(5, "Total cash / inventory value / wealth", "timeseries", PG_DS, _portfolio_total_value_sql(), x=12, y=11),
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


def _orderbook_depth_sql(kind: str) -> str:
    if kind == "raw":
        selected = """selected AS (
  SELECT COALESCE(
           NULLIF(ctx.mid, 0.0),
           (
             NULLIF(m.payload #>> ARRAY['orderbook','bids','0','price'], '')::double precision +
             NULLIF(m.payload #>> ARRAY['orderbook','asks','0','price'], '')::double precision
           ) / 2.0,
           0.0
         ) AS mid,
         ctx.spread_max,
         m.payload
  FROM ctx
  LEFT JOIN LATERAL (
    SELECT m.payload, m.ts
    FROM mm_events m
    WHERE m.event_type = 'MarketEvent'
      AND m.exchange = ctx.exchange
      AND m.symbol = ctx.symbol
      AND m.payload->>'t_end_ms' IS NOT NULL
      AND (m.payload->>'t_end_ms')::bigint <= ctx.t_end_ms
      AND $__timeFilter(m.ts)
    ORDER BY (m.payload->>'t_end_ms')::bigint DESC, m.ts DESC
    LIMIT 1
  ) m ON TRUE
)"""
        bid_levels = """bid_levels AS (
  SELECT (b->>'price')::double precision AS price,
         COALESCE((b->>'qty')::double precision, 0.0) AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'orderbook'->'bids', '[]'::jsonb)) AS b
)"""
        ask_levels = """ask_levels AS (
  SELECT (a->>'price')::double precision AS price,
         COALESCE((a->>'qty')::double precision, 0.0) AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'orderbook'->'asks', '[]'::jsonb)) AS a
)"""
    elif kind == "clean":
        selected = """selected AS (
  SELECT ctx.mid, ctx.spread_max, ctx.diag_payload AS payload
  FROM ctx
)"""
        bid_levels = """bid_levels AS (
  SELECT (b->>'price')::double precision AS price,
         COALESCE((b->>'qty')::double precision, 0.0) AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'cleared_orderbook'->'bids', '[]'::jsonb)) AS b
)"""
        ask_levels = """ask_levels AS (
  SELECT (a->>'price')::double precision AS price,
         COALESCE((a->>'qty')::double precision, 0.0) AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'cleared_orderbook'->'asks', '[]'::jsonb)) AS a
)"""
    elif kind == "ladder":
        selected = """selected AS (
  SELECT COALESCE(
           NULLIF(ctx.mid, 0.0),
           NULLIF(d.payload->>'mid', '')::double precision,
           0.0
         ) AS mid,
         ctx.spread_max,
         d.payload
  FROM ctx
  LEFT JOIN LATERAL (
    SELECT d.payload, d.ts
    FROM mm_events d
    WHERE d.event_type = 'StrategyDecision'
      AND d.exchange = ctx.exchange
      AND d.symbol = ctx.symbol
      AND COALESCE(d.actor_id, d.payload->>'actor_id', '') = ctx.actor_id
      AND d.payload->>'t_end_ms' IS NOT NULL
      AND (d.payload->>'t_end_ms')::bigint <= ctx.t_end_ms
      AND $__timeFilter(d.ts)
    ORDER BY (d.payload->>'t_end_ms')::bigint DESC, d.ts DESC
    LIMIT 1
  ) d ON TRUE
)"""
        bid_levels = """bid_levels AS (
  SELECT (o->>'price')::double precision AS price,
         COALESCE((o->>'qty')::double precision, 0.0) AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'ladder_orders', '[]'::jsonb)) AS o
  WHERE lower(trim(o->>'side')) = 'buy'
)"""
        ask_levels = """ask_levels AS (
  SELECT (o->>'price')::double precision AS price,
         COALESCE((o->>'qty')::double precision, 0.0) AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'ladder_orders', '[]'::jsonb)) AS o
  WHERE lower(trim(o->>'side')) = 'sell'
)"""
    else:
        raise ValueError(f"unknown orderbook depth kind: {kind}")
    return f"""WITH {_orderbook_context_cte()},
{selected},
{bid_levels},
{ask_levels},
{_orderbook_depth_plot_ctes()}
SELECT 'bid'::text AS side, spread, value FROM bid_step_plot
UNION ALL
SELECT 'ask'::text AS side, spread, value FROM ask_step_plot
ORDER BY side, spread"""


def _orderbook_context_cte() -> str:
    return f"""ctx AS (
  SELECT e.ts AS diag_ts,
         e.payload AS diag_payload,
         e.exchange,
         e.symbol,
         e.actor_id,
         COALESCE(NULLIF(e.payload->>'t_end_ms', '')::bigint, 0) AS t_end_ms,
         COALESCE(
           NULLIF(e.payload #>> ARRAY['state','mid'], '')::double precision,
           (
             NULLIF(e.payload #>> ARRAY['cleared_orderbook','bids','0','price'], '')::double precision +
             NULLIF(e.payload #>> ARRAY['cleared_orderbook','asks','0','price'], '')::double precision
           ) / 2.0,
           0.0
         ) AS mid,
         GREATEST(${{spread_max}}::double precision, 1.0) AS spread_max
  FROM mm_events e
  WHERE {_diagnostics_filter()}
  ORDER BY e.ts DESC
  LIMIT 1
)"""


def _orderbook_depth_plot_ctes() -> str:
    return """bid_step AS (
  SELECT (l.price - selected.mid) AS spread,
         SUM(l.qty) OVER (ORDER BY l.price DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS value
  FROM selected
  JOIN bid_levels l ON TRUE
  WHERE l.price IS NOT NULL
    AND l.qty > 0
    AND (l.price - selected.mid) >= -selected.spread_max
    AND (l.price - selected.mid) <= 0
),
ask_step AS (
  SELECT (l.price - selected.mid) AS spread,
         SUM(l.qty) OVER (ORDER BY l.price ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS value
  FROM selected
  JOIN ask_levels l ON TRUE
  WHERE l.price IS NOT NULL
    AND l.qty > 0
    AND (l.price - selected.mid) >= 0
    AND (l.price - selected.mid) <= selected.spread_max
),
bid_step_plot AS (
  SELECT spread, COALESCE(value, 0::double precision) AS value FROM bid_step
  UNION ALL
  SELECT -selected.spread_max,
         COALESCE((SELECT value FROM bid_step ORDER BY spread ASC LIMIT 1), 0::double precision)
  FROM selected
  UNION ALL
  SELECT 0::double precision, 0::double precision FROM selected
  UNION ALL
  SELECT selected.spread_max, 0::double precision FROM selected
),
ask_step_plot AS (
  SELECT spread, COALESCE(value, 0::double precision) AS value FROM ask_step
  UNION ALL
  SELECT -selected.spread_max, 0::double precision FROM selected
  UNION ALL
  SELECT 0::double precision, 0::double precision FROM selected
  UNION ALL
  SELECT selected.spread_max,
         COALESCE((SELECT value FROM ask_step ORDER BY spread DESC LIMIT 1), 0::double precision)
  FROM selected
)"""


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


def _role_exchange_cte() -> str:
    return """role_exchange AS (
  SELECT DISTINCT role, exchange, symbol
  FROM (
    SELECT 'local'::text AS role, e.exchange AS exchange, e.symbol AS symbol
    FROM mm_events e
    WHERE e.topic = '${diagnostics_topic}'
      AND e.event_type = 'ControllerDiagnostics'
      AND e.symbol IN (${symbol:sqlstring})
      AND e.actor_id IN (${actor_id:sqlstring})
      AND $__timeFilter(e.ts)
    UNION ALL
    SELECT gates.value->>'role' AS role, gates.key AS exchange, e.symbol AS symbol
    FROM mm_events e
    CROSS JOIN LATERAL jsonb_each(
      COALESCE(e.payload #> ARRAY['execution_gates','venue_gates'], '{}'::jsonb)
    ) AS gates(key, value)
    WHERE e.topic = '${diagnostics_topic}'
      AND e.event_type = 'ControllerDiagnostics'
      AND e.symbol IN (${symbol:sqlstring})
      AND e.actor_id IN (${actor_id:sqlstring})
      AND $__timeFilter(e.ts)
      AND gates.value->>'role' IN ('local', 'global')
    UNION ALL
    SELECT 'local'::text,
           jf.payload #>> ARRAY['local_market','exchange'],
           jf.payload #>> ARRAY['local_market','symbol']
    FROM mm_events jf
    WHERE jf.event_type = 'JointStrategyFrame'
      AND (jf.payload #>> ARRAY['local_market','symbol']) IN (${symbol:sqlstring})
      AND $__timeFilter(jf.ts)
    UNION ALL
    SELECT 'global'::text,
           jf.payload #>> ARRAY['global_market','exchange'],
           jf.payload #>> ARRAY['global_market','symbol']
    FROM mm_events jf
    WHERE jf.event_type = 'JointStrategyFrame'
      AND (jf.payload #>> ARRAY['global_market','exchange']) IS NOT NULL
      AND (jf.payload #>> ARRAY['global_market','symbol']) IN (${symbol:sqlstring})
      AND $__timeFilter(jf.ts)
  ) r
  WHERE role IN ('local', 'global')
    AND exchange IS NOT NULL AND exchange <> ''
    AND symbol IS NOT NULL AND symbol <> ''
)"""


def _role_price_vwap_sql(role: str) -> str:
    role_key = "global" if role == "global" else "local"
    if role_key == "local":
        vwap_join = "r.role = 'local' AND r.exchange = e.exchange AND r.symbol = e.symbol"
        buy_field = "local_buy_vwap"
        sell_field = "local_sell_vwap"
    else:
        vwap_join = "r.role = 'global' AND r.symbol = e.symbol"
        buy_field = "global_buy_vwap"
        sell_field = "global_sell_vwap"
    return f"""WITH {_role_exchange_cte()}
SELECT m.ts AS "time", (m.exchange || ' best_bid') AS metric,
       (m.payload->'orderbook'->'bids'->0->>'price')::double precision AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange IN (${{exchange:sqlstring}})
  AND m.symbol IN (${{symbol:sqlstring}})
  AND (m.payload->'orderbook'->'bids'->0->>'price') IS NOT NULL
  AND $__timeFilter(m.ts)
UNION ALL
SELECT m.ts AS "time", (m.exchange || ' best_ask') AS metric,
       (m.payload->'orderbook'->'asks'->0->>'price')::double precision AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange IN (${{exchange:sqlstring}})
  AND m.symbol IN (${{symbol:sqlstring}})
  AND (m.payload->'orderbook'->'asks'->0->>'price') IS NOT NULL
  AND $__timeFilter(m.ts)
UNION ALL
SELECT e.ts AS "time", (r.exchange || ' buy_vwap') AS metric,
       (e.payload #>> ARRAY['state','{buy_field}'])::double precision AS value
FROM mm_events e
JOIN role_exchange r ON {vwap_join}
WHERE e.topic = '${{diagnostics_topic}}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.symbol IN (${{symbol:sqlstring}})
  AND e.actor_id IN (${{actor_id:sqlstring}})
  AND r.exchange IN (${{exchange:sqlstring}})
  AND (e.payload #>> ARRAY['state','{buy_field}']) IS NOT NULL
  AND $__timeFilter(e.ts)
UNION ALL
SELECT e.ts AS "time", (r.exchange || ' sell_vwap') AS metric,
       (e.payload #>> ARRAY['state','{sell_field}'])::double precision AS value
FROM mm_events e
JOIN role_exchange r ON {vwap_join}
WHERE e.topic = '${{diagnostics_topic}}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.symbol IN (${{symbol:sqlstring}})
  AND e.actor_id IN (${{actor_id:sqlstring}})
  AND r.exchange IN (${{exchange:sqlstring}})
  AND (e.payload #>> ARRAY['state','{sell_field}']) IS NOT NULL
  AND $__timeFilter(e.ts)
ORDER BY 1"""


def _role_volume_sql(role: str) -> str:
    role_key = "global" if role == "global" else "local"
    return f"""WITH {_role_exchange_cte()}
SELECT m.ts AS "time", (m.exchange || ' buy_volume') AS metric,
       (m.payload->>'buy_volume')::double precision AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange IN (${{exchange:sqlstring}})
  AND m.symbol IN (${{symbol:sqlstring}})
  AND (m.payload->>'buy_volume') IS NOT NULL
  AND $__timeFilter(m.ts)
UNION ALL
SELECT m.ts AS "time", (m.exchange || ' sell_volume') AS metric,
       (m.payload->>'sell_volume')::double precision AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange IN (${{exchange:sqlstring}})
  AND m.symbol IN (${{symbol:sqlstring}})
  AND (m.payload->>'sell_volume') IS NOT NULL
  AND $__timeFilter(m.ts)
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


def _portfolio_total_value_sql() -> str:
    return f"""WITH state_rows AS (
  SELECT e.ts AS "time",
         COALESCE((e.payload #>> ARRAY['state','local_cash'])::double precision, 0.0) AS local_cash,
         COALESCE((e.payload #>> ARRAY['state','global_cash'])::double precision, 0.0) AS global_cash,
         COALESCE((e.payload #>> ARRAY['state','local_inventory'])::double precision, 0.0) AS local_inventory,
         COALESCE((e.payload #>> ARRAY['state','global_inventory'])::double precision, 0.0) AS global_inventory,
         COALESCE(
           NULLIF(e.payload #>> ARRAY['optimizer','config','reference_price'], '')::double precision,
           NULLIF(e.payload #>> ARRAY['state','global_sell_vwap'], '')::double precision,
           NULLIF(e.payload #>> ARRAY['state','local_sell_vwap'], '')::double precision,
           NULLIF(e.payload #>> ARRAY['state','mid'], '')::double precision,
           0.0
         ) AS reference_vwap
  FROM mm_events e
  WHERE {_diagnostics_filter()}
)
SELECT "time", 'total_cash' AS metric,
       (local_cash + global_cash) AS value
FROM state_rows
UNION ALL
SELECT "time", 'total_inventory_value',
       ((local_inventory + global_inventory) * reference_vwap) AS value
FROM state_rows
UNION ALL
SELECT "time", 'total_wealth',
       (local_cash + global_cash + (local_inventory + global_inventory) * reference_vwap) AS value
FROM state_rows
ORDER BY 1"""


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
        orderbook_depth(),
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
