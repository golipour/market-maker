#!/usr/bin/env python3
"""Generate the Observability v2 Grafana dashboards.

The existing dashboards stay in the ``Market Maker`` folder. These dashboards
are provisioned by a separate provider into ``Market Maker - Observability v2``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mm.tools.grafana_dashboard_variables import (
    MM_SCOPE_EVENT_TYPES,
    actor_id_var_query,
    diagnostics_topic_var_query,
    exchange_var_query,
    interval_ms_var_query,
    run_id_var_query,
    symbol_var_query,
    target_books_topic_var_query,
)

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
    description: str | None = None,
) -> dict[str, Any]:
    field_defaults: dict[str, Any] = {"unit": unit, "decimals": decimals}
    if panel_type == "stat":
        field_defaults["thresholds"] = {
            "mode": "absolute",
            "steps": [
                {"color": "red", "value": None},
                {"color": "gray", "value": 0},
                {"color": "green", "value": 1e-12},
            ],
        }
    panel: dict[str, Any] = {
        "datasource": datasource,
        "fieldConfig": {"defaults": field_defaults, "overrides": []},
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": id_,
        "targets": [_target(datasource, expr, fmt=fmt, legend=legend)],
        "title": title,
        "type": panel_type,
    }
    if description:
        panel["description"] = description
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


def _xy_depth_panel(
    id_: int,
    title: str,
    expr: str,
    *,
    y: int,
    x: int = 0,
    w: int = 24,
) -> dict[str, Any]:
    return {
        "datasource": PG_DS,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
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
                    "matcher": {"id": "byRegexp", "options": "^(value|bid|ask|Bid|Ask|value bid|value ask)$"},
                    "properties": [
                        {"id": "custom.axisSoftMin", "value": 0},
                        {"id": "min", "value": 0},
                    ],
                },
                {
                    "matcher": {"id": "byName", "options": "bid"},
                    "properties": [
                        {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}},
                        {"id": "custom.axisSoftMin", "value": 0},
                        {"id": "displayName", "value": "Bid"},
                        {"id": "min", "value": 0},
                    ],
                },
                {
                    "matcher": {"id": "byName", "options": "ask"},
                    "properties": [
                        {"id": "color", "value": {"fixedColor": "red", "mode": "fixed"}},
                        {"id": "custom.axisSoftMin", "value": 0},
                        {"id": "displayName", "value": "Ask"},
                        {"id": "min", "value": 0},
                    ],
                },
            ],
        },
        "gridPos": {"h": 8, "w": w, "x": x, "y": y},
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


def _xy_order_qty_price_panel(id_: int, title: str, expr: str, *, y: int) -> dict[str, Any]:
    return {
        "datasource": PG_DS,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
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
                    "matcher": {"id": "byName", "options": "price"},
                    "properties": [
                        {"id": "custom.axisLabel", "value": "price"},
                        {"id": "decimals", "value": 6},
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
                {
                    "matcher": {"id": "byName", "options": "placed orders"},
                    "properties": [
                        {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}},
                        {"id": "displayName", "value": "Placed orders"},
                    ],
                },
                {
                    "matcher": {"id": "byName", "options": "canceled order ids"},
                    "properties": [
                        {"id": "color", "value": {"fixedColor": "red", "mode": "fixed"}},
                        {"id": "displayName", "value": "Canceled order ids"},
                    ],
                },
            ],
        },
        "gridPos": {"h": 10, "w": 24, "x": 0, "y": y},
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
                    "fields": ["series"],
                    "keepFields": False,
                    "naming": {"mode": "as_is", "asLabels": False},
                },
            }
        ],
    }


def _stem_order_qty_price_panel(
    id_: int,
    title: str,
    expr: str,
    *,
    x: int,
    y: int,
    w: int = 8,
    h: int = 10,
    description: str | None = None,
) -> dict[str, Any]:
    panel: dict[str, Any] = {
        "datasource": PG_DS,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "qty",
                    "axisPlacement": "auto",
                    "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                    "lineWidth": 0,
                    "pointSize": {"fixed": 6, "max": 20, "min": 1},
                    "scaleDistribution": {"type": "linear"},
                    "show": "points",
                    "spanNulls": False,
                },
                "unit": "short",
                "decimals": 3,
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": "price"},
                    "properties": [
                        {"id": "custom.axisLabel", "value": "price"},
                        {"id": "decimals", "value": 6},
                    ],
                },
                {
                    "matcher": {
                        "id": "byRegexp",
                        "options": ".*bid$",
                    },
                    "properties": [
                        {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}},
                        {"id": "custom.axisSoftMin", "value": 0},
                        {"id": "min", "value": 0},
                    ],
                },
                {
                    "matcher": {
                        "id": "byRegexp",
                        "options": ".*ask$",
                    },
                    "properties": [
                        {"id": "color", "value": {"fixedColor": "red", "mode": "fixed"}},
                        {"id": "custom.axisSoftMin", "value": 0},
                        {"id": "min", "value": 0},
                    ],
                },
                {
                    "matcher": {"id": "byRegexp", "options": "^axis range$"},
                    "properties": [
                        {
                            "id": "custom.hideFrom",
                            "value": {"legend": True, "tooltip": True, "viz": False},
                        },
                        {"id": "color", "value": {"fixedColor": "black", "mode": "fixed"}},
                    ],
                },
            ],
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": id_,
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
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
                    "fields": ["series"],
                    "keepFields": False,
                    "naming": {"mode": "as_is", "asLabels": False},
                },
            }
        ],
    }
    if description:
        panel["description"] = description
    return panel


def _execution_command_panel(id_: int, title: str, expr: str, *, y: int) -> dict[str, Any]:
    panel = _panel(id_, title, "timeseries", PG_DS, expr, x=0, y=y, w=24)
    panel["fieldConfig"]["overrides"] = [
        {
            "matcher": {"id": "byName", "options": "placed orders"},
            "properties": [
                {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}},
                {"id": "displayName", "value": "Placed orders"},
            ],
        },
        {
            "matcher": {"id": "byName", "options": "canceled order ids"},
            "properties": [
                {"id": "color", "value": {"fixedColor": "red", "mode": "fixed"}},
                {"id": "displayName", "value": "Canceled order ids"},
            ],
        },
    ]
    return panel


def _text(id_: int, title: str, content: str, *, y: int, h: int = 3) -> dict[str, Any]:
    return {
        "gridPos": {"h": h, "w": 24, "x": 0, "y": y},
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
        _query_var(
            "exchange",
            "Exchange",
            exchange_var_query(),
            multi=True,
        ),
        _query_var(
            "symbol",
            "Symbol",
            symbol_var_query(),
            multi=True,
        ),
        _query_var(
            "actor_id",
            "Actor",
            actor_id_var_query(),
            multi=True,
        ),
        _query_var("run_id", "Run", run_id_var_query(), multi=True),
        _query_var("diagnostics_topic", "Diagnostics topic", diagnostics_topic_var_query()),
        _query_var("target_books_topic", "Target books topic", target_books_topic_var_query()),
        _query_var(
            "interval_ms",
            "Interval ms",
            interval_ms_var_query(),
        ),
    ]


def _query_var(name: str, label: str, query: str, *, multi: bool = False, refresh: int = 1) -> dict[str, Any]:
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
        "refresh": refresh,
        "sort": 0,
        "type": "query",
    }


def _target_stems_frame_query() -> str:
    return """SELECT
  e.event_id AS __value,
  (
    'step ' || COALESCE(e.payload->>'frame_step', '?')
    || ' / t_end ' || COALESCE(e.payload->>'t_end_ms', '?')
    || ' / created ' || COALESCE(e.payload->>'created_at_ms', '?')
  ) AS __text
FROM mm_events e
WHERE e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.exchange = ANY(ARRAY[${exchange:sqlstring}]::text[])
  AND e.symbol = ANY(ARRAY[${symbol:sqlstring}]::text[])
  AND e.actor_id = ANY(ARRAY[${actor_id:sqlstring}]::text[])
  AND $__timeFilter(e.ts)
ORDER BY e.ts DESC
LIMIT 100"""


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


def _custom_var_multi(
    name: str,
    label: str,
    query: str,
    *,
    default: str | None = None,
) -> dict[str, Any]:
    items = [item.strip() for item in query.split(",") if item.strip()]
    selected = [item.strip() for item in (default or query).split(",") if item.strip()]
    all_selected = len(selected) == len(items)
    return {
        "current": {
            "selected": True,
            "text": ["All"] if all_selected else selected,
            "value": ["$__all"] if all_selected else selected,
        },
        "hide": 0,
        "includeAll": True,
        "label": label,
        "multi": True,
        "name": name,
        "options": [{"selected": all_selected or item in selected, "text": item, "value": item} for item in items],
        "query": ",".join(items),
        "type": "custom",
    }


def _time_bucket_sql(expr: str) -> str:
    seconds = _rate_denominator_sql()
    return f"(floor(EXTRACT(EPOCH FROM ({expr})::timestamptz) / {seconds}) * {seconds})"


def _time_bucket_alias_sql(expr: str) -> str:
    return f"{_time_bucket_sql(expr)} AS \"time\""


def _numeric_sql(expr: str, default: str = "0.0") -> str:
    return f"""CASE
  WHEN trim(COALESCE({expr}, '')) ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$'
  THEN trim(COALESCE({expr}, ''))::double precision
  ELSE {default}
END"""


def _jsonb_array_or_empty_sql(expr: str) -> str:
    return f"""CASE
  WHEN jsonb_typeof({expr}) = 'array' THEN {expr}
  ELSE '[]'::jsonb
END"""


def _action_pipeline_orders_array_sql(stage: str, *, field: str = "orders") -> str:
    """Prefer full stage orders; fall back to persisted samples for legacy rows."""

    orders_expr = f"l.payload #> ARRAY['action_pipeline', '{stage}', '{field}']"
    sample_expr = f"l.payload #> ARRAY['action_pipeline', '{stage}', 'orders_sample']"
    return f"""CASE
  WHEN jsonb_typeof({orders_expr}) = 'array' AND jsonb_array_length({orders_expr}) > 0 THEN {orders_expr}
  WHEN jsonb_typeof({sample_expr}) = 'array' AND jsonb_array_length({sample_expr}) > 0 THEN {sample_expr}
  ELSE '[]'::jsonb
END"""


def _jsonb_object_or_empty_sql(expr: str) -> str:
    return f"""CASE
  WHEN jsonb_typeof({expr}) = 'object' THEN {expr}
  ELSE '{{}}'::jsonb
END"""


def _step_time_sql(t_end_expr: str) -> str:
    return f"to_timestamp(({_numeric_sql(t_end_expr, default='NULL')}) / 1000.0)"


def _step_interval_seconds_sql() -> str:
    return "GREATEST(COALESCE(NULLIF('${interval_ms}', '')::double precision, 1000.0) / 1000.0, 0.001)"


def _approx_step_time_sql(expr: str) -> str:
    seconds = _step_interval_seconds_sql()
    return f"to_timestamp(floor(EXTRACT(EPOCH FROM ({expr})::timestamptz) / {seconds}) * {seconds})"


def _approx_step_time_alias_sql(expr: str) -> str:
    return f"{_approx_step_time_sql(expr)} AS \"time\""


def _event_step_source_sql(t_end_expr: str, fallback_ts_expr: str) -> str:
    return f"COALESCE({_step_time_sql(t_end_expr)}, {fallback_ts_expr})"


def _market_event_scope_pairs_sql() -> str:
    return """SELECT DISTINCT e.exchange, e.symbol
  FROM mm_events e
  WHERE e.event_type = 'MarketEvent'
    AND e.exchange IS NOT NULL
    AND e.symbol IS NOT NULL
    AND e.exchange = ANY(ARRAY[${exchange:sqlstring}]::text[])
    AND e.symbol = ANY(ARRAY[${symbol:sqlstring}]::text[])"""


def _market_event_scope_pair_metric_sql() -> str:
    return f"""  SELECT scope.exchange || ' ' || scope.symbol AS metric
  FROM (
    {_market_event_scope_pairs_sql()}
  ) AS scope"""


def _mm_scope_pairs_sql() -> str:
    return f"""SELECT DISTINCT e.exchange, e.symbol
  FROM mm_events e
  WHERE e.event_type IN ({MM_SCOPE_EVENT_TYPES})
    AND e.exchange IS NOT NULL
    AND e.symbol IS NOT NULL
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])"""


def _mm_scope_pair_metric_sql() -> str:
    return f"""  SELECT scope.exchange || ' ' || scope.symbol AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope"""


def _mm_scope_exchange_metric_sql() -> str:
    return f"""  SELECT DISTINCT scope.exchange AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope"""


def _step_buckets_cte_sql() -> str:
    seconds = _step_interval_seconds_sql()
    return f"""buckets AS (
  SELECT generate_series(
    {_approx_step_time_sql("$__timeFrom()")},
    {_approx_step_time_sql("$__timeTo()")},
    make_interval(secs => {seconds})
  ) AS "time"
)"""


def _zero_filled_step_sql(data_sql: str, metrics_sql: str) -> str:
    return f"""WITH {_step_buckets_cte_sql()},
metrics AS (
{metrics_sql}
),
data AS (
{data_sql}
)
SELECT b."time", m.metric, COALESCE(d.value, 0.0) AS value
FROM buckets b
CROSS JOIN metrics m
LEFT JOIN data d
  ON d."time" = b."time"
 AND d.metric = m.metric
ORDER BY 1, 2"""


def _rate_denominator_sql() -> str:
    return """GREATEST(EXTRACT(EPOCH FROM '${__interval}'::interval), 1.0)"""


def _side_sql(expr: str) -> str:
    return f"""CASE lower(trim(COALESCE({expr}, '')))
  WHEN 'buy' THEN 'bid'
  WHEN 'bid' THEN 'bid'
  WHEN 'sell' THEN 'ask'
  WHEN 'ask' THEN 'ask'
  ELSE 'unknown'
END"""


def _runtime_frame_rate_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  ((e.payload #>> ARRAY['local_market','exchange']) || ' ' || (e.payload #>> ARRAY['local_market','symbol'])) AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE e.event_type = 'JointStrategyFrame'
  AND (e.payload #>> ARRAY['local_market','exchange']) = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.payload #>> ARRAY['local_market','symbol']) = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    metrics_sql = _mm_scope_pair_metric_sql()
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _runtime_decision_rate_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  (e.actor_id || ' ' || e.exchange || ' ' || e.symbol) AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE e.event_type = 'StrategyDecision'
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT actor.actor_id || ' ' || scope.exchange || ' ' || scope.symbol AS metric
  FROM unnest(ARRAY[${{actor_id:sqlstring}}]::text[]) AS actor(actor_id)
  CROSS JOIN (
    {_mm_scope_pairs_sql()}
  ) AS scope"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _runtime_controller_stage_p95_sql() -> str:
    numeric_value = _numeric_sql("kv.value", default="NULL")
    value_ms = f"""CASE
  WHEN kv.key LIKE '%_sec' THEN ({numeric_value}) * 1000.0
  ELSE ({numeric_value})
END"""
    stage = """CASE
  WHEN kv.key LIKE '%_sec' THEN substring(kv.key from 1 for length(kv.key) - 4)
  WHEN kv.key LIKE '%_ms' THEN substring(kv.key from 1 for length(kv.key) - 3)
  ELSE kv.key
END"""
    return f"""SELECT
  {_time_bucket_alias_sql("e.ts")},
  {stage} AS metric,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY {value_ms}) AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['optimizer','timing'], '{{}}'::jsonb)) AS kv(key, value)
WHERE {_diagnostics_filter()}
  AND kv.key <> 'best_epoch'
  AND trim(kv.value) ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$'
GROUP BY 1, 2
ORDER BY 1"""


def _runtime_order_sync_p95_sql() -> str:
    latency = _numeric_sql("e.payload->>'latency_ms'", default="NULL")
    return f"""SELECT
  {_time_bucket_alias_sql("e.ts")},
  COALESCE(e.exchange, 'unknown') AS metric,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY {latency}) AS value
FROM mm_events e
WHERE e.event_type IN ('ExecutionSyncCompleted', 'ExecutionSyncFailed')
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
  AND e.payload->>'latency_ms' IS NOT NULL
  AND trim(e.payload->>'latency_ms') ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$'
  AND $__timeFilter(e.ts)
GROUP BY 1, 2
ORDER BY 1"""


def _runtime_execution_event_rate_sql(event_type: str) -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  COALESCE(e.exchange, 'unknown') AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE e.event_type = '{event_type}'
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    metrics_sql = _mm_scope_exchange_metric_sql()
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _runtime_order_payload_rate_sql(event_type: str, measure: str, *, include_reason: bool = False) -> str:
    side = _side_sql("o.value->>'side'")
    qty = _numeric_sql("COALESCE(o.value->>'qty', o.value->>'remaining_qty')")
    value = "1.0" if measure == "count" else qty
    reason = " || ' ' || COALESCE(NULLIF(e.payload->>'error', ''), 'unknown')" if include_reason else ""
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  (COALESCE(o.value->>'exchange', e.exchange, 'unknown') || ' ' ||
   COALESCE(o.value->>'symbol', e.symbol, 'unknown') || ' ' ||
   {side}{reason}) AS metric,
  SUM({value}) AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
WHERE e.event_type = '{event_type}'
  AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    reason_suffix = " || ' unknown'" if include_reason else ""
    metrics_sql = f"""  SELECT scope.exchange || ' ' || scope.symbol || ' ' || side.side{reason_suffix} AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope
  CROSS JOIN (VALUES ('bid'), ('ask')) AS side(side)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _runtime_order_id_event_sql(event_type: str) -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  COALESCE(e.exchange, 'unknown') AS metric,
  SUM(jsonb_array_length(COALESCE(e.payload->'order_ids', '[]'::jsonb)))::double precision AS value
FROM mm_events e
WHERE e.event_type = '{event_type}'
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    metrics_sql = _mm_scope_exchange_metric_sql()
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _runtime_fill_rate_sql(measure: str) -> str:
    side = _side_sql("e.payload->>'side'")
    qty = _numeric_sql("e.payload->>'qty'")
    value = "1.0" if measure == "count" else qty
    exchange = "COALESCE(e.exchange, e.payload->>'exchange', substring(e.topic from '^private:state:([^:]+):fills$'), 'unknown')"
    symbol = "COALESCE(e.symbol, e.payload->>'symbol')"
    fill_ts = "COALESCE(to_timestamp((" + _numeric_sql("e.payload->>'timestamp_ms'", default="NULL") + ") / 1000.0), e.ts)"
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(fill_ts)},
  ({exchange} || ' ' || COALESCE({symbol}, 'unknown') || ' ' || {side}) AS metric,
  SUM({value}) AS value
FROM mm_events e
WHERE e.topic LIKE 'private:state:%:fills'
  AND {exchange} = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT scope.exchange || ' ' || scope.symbol || ' ' || side.side AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope
  CROSS JOIN (VALUES ('bid'), ('ask')) AS side(side)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _runtime_live_orders_sql(measure: str) -> str:
    side = _side_sql("open_order.value #>> ARRAY['order','side']")
    qty = _numeric_sql("open_order.value->>'remaining_qty'")
    value = "COUNT(*)::double precision" if measure == "count" else "SUM(qty)"
    bucket = _approx_step_time_sql("e.ts")
    return f"""WITH {_step_buckets_cte_sql()},
snapshots AS (
  SELECT
    {bucket} AS bucket,
    e.ts,
    COALESCE(e.exchange, 'unknown') AS exchange,
    e.payload
  FROM mm_events e
  WHERE e.topic LIKE 'execution:state:%'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
),
latest_per_bucket AS (
  SELECT DISTINCT ON (bucket, exchange)
    bucket,
    ts,
    exchange,
    payload
  FROM snapshots
  ORDER BY bucket, exchange, ts DESC
),
expanded AS (
  SELECT
    s.bucket,
    s.exchange,
    COALESCE(open_order.value #>> ARRAY['order','symbol'], 'unknown') AS symbol,
    {side} AS side,
    {qty} AS qty
  FROM latest_per_bucket s
  CROSS JOIN LATERAL jsonb_each(COALESCE(s.payload->'live_orders', '{{}}'::jsonb)) AS open_order(key, value)
  WHERE COALESCE(open_order.value #>> ARRAY['order','symbol'], '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
),
metrics AS (
  SELECT scope.exchange, scope.symbol, side.side
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope
  CROSS JOIN (VALUES ('bid'), ('ask')) AS side(side)
),
data AS (
  SELECT
    bucket AS "time",
    exchange,
    symbol,
    side,
    {value} AS value
  FROM expanded
  GROUP BY 1, 2, 3, 4
)
SELECT
  b."time",
  (m.exchange || ' ' || m.symbol || ' ' || m.side) AS metric,
  COALESCE(d.value, 0.0) AS value
FROM buckets b
CROSS JOIN metrics m
LEFT JOIN data d
  ON d."time" = b."time"
 AND d.exchange = m.exchange
 AND d.symbol = m.symbol
 AND d.side = m.side
ORDER BY 1"""


def _runtime_latest_order_counts_sql() -> str:
    return """WITH latest_execution AS (
  SELECT DISTINCT ON (e.exchange)
    'execution state'::text AS source,
    e.ts,
    e.exchange,
    e.payload
  FROM mm_events e
  WHERE e.topic LIKE 'execution:state:%'
    AND e.exchange = ANY(ARRAY[${exchange:sqlstring}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.exchange, e.ts DESC
),
latest_private AS (
  SELECT DISTINCT ON (COALESCE(e.exchange, substring(e.topic from '^private:state:([^:]+):snapshot$')))
    'private snapshot'::text AS source,
    e.ts,
    COALESCE(e.exchange, substring(e.topic from '^private:state:([^:]+):snapshot$'), 'unknown') AS exchange,
    e.payload
  FROM mm_events e
  WHERE e.topic LIKE 'private:state:%:snapshot'
    AND COALESCE(e.exchange, substring(e.topic from '^private:state:([^:]+):snapshot$'), 'unknown') = ANY(ARRAY[${exchange:sqlstring}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY COALESCE(e.exchange, substring(e.topic from '^private:state:([^:]+):snapshot$')), e.ts DESC
),
latest AS (
  SELECT * FROM latest_execution
  UNION ALL
  SELECT * FROM latest_private
)
SELECT
  l.source,
  l.ts,
  l.exchange,
  COUNT(open_order.key)::double precision AS open_order_count
FROM latest l
LEFT JOIN LATERAL jsonb_each(
  CASE l.source
    WHEN 'execution state' THEN COALESCE(l.payload->'live_orders', '{}'::jsonb)
    ELSE COALESCE(l.payload->'open_orders', '{}'::jsonb)
  END
) AS open_order(key, value)
  ON COALESCE(open_order.value #>> ARRAY['order','symbol'], '') = ANY(ARRAY[${symbol:sqlstring}]::text[])
GROUP BY l.source, l.ts, l.exchange
ORDER BY l.exchange, l.source"""


def _runtime_recent_fills_sql(limit: int = 50) -> str:
    exchange = "COALESCE(e.exchange, e.payload->>'exchange', substring(e.topic from '^private:state:([^:]+):fills$'), 'unknown')"
    qty = _numeric_sql("e.payload->>'qty'", default="NULL")
    price = _numeric_sql("e.payload->>'price'", default="NULL")
    return f"""SELECT
  e.ts,
  {exchange} AS exchange,
  COALESCE(e.symbol, e.payload->>'symbol', '') AS symbol,
  e.actor_id,
  e.payload->>'exchange_order_id' AS exchange_order_id,
  e.payload->>'side' AS side,
  {qty} AS qty,
  {price} AS price,
  e.payload->>'timestamp_ms' AS fill_timestamp_ms,
  e.payload
FROM mm_events e
WHERE e.topic LIKE 'private:state:%:fills'
  AND {exchange} = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND $__timeFilter(e.ts)
ORDER BY e.ts DESC
LIMIT {limit}"""


def _market_event_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  (e.exchange || ' ' || e.symbol) AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE e.event_type = 'MarketEvent'
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    metrics_sql = _market_event_scope_pair_metric_sql()
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _controller_stage_step_sql() -> str:
    numeric_value = _numeric_sql("kv.value", default="NULL")
    value_ms = f"""CASE
  WHEN kv.key LIKE '%_sec' THEN ({numeric_value}) * 1000.0
  ELSE ({numeric_value})
END"""
    stage = """CASE
  WHEN kv.key LIKE '%_sec' THEN substring(kv.key from 1 for length(kv.key) - 4)
  WHEN kv.key LIKE '%_ms' THEN substring(kv.key from 1 for length(kv.key) - 3)
  ELSE kv.key
END"""
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    return f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  {stage} AS metric,
  {value_ms} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['optimizer','timing'], '{{}}'::jsonb)) AS kv(key, value)
WHERE {_diagnostics_filter()}
  AND kv.key <> 'best_epoch'
  AND trim(kv.value) ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$'
ORDER BY 1"""


def _controller_warning_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  w.warning AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(e.payload->'warnings', '[]'::jsonb)) AS w(warning)
WHERE {_diagnostics_filter()}
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT 'warnings'::text AS metric
  UNION
  SELECT DISTINCT w.warning AS metric
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(e.payload->'warnings', '[]'::jsonb)) AS w(warning)
  WHERE {_diagnostics_filter()}"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _pipeline_profile_sql() -> str:
    market_orderbook_age = "GREATEST((m.t_end_ms - m.orderbook_timestamp_ms)::double precision, 0.0)"
    market_persist_lag = "GREATEST(EXTRACT(EPOCH FROM (m.ts - to_timestamp(m.t_end_ms / 1000.0))) * 1000.0, 0.0)"
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    controller_metric_prefix = "(e.actor_id || ' ' || e.exchange || ' ' || e.symbol || ' ')"
    market_metric_prefix = "(m.exchange || ' ' || m.symbol || ' ')"
    value_expr = _pipeline_value_ms_sql("s.raw_value", "s.unit")
    return f"""WITH market AS (
  SELECT
    {_approx_step_time_alias_sql("to_timestamp(m.t_end_ms / 1000.0)")},
    {market_metric_prefix} || s.metric AS metric,
    s.value AS value
  FROM mm_market_facts m
  CROSS JOIN LATERAL (
    VALUES
      ('market.orderbook_age_at_frame_close', {market_orderbook_age}),
      ('market.frame_persist_lag', {market_persist_lag})
  ) AS s(metric, value)
  WHERE m.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND m.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND m.orderbook_timestamp_ms IS NOT NULL
    AND $__timeFilter(m.ts)
),
controller AS (
  SELECT
    {_approx_step_time_alias_sql(step_source)},
    {controller_metric_prefix} || s.metric AS metric,
    {value_expr} AS value
  FROM mm_events e
  CROSS JOIN LATERAL (
    VALUES
      ('frame.age_at_strategy', e.payload #>> ARRAY['frame','frame_age_ms'], 'ms'),
      ('frame.global_lag', e.payload #>> ARRAY['frame','global_lag_ms'], 'ms'),
      ('strategy.clean', e.payload #>> ARRAY['optimizer','timing','clean_ms'], 'ms'),
      ('strategy.decide_total', e.payload #>> ARRAY['optimizer','timing','decide_ms'], 'ms'),
      ('strategy.optimizer_sample_draw', e.payload #>> ARRAY['optimizer','timing','sample_draw_sec'], 'sec'),
      ('strategy.optimizer_train_loop', e.payload #>> ARRAY['optimizer','timing','train_sec'], 'sec'),
      ('strategy.optimizer_total', e.payload #>> ARRAY['optimizer','timing','total_sec'], 'sec'),
      ('strategy.risk', e.payload #>> ARRAY['optimizer','timing','risk_ms'], 'ms'),
      ('strategy.publish', e.payload #>> ARRAY['optimizer','timing','publish_ms'], 'ms'),
      ('execution.cancel', e.payload #>> ARRAY['optimizer','timing','cancel_ms'], 'ms'),
      ('execution.place', e.payload #>> ARRAY['optimizer','timing','place_ms'], 'ms'),
      ('execution.sync', e.payload #>> ARRAY['optimizer','timing','execute_ms'], 'ms'),
      ('controller.total', e.payload #>> ARRAY['optimizer','timing','controller_total_ms'], 'ms')
  ) AS s(metric, raw_value, unit)
  WHERE {_diagnostics_filter()}
)
SELECT "time", metric, value
FROM market
WHERE value IS NOT NULL
UNION ALL
SELECT "time", metric, value
FROM controller
WHERE value IS NOT NULL
ORDER BY 1, 2"""


def _optimizer_substage_profile_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    metric_prefix = "(e.actor_id || ' ' || e.exchange || ' ' || e.symbol || ' ')"
    value_expr = _pipeline_value_ms_sql("s.raw_value", "s.unit")
    return f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  {metric_prefix} || s.metric AS metric,
  {value_expr} AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES
    ('sample_draw', e.payload #>> ARRAY['optimizer','timing','sample_draw_sec'], 'sec'),
    ('train_loop', e.payload #>> ARRAY['optimizer','timing','train_sec'], 'sec'),
    ('optimizer_total', e.payload #>> ARRAY['optimizer','timing','total_sec'], 'sec'),
    ('decision_wrapper', e.payload #>> ARRAY['optimizer','timing','decide_ms'], 'ms')
) AS s(metric, raw_value, unit)
WHERE {_diagnostics_filter()}
  AND {value_expr} IS NOT NULL
ORDER BY 1, 2"""


def _slowest_frames_sql() -> str:
    controller_total = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','controller_total_ms']", "'ms'")
    decide = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','decide_ms']", "'ms'")
    optimizer_total = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','total_sec']", "'sec'")
    train_loop = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','train_sec']", "'sec'")
    execute = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','execute_ms']", "'ms'")
    cancel = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','cancel_ms']", "'ms'")
    place = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','place_ms']", "'ms'")
    frame_age = _pipeline_value_ms_sql("e.payload #>> ARRAY['frame','frame_age_ms']", "'ms'")
    return f"""SELECT
  e.ts,
  e.actor_id,
  e.exchange,
  e.symbol,
  e.payload->>'frame_id' AS frame_id,
  e.payload->>'decision_id' AS decision_id,
  {controller_total} AS controller_total_ms,
  {decide} AS decide_ms,
  {optimizer_total} AS optimizer_total_ms,
  {train_loop} AS optimizer_train_loop_ms,
  {cancel} AS execution_cancel_ms,
  {place} AS execution_place_ms,
  {execute} AS execution_sync_ms,
  {frame_age} AS frame_age_ms,
  COALESCE(e.payload->'warnings', '[]'::jsonb) AS warnings
FROM mm_events e
WHERE {_diagnostics_filter()}
ORDER BY COALESCE({controller_total}, 0.0) DESC, e.ts DESC
LIMIT 50"""


def _pipeline_value_ms_sql(value_expr: str, unit_expr: str) -> str:
    numeric = _numeric_sql(value_expr, default="NULL")
    return f"""CASE
  WHEN {numeric} IS NULL THEN NULL
  WHEN lower(trim(COALESCE({unit_expr}, 'ms'))) = 'sec' THEN ({numeric}) * 1000.0
  ELSE ({numeric})
END"""


def _pipeline_frames_cte_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    t_end = _numeric_sql("e.payload->>'t_end_ms'")
    controller_start = _numeric_sql("e.payload->>'created_at_ms'")
    frame_created = _numeric_sql("e.payload #>> ARRAY['frame','frame_created_at_ms']")
    orderbook_ts = f"""COALESCE(
  {_numeric_sql("e.payload #>> ARRAY['frame','local_orderbook_timestamp_ms']")},
  m.orderbook_timestamp_ms
)"""
    clean = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','clean_ms']", "'ms'")
    decide = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','decide_ms']", "'ms'")
    risk = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','risk_ms']", "'ms'")
    publish = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','publish_ms']", "'ms'")
    cancel = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','cancel_ms']", "'ms'")
    place = _pipeline_value_ms_sql("e.payload #>> ARRAY['optimizer','timing','place_ms']", "'ms'")
    controller_total = _pipeline_value_ms_sql(
        "e.payload #>> ARRAY['optimizer','timing','controller_total_ms']",
        "'ms'",
    )
    metric_prefix = "(e.actor_id || ' ' || e.exchange || ' ' || e.symbol || ' ')"
    return f"""frames AS (
  SELECT
    {_approx_step_time_alias_sql(step_source)},
    {metric_prefix} AS metric_prefix,
    {t_end} AS t_end_ms,
    {orderbook_ts} AS orderbook_timestamp_ms,
    {frame_created} AS frame_created_at_ms,
    {controller_start} AS controller_start_ms,
    {controller_total} AS controller_total_ms,
    {clean} AS clean_ms,
    {decide} AS decide_ms,
    {risk} AS risk_ms,
    {publish} AS publish_ms,
    {cancel} AS cancel_ms,
    {place} AS place_ms
  FROM mm_events e
  LEFT JOIN mm_market_facts m
    ON m.exchange = e.exchange
   AND m.symbol = e.symbol
   AND m.t_end_ms = ({t_end})::bigint
  WHERE {_diagnostics_filter()}
)"""


def _pipeline_cumulative_milestones(
    *,
    anchor: str,
    strategy_start: str,
    stage_sum: str,
    pipeline_end: str,
) -> tuple[str, list[tuple[int, str, str]]]:
    if anchor == "orderbook":
        anchor_filter = "f.orderbook_timestamp_ms IS NOT NULL"
        anchor_ms = "f.orderbook_timestamp_ms"
        rows: list[tuple[int, str, str]] = [
            (1, "+orderbook.ref", "0.0"),
            (2, "+frame_close", f"f.t_end_ms - {anchor_ms}"),
            (3, "+frame_assembled", f"f.frame_created_at_ms - {anchor_ms}"),
            (4, "+strategy_start", f"{strategy_start} - {anchor_ms}"),
            (5, "+after_clean", f"{strategy_start} + COALESCE(f.clean_ms, 0) - {anchor_ms}"),
            (
                6,
                "+after_decide",
                f"{strategy_start} + COALESCE(f.clean_ms, 0) + COALESCE(f.decide_ms, 0) - {anchor_ms}",
            ),
            (
                7,
                "+after_risk",
                f"{strategy_start} + COALESCE(f.clean_ms, 0) + COALESCE(f.decide_ms, 0) + COALESCE(f.risk_ms, 0) - {anchor_ms}",
            ),
            (8, "+after_publish", f"{stage_sum} - {anchor_ms}"),
            (9, "+after_cancel", f"{stage_sum} + COALESCE(f.cancel_ms, 0) - {anchor_ms}"),
            (
                10,
                "+after_place",
                f"{stage_sum} + COALESCE(f.cancel_ms, 0) + COALESCE(f.place_ms, 0) - {anchor_ms}",
            ),
            (11, "+pipeline_end", f"{pipeline_end} - {anchor_ms}"),
        ]
    elif anchor == "t_end":
        anchor_filter = "f.t_end_ms IS NOT NULL"
        anchor_ms = "f.t_end_ms"
        rows = [
            (1, "+frame_close.ref", "0.0"),
            (2, "+frame_assembled", f"f.frame_created_at_ms - {anchor_ms}"),
            (3, "+strategy_start", f"{strategy_start} - {anchor_ms}"),
            (4, "+after_clean", f"{strategy_start} + COALESCE(f.clean_ms, 0) - {anchor_ms}"),
            (
                5,
                "+after_decide",
                f"{strategy_start} + COALESCE(f.clean_ms, 0) + COALESCE(f.decide_ms, 0) - {anchor_ms}",
            ),
            (
                6,
                "+after_risk",
                f"{strategy_start} + COALESCE(f.clean_ms, 0) + COALESCE(f.decide_ms, 0) + COALESCE(f.risk_ms, 0) - {anchor_ms}",
            ),
            (7, "+after_publish", f"{stage_sum} - {anchor_ms}"),
            (8, "+after_cancel", f"{stage_sum} + COALESCE(f.cancel_ms, 0) - {anchor_ms}"),
            (
                9,
                "+after_place",
                f"{stage_sum} + COALESCE(f.cancel_ms, 0) + COALESCE(f.place_ms, 0) - {anchor_ms}",
            ),
            (10, "+pipeline_end", f"{pipeline_end} - {anchor_ms}"),
        ]
    else:
        raise ValueError(f"unknown cumulative pipeline anchor: {anchor}")
    return anchor_filter, rows


def _pipeline_cumulative_profile_sql(*, anchor: str) -> str:
    # Diagnostics created_at_ms is captured when StrategyRunner opens DecisionContext,
    # i.e. strategy pipeline start. Stage timings are monotonic deltas from that point.
    strategy_start = "f.controller_start_ms"
    pipeline_end = "(f.controller_start_ms + COALESCE(f.controller_total_ms, 0))"
    stage_sum = (
        f"{strategy_start} + "
        "COALESCE(f.clean_ms, 0) + COALESCE(f.decide_ms, 0) + "
        "COALESCE(f.risk_ms, 0) + COALESCE(f.publish_ms, 0)"
    )
    anchor_filter, rows = _pipeline_cumulative_milestones(
        anchor=anchor,
        strategy_start=strategy_start,
        stage_sum=stage_sum,
        pipeline_end=pipeline_end,
    )
    milestones = ",\n      ".join(
        f"({pipeline_order}, '{metric}', {value_expr})"
        for pipeline_order, metric, value_expr in rows
    )
    return f"""WITH {_pipeline_frames_cte_sql()}
SELECT
  f."time",
  f.metric_prefix || LPAD(s.pipeline_order::text, 2, '0') || ' ' || s.metric AS metric,
  s.value AS value
FROM frames f
CROSS JOIN LATERAL (
  VALUES
      {milestones}
) AS s(pipeline_order, metric, value)
WHERE {anchor_filter}
  AND s.value IS NOT NULL
ORDER BY 1, s.pipeline_order"""


def _pipeline_cumulative_panel(
    id_: int,
    title: str,
    expr: str,
    *,
    x: int,
    y: int,
    w: int = 24,
    h: int = 10,
) -> dict[str, Any]:
    panel = _panel(
        id_,
        title,
        "timeseries",
        PG_DS,
        expr,
        x=x,
        y=y,
        w=w,
        h=h,
        unit="ms",
    )
    panel["options"] = {
        "legend": {
            "calcs": [],
            "displayMode": "list",
            "placement": "bottom",
            "showLegend": True,
            "sortBy": "Name",
            "sortDesc": False,
        },
        "tooltip": {"mode": "multi", "sort": "asc"},
    }
    return panel


def _model_nll_step_sql() -> str:
    nll = _numeric_sql("m.value->>'nll'", default="NULL")
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    return f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  m.key AS metric,
  {nll} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(COALESCE(e.payload->'models', '{{}}'::jsonb)) AS m(key, value)
WHERE {_diagnostics_filter()}
  AND m.value->>'nll' IS NOT NULL
ORDER BY 1"""


def _strategy_decision_filter(alias: str = "e") -> str:
    prefix = f"{alias}." if alias else ""
    return f"""{prefix}event_type = 'StrategyDecision'
  AND {prefix}exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND {prefix}symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND {prefix}actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
  AND $__timeFilter({prefix}ts)"""


def _model_params_values_sql(scope: str = "all") -> str:
    local = """('local_price_model', 'local', 'price'),
    ('local_market_flow_model', 'local', 'market_flow')"""
    global_ = """('global_price_model', 'global', 'price'),
    ('global_market_flow_model', 'global', 'market_flow')"""
    if scope == "local":
        values = local
    elif scope == "global":
        values = global_
    elif scope == "price":
        values = """('local_price_model', 'local', 'price'),
    ('global_price_model', 'global', 'price')"""
    elif scope == "market_flow":
        values = """('local_market_flow_model', 'local', 'market_flow'),
    ('global_market_flow_model', 'global', 'market_flow')"""
    else:
        values = f"{local},\n    {global_}"
    return f"""VALUES
    {values}"""


def _strategy_model_params_flat_cte(scope: str = "all") -> str:
    return f"""WITH RECURSIVE model_defs(model_key, venue_scope, model_type) AS (
  {_model_params_values_sql(scope)}
),
source AS (
  SELECT e.ts,
         e.actor_id,
         e.exchange,
         e.symbol,
         d.model_key,
         d.venue_scope,
         d.model_type,
         e.payload->d.model_key AS model_payload,
         {_jsonb_object_or_empty_sql("e.payload->d.model_key->'params'")} AS params
  FROM mm_events e
  CROSS JOIN model_defs d
  WHERE {_strategy_decision_filter("e")}
    AND e.payload ? d.model_key
),
flat(ts, actor_id, exchange, symbol, model_key, venue_scope, model_type, path, value) AS (
  SELECT s.ts,
         s.actor_id,
         s.exchange,
         s.symbol,
         s.model_key,
         s.venue_scope,
         s.model_type,
         kv.key AS path,
         kv.value AS value
  FROM source s
  CROSS JOIN LATERAL jsonb_each(s.params) AS kv(key, value)
  UNION ALL
  SELECT f.ts,
         f.actor_id,
         f.exchange,
         f.symbol,
         f.model_key,
         f.venue_scope,
         f.model_type,
         child.path AS path,
         child.value AS value
  FROM flat f
  CROSS JOIN LATERAL (
    SELECT f.path || '.' || kv.key AS path,
           kv.value AS value
    FROM jsonb_each({_jsonb_object_or_empty_sql("f.value")}) AS kv(key, value)
    WHERE jsonb_typeof(f.value) = 'object'
    UNION ALL
    SELECT f.path || '[' || (arr.idx - 1)::text || ']' AS path,
           arr.value AS value
    FROM jsonb_array_elements({_jsonb_array_or_empty_sql("f.value")}) WITH ORDINALITY AS arr(value, idx)
    WHERE jsonb_typeof(f.value) = 'array'
  ) AS child
),
numeric_params AS (
  SELECT ts,
         actor_id,
         exchange,
         symbol,
         model_key,
         venue_scope,
         model_type,
         path,
         model_key || '.' || path AS metric,
         (value #>> '{{}}')::double precision AS value
  FROM flat
  WHERE jsonb_typeof(value) = 'number'
  UNION ALL
  SELECT s.ts,
         s.actor_id,
         s.exchange,
         s.symbol,
         s.model_key,
         s.venue_scope,
         s.model_type,
         'active_in_mpc' AS path,
         s.model_key || '.active_in_mpc' AS metric,
         CASE WHEN s.model_payload->'active_in_mpc' = 'true'::jsonb THEN 1.0 ELSE 0.0 END AS value
  FROM source s
  WHERE s.model_payload ? 'active_in_mpc'
)"""


def _strategy_model_params_sql(
    *,
    scope: str = "all",
    model_type: str | None = None,
    param_regex: str | None = None,
    exclude_regex: str | None = None,
) -> str:
    filters = []
    if model_type is not None:
        filters.append(f"model_type = '{model_type}'")
    if param_regex is not None:
        filters.append(f"metric ~ '{param_regex}'")
    if exclude_regex is not None:
        filters.append(f"metric !~ '{exclude_regex}'")
    where = ""
    if filters:
        where = "\nWHERE " + "\n  AND ".join(filters)
    return f"""{_strategy_model_params_flat_cte(scope)}
SELECT ts AS "time",
       metric,
       value
FROM numeric_params{where}
ORDER BY 1, 2"""


_PRICE_SCALE_CONFIG_PARAM_REGEX = r"\.(reference_vwap_vol|shape_alpha|student_df|active_in_mpc)$"


def _latest_strategy_model_params_sql() -> str:
    return f"""WITH model_defs(model_key, venue_scope, model_type) AS (
  {_model_params_values_sql("all")}
),
latest AS (
  SELECT e.ts,
         e.actor_id,
         e.exchange,
         e.symbol,
         e.payload
  FROM mm_events e
  WHERE {_strategy_decision_filter("e")}
  ORDER BY e.ts DESC
  LIMIT 1
)
SELECT l.ts,
       l.actor_id,
       l.exchange,
       l.symbol,
       d.venue_scope,
       d.model_type,
       d.model_key,
       COALESCE(l.payload->d.model_key->>'kind', '') AS kind,
       CASE
         WHEN l.payload->d.model_key ? 'active_in_mpc'
         THEN COALESCE(l.payload->d.model_key->>'active_in_mpc', '')
         ELSE ''
       END AS active_in_mpc,
       l.payload->d.model_key->'params' AS params
FROM latest l
CROSS JOIN model_defs d
WHERE l.payload ? d.model_key
ORDER BY d.venue_scope, d.model_type"""


def _risk_approved_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    return f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  (e.exchange || ' ' || e.symbol) AS metric,
  CASE WHEN COALESCE(e.payload #>> ARRAY['risk','approved'], 'false') = 'true' THEN 1.0 ELSE 0.0 END AS value
FROM mm_events e
WHERE {_diagnostics_filter()}
ORDER BY 1"""


def _risk_reject_step_sql() -> str:
    count_value = _numeric_sql("reason.value", default="0.0")
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  reason.key AS metric,
  SUM({count_value}) AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['risk','rejection_counts'], '{{}}'::jsonb)) AS reason(key, value)
WHERE {_diagnostics_filter()}
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT 'risk rejects'::text AS metric
  UNION
  SELECT DISTINCT reason.key AS metric
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['risk','rejection_counts'], '{{}}'::jsonb)) AS reason(key, value)
  WHERE {_diagnostics_filter()}"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _execution_gate_block_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  reason.value AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(e.payload #> ARRAY['execution_gates','blocked_reasons'], '[]'::jsonb)) AS reason(value)
WHERE {_diagnostics_filter()}
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT 'gate blocks'::text AS metric
  UNION
  SELECT DISTINCT reason.value AS metric
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(e.payload #> ARRAY['execution_gates','blocked_reasons'], '[]'::jsonb)) AS reason(value)
  WHERE {_diagnostics_filter()}"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _execution_sync_outcome_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  (COALESCE(e.exchange, 'unknown') || ' ' ||
   CASE e.event_type WHEN 'ExecutionSyncCompleted' THEN 'completed' ELSE 'failed' END) AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE e.event_type IN ('ExecutionSyncCompleted', 'ExecutionSyncFailed')
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT scope.exchange || ' ' || outcome.outcome AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope
  CROSS JOIN (VALUES ('completed'), ('failed')) AS outcome(outcome)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _router_failure_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  (COALESCE(e.exchange, 'unknown') || ' ' ||
   CASE WHEN e.event_type = 'CancelOrdersFailed' THEN 'cancel' ELSE 'place' END || ' ' ||
   COALESCE(NULLIF(e.payload->>'error', ''), 'unknown')) AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE e.event_type IN ('CancelOrdersFailed', 'PlaceOrdersFailed')
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT scope.exchange || ' ' || kind.kind || ' unknown' AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope
  CROSS JOIN (VALUES ('cancel'), ('place')) AS kind(kind)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _validation_reject_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  (COALESCE(e.exchange, 'unknown') || ' ' || COALESCE(NULLIF(e.payload->>'reason', ''), 'unknown')) AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE e.event_type = 'OrderValidationRejected'
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT DISTINCT scope.exchange || ' unknown' AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _execution_error_event_filter(alias: str = "e") -> str:
    return f"""{alias}.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (
    {alias}.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    OR {alias}.symbol IS NULL
    OR EXISTS (
      SELECT 1
      FROM jsonb_array_elements(COALESCE({alias}.payload->'orders', '[]'::jsonb)) AS ord(value)
      WHERE ord.value->>'symbol' = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    )
  )
  AND $__timeFilter({alias}.ts)"""


def _execution_error_totals_sql() -> str:
    return f"""SELECT
  CASE e.event_type
    WHEN 'PlaceOrdersFailed' THEN 'place failed'
    WHEN 'CancelOrdersFailed' THEN 'cancel failed'
    WHEN 'OrderValidationRejected' THEN 'validation rejected'
    WHEN 'ExecutionSyncFailed' THEN 'sync failed'
    ELSE e.event_type
  END AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE e.event_type IN ('PlaceOrdersFailed', 'CancelOrdersFailed', 'OrderValidationRejected', 'ExecutionSyncFailed')
  AND {_execution_error_event_filter()}
GROUP BY 1
ORDER BY 2 DESC"""


def _execution_error_by_exchange_sql() -> str:
    return f"""SELECT
  e.exchange || ' ' ||
  CASE e.event_type
    WHEN 'PlaceOrdersFailed' THEN 'place failed'
    WHEN 'CancelOrdersFailed' THEN 'cancel failed'
    WHEN 'OrderValidationRejected' THEN 'validation rejected'
    WHEN 'ExecutionSyncFailed' THEN 'sync failed'
    ELSE e.event_type
  END AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE e.event_type IN ('PlaceOrdersFailed', 'CancelOrdersFailed', 'OrderValidationRejected', 'ExecutionSyncFailed')
  AND {_execution_error_event_filter()}
GROUP BY 1
ORDER BY 2 DESC"""


def _execution_error_flow_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  metric,
  COUNT(*)::double precision AS value
FROM (
  SELECT e.ts,
         e.payload,
         e.event_type,
         CASE e.event_type
           WHEN 'PlaceOrdersRequested' THEN e.exchange || ' place requested'
           WHEN 'PlaceOrdersAccepted' THEN e.exchange || ' place accepted'
           WHEN 'PlaceOrdersFailed' THEN e.exchange || ' place failed'
           WHEN 'CancelOrdersFailed' THEN e.exchange || ' cancel failed'
           WHEN 'OrderValidationRejected' THEN e.exchange || ' validation rejected'
           ELSE e.exchange || ' sync failed'
         END AS metric
  FROM mm_events e
  WHERE e.event_type IN ('PlaceOrdersRequested', 'PlaceOrdersAccepted', 'PlaceOrdersFailed', 'CancelOrdersFailed', 'OrderValidationRejected', 'ExecutionSyncFailed')
    AND {_execution_error_event_filter()}
) e
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT scope.exchange || ' ' || state.state AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope
  CROSS JOIN (
    VALUES ('place requested'), ('place accepted'), ('place failed'), ('cancel failed'), ('validation rejected'), ('sync failed')
  ) AS state(state)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _execution_error_reason_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  e.exchange || ' ' ||
  CASE
    WHEN e.event_type = 'OrderValidationRejected' THEN COALESCE(NULLIF(e.payload->>'reason', ''), 'unknown')
    WHEN e.event_type IN ('PlaceOrdersFailed', 'CancelOrdersFailed', 'ExecutionSyncFailed') THEN
      COALESCE(NULLIF(e.payload->>'error', ''), 'unknown')
    ELSE 'unknown'
  END AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE e.event_type IN ('PlaceOrdersFailed', 'CancelOrdersFailed', 'OrderValidationRejected', 'ExecutionSyncFailed')
  AND {_execution_error_event_filter()}
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT scope.exchange || ' unknown' AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _execution_error_message_table_sql() -> str:
    return f"""SELECT
  MAX(e.ts) AS last_seen,
  e.exchange,
  CASE e.event_type
    WHEN 'PlaceOrdersFailed' THEN 'place'
    WHEN 'CancelOrdersFailed' THEN 'cancel'
    WHEN 'OrderValidationRejected' THEN 'validation'
    WHEN 'ExecutionSyncFailed' THEN 'sync'
    ELSE e.event_type
  END AS operation,
  COALESCE(NULLIF(e.payload->>'error', ''), NULLIF(e.payload->>'reason', ''), 'unknown') AS error_or_reason,
  COALESCE(NULLIF(e.payload->>'message', ''), NULLIF(e.payload->>'reason', ''), 'unknown') AS message,
  COUNT(*) AS count
FROM mm_events e
WHERE e.event_type IN ('PlaceOrdersFailed', 'CancelOrdersFailed', 'OrderValidationRejected', 'ExecutionSyncFailed')
  AND {_execution_error_event_filter()}
GROUP BY 2, 3, 4, 5
ORDER BY count DESC, last_seen DESC
LIMIT 100"""


def _recent_execution_errors_sql() -> str:
    return f"""SELECT
  e.ts,
  e.exchange,
  COALESCE(e.symbol, ord.value->>'symbol') AS symbol,
  e.event_type,
  COALESCE(NULLIF(e.payload->>'error', ''), NULLIF(e.payload->>'reason', ''), 'unknown') AS error_or_reason,
  COALESCE(NULLIF(e.payload->>'message', ''), NULLIF(e.payload->>'reason', ''), '') AS message,
  COALESCE(e.payload->>'side', ord.value->>'side') AS side,
  COALESCE(e.payload->>'price', ord.value->>'price') AS price,
  COALESCE(e.payload->>'qty', ord.value->>'qty') AS qty
FROM mm_events e
LEFT JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[{{}}]'::jsonb)) AS ord(value) ON TRUE
WHERE e.event_type IN ('PlaceOrdersFailed', 'CancelOrdersFailed', 'OrderValidationRejected', 'ExecutionSyncFailed')
  AND {_execution_error_event_filter()}
ORDER BY e.ts DESC
LIMIT 200"""


def _operator_order_flow_step_sql() -> str:
    target_step = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    event_step = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    fill_ts = "COALESCE(to_timestamp((" + _numeric_sql("e.payload->>'timestamp_ms'", default="NULL") + ") / 1000.0), e.ts)"
    data_sql = f"""SELECT "time", metric, SUM(value) AS value
FROM (
  SELECT {_approx_step_time_alias_sql(target_step)},
         'target orders'::text AS metric,
         SUM(
           jsonb_array_length(COALESCE(e.payload->'local'->'orders', '[]'::jsonb))
           + jsonb_array_length(COALESCE(e.payload->'global'->'orders', '[]'::jsonb))
         )::double precision AS value
  FROM mm_events e
  WHERE e.topic = '${{target_books_topic}}'
    AND e.event_type = 'JointTargetBook'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
  UNION ALL
  SELECT {_approx_step_time_alias_sql(event_step)},
         'placed orders'::text AS metric,
         COUNT(*)::double precision AS value
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE e.event_type = 'PlaceOrdersAccepted'
    AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
  UNION ALL
  SELECT {_approx_step_time_alias_sql(event_step)},
         'canceled order ids'::text AS metric,
         SUM(jsonb_array_length(COALESCE(e.payload->'order_ids', '[]'::jsonb)))::double precision AS value
  FROM mm_events e
  WHERE e.event_type = 'CancelOrdersAccepted'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
  UNION ALL
  SELECT {_approx_step_time_alias_sql(event_step)},
         'rejected order commands'::text AS metric,
         COUNT(*)::double precision AS value
  FROM mm_events e
  WHERE e.event_type = 'OrderValidationRejected'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
  UNION ALL
  SELECT {_approx_step_time_alias_sql(event_step)},
         'rejected order commands'::text AS metric,
         COUNT(*)::double precision AS value
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE e.event_type = 'PlaceOrdersFailed'
    AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
  UNION ALL
  SELECT {_approx_step_time_alias_sql(fill_ts)},
         'fill events'::text AS metric,
         COUNT(*)::double precision AS value
  FROM mm_events e
  WHERE e.topic LIKE 'private:state:%:fills'
    AND COALESCE(e.exchange, e.payload->>'exchange', substring(e.topic from '^private:state:([^:]+):fills$'), 'unknown') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(e.symbol, e.payload->>'symbol', '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
) flow
GROUP BY 1, 2"""
    metrics_sql = """  SELECT metric
  FROM (
    VALUES
      ('target orders'),
      ('placed orders'),
      ('canceled order ids'),
      ('rejected order commands'),
      ('fill events')
  ) AS m(metric)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _target_ladder_level_sql(measure: str) -> str:
    if measure not in {"price", "qty"}:
        raise ValueError(f"unknown target ladder measure: {measure}")
    value = _numeric_sql(f"o.value->>'{measure}'", default="NULL")
    side = _side_sql("o.value->>'side'")
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    bucket = _approx_step_time_sql(step_source)
    return f"""WITH expanded AS (
  SELECT {bucket} AS "time",
         e.ts,
         (book.role || ' ' ||
          COALESCE(book.value->>'exchange', e.exchange, 'unknown') || ' ' ||
          COALESCE(book.value->>'symbol', e.symbol, 'unknown') || ' ' ||
          {side} || ' #' || o.order_no::text) AS metric,
         {value} AS value
  FROM mm_events e
  CROSS JOIN LATERAL (
    VALUES ('local', e.payload->'local'), ('global', e.payload->'global')
  ) AS book(role, value)
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(book.value->'orders', '[]'::jsonb)) WITH ORDINALITY AS o(value, order_no)
  WHERE e.topic = '${{target_books_topic}}'
    AND e.event_type = 'JointTargetBook'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND book.value IS NOT NULL
    AND $__timeFilter(e.ts)
),
latest AS (
  SELECT DISTINCT ON ("time", metric)
         "time",
         metric,
         value
  FROM expanded
  WHERE value IS NOT NULL
  ORDER BY "time", metric, ts DESC
)
SELECT "time", metric, value
FROM latest
ORDER BY 1, 2"""


def _action_frame_order_flow_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  flow.metric,
  SUM(flow.value) AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES
    ('target orders', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','target_book','order_count']")}),
    ('risk passed', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','risk_approved_target','order_count']")}),
    ('kept live', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','kept','order_count']")}),
    ('cancel commands', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','planned_cancel','order_count']")}),
    ('new order candidates', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','planned_place','order_count']")}),
    ('valid new orders', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','rules_valid','order_count']")}),
    ('exchange accepted', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','router_accepted','order_count']")})
) AS flow(metric, value)
WHERE {_diagnostics_filter()}
GROUP BY 1, 2"""
    metrics_sql = """  SELECT metric
  FROM (
    VALUES
      ('target orders'),
      ('risk passed'),
      ('kept live'),
      ('cancel commands'),
      ('new order candidates'),
      ('valid new orders'),
      ('exchange accepted')
  ) AS m(metric)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _execution_sync_counts_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    is_requested = "CASE WHEN e.event_type = 'ExecutionSyncRequested' THEN 1.0 ELSE 0.0 END"
    is_completed = "CASE WHEN e.event_type = 'ExecutionSyncCompleted' THEN 1.0 ELSE 0.0 END"
    is_failed = "CASE WHEN e.event_type = 'ExecutionSyncFailed' THEN 1.0 ELSE 0.0 END"

    def completed_value(field: str) -> str:
        payload_value = _numeric_sql(f"e.payload->>'{field}'")
        return f"CASE WHEN e.event_type = 'ExecutionSyncCompleted' THEN {payload_value} ELSE 0.0 END"

    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  sync.metric,
  SUM(sync.value) AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES
    ('sync requested', {is_requested}),
    ('completed', {is_completed}),
    ('failed', {is_failed}),
    ('target', {completed_value("target_count")}),
    ('kept', {completed_value("keep_count")}),
    ('cancel', {completed_value("cancel_count")}),
    ('place', {completed_value("place_count")}),
    ('accepted', {completed_value("accepted_count")}),
    ('rejected', {completed_value("rejected_count")})
) AS sync(metric, value)
WHERE e.event_type IN ('ExecutionSyncRequested', 'ExecutionSyncCompleted', 'ExecutionSyncFailed')
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
  AND $__timeFilter(e.ts)
GROUP BY 1, 2"""
    metrics_sql = """  SELECT metric
  FROM (
    VALUES
      ('sync requested'),
      ('completed'),
      ('failed'),
      ('target'),
      ('kept'),
      ('cancel'),
      ('place'),
      ('accepted'),
      ('rejected')
  ) AS m(metric)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _execution_order_payload_step_sql(event_type: str) -> str:
    return _runtime_order_payload_rate_sql(event_type, "count")


def _rejected_order_commands_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT "time", metric, SUM(value) AS value
FROM (
  SELECT {_approx_step_time_alias_sql(step_source)},
         (COALESCE(e.exchange, 'unknown') || ' ' ||
          COALESCE(e.symbol, 'unknown') || ' validation ' ||
          COALESCE(NULLIF(e.payload->>'reason', ''), 'unknown')) AS metric,
         COUNT(*)::double precision AS value
  FROM mm_events e
  WHERE e.event_type = 'OrderValidationRejected'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
  UNION ALL
  SELECT {_approx_step_time_alias_sql(step_source)},
         (COALESCE(o.value->>'exchange', e.exchange, 'unknown') || ' ' ||
          COALESCE(o.value->>'symbol', e.symbol, 'unknown') || ' place failed ' ||
          COALESCE(NULLIF(e.payload->>'error', ''), 'unknown')) AS metric,
         COUNT(*)::double precision AS value
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE e.event_type = 'PlaceOrdersFailed'
    AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
) rejected
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT scope.exchange || ' ' || scope.symbol || ' validation unknown' AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope
  UNION
  SELECT scope.exchange || ' ' || scope.symbol || ' place failed unknown' AS metric
  FROM (
    {_mm_scope_pairs_sql()}
  ) AS scope
  UNION
  SELECT DISTINCT
         COALESCE(e.exchange, 'unknown') || ' ' ||
         COALESCE(e.symbol, 'unknown') || ' validation ' ||
         COALESCE(NULLIF(e.payload->>'reason', ''), 'unknown') AS metric
  FROM mm_events e
  WHERE e.event_type = 'OrderValidationRejected'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
    AND $__timeFilter(e.ts)
  UNION
  SELECT DISTINCT
         COALESCE(o.value->>'exchange', e.exchange, 'unknown') || ' ' ||
         COALESCE(o.value->>'symbol', e.symbol, 'unknown') || ' place failed ' ||
         COALESCE(NULLIF(e.payload->>'error', ''), 'unknown') AS metric
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE e.event_type = 'PlaceOrdersFailed'
    AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _venue_gate_status_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    bool_value = lambda field: (
        f"lower(trim(COALESCE(gate.value->>'{field}', 'false'))) IN ('true', 't', '1', 'yes')"
    )
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  (COALESCE(gate.value->>'role', 'unknown') || ' ' || gate.key || ' ' || status.metric) AS metric,
  MAX(CASE WHEN status.ok THEN 1.0 ELSE 0.0 END) AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(
  COALESCE(e.payload #> ARRAY['execution_gates','venue_gates'], '{{}}'::jsonb)
) AS gate(key, value)
CROSS JOIN LATERAL (
  VALUES
    ('has orders', {bool_value("has_orders")}),
    ('can sync', {bool_value("can_sync")}),
    ('live', {bool_value("live")})
) AS status(metric, ok)
WHERE {_diagnostics_filter()}
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT DISTINCT
         COALESCE(gate.value->>'role', 'unknown') || ' ' || gate.key || ' ' || status.metric AS metric
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_each(
    COALESCE(e.payload #> ARRAY['execution_gates','venue_gates'], '{{}}'::jsonb)
  ) AS gate(key, value)
  CROSS JOIN LATERAL (
    VALUES ('has orders'), ('can sync'), ('live')
  ) AS status(metric)
  WHERE {_diagnostics_filter()}"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _latest_target_ladder_qty_price_sql() -> str:
    side = _side_sql("o.value->>'side'")
    price = _numeric_sql("o.value->>'price'", default="NULL")
    qty = _numeric_sql("o.value->>'qty'")
    return f"""WITH latest_book AS (
  SELECT e.ts,
         e.exchange,
         e.symbol,
         e.actor_id,
         e.payload
  FROM mm_events e
  WHERE e.topic = '${{target_books_topic}}'
    AND e.event_type = 'JointTargetBook'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
),
orders AS (
  SELECT {side} AS side,
         {price} AS price,
         {qty} AS qty
  FROM latest_book lb
  CROSS JOIN LATERAL (
    VALUES ('local', lb.payload->'local'), ('global', lb.payload->'global')
  ) AS book(role, value)
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(book.value->'orders', '[]'::jsonb)) AS o(value)
  WHERE book.value IS NOT NULL
    AND ('${{target_role}}' = 'all' OR book.role = '${{target_role}}')
)
SELECT side AS series,
       price,
       SUM(qty) AS value
FROM orders
WHERE side IN ('bid', 'ask')
  AND price IS NOT NULL
  AND qty > 0
GROUP BY 1, 2
ORDER BY 1, 2"""


def _latest_strategy_ladder_qty_price_sql() -> str:
    side = _side_sql("o.value->>'side'")
    price = _numeric_sql("o.value->>'price'", default="NULL")
    qty = _numeric_sql("o.value->>'qty'")
    return f"""WITH latest_decision AS (
  SELECT e.ts,
         e.payload
  FROM mm_events e
  WHERE e.event_type = 'StrategyDecision'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
),
orders AS (
  SELECT {side} AS side,
         {price} AS price,
         {qty} AS qty
  FROM latest_decision d
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(d.payload->'ladder_orders', '[]'::jsonb)) AS o(value)
)
SELECT side AS series,
       price,
       SUM(qty) AS value
FROM orders
WHERE side IN ('bid', 'ask')
  AND price IS NOT NULL
  AND qty > 0
GROUP BY 1, 2
ORDER BY 1, 2"""


def _latest_place_request_qty_price_sql() -> str:
    side = _side_sql("o.value->>'side'")
    price = _numeric_sql("o.value->>'price'", default="NULL")
    qty = _numeric_sql("o.value->>'qty'")
    return f"""WITH latest_request AS (
  SELECT e.ts,
         e.exchange,
         e.payload
  FROM mm_events e
  WHERE e.event_type = 'PlaceOrdersRequested'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
),
orders AS (
  SELECT {side} AS side,
         {price} AS price,
         {qty} AS qty
  FROM latest_request e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
)
SELECT side AS series,
       price,
       SUM(qty) AS value
FROM orders
WHERE side IN ('bid', 'ask')
  AND price IS NOT NULL
  AND qty > 0
GROUP BY 1, 2
ORDER BY 1, 2"""


def _rejected_command_qty_price_sql() -> str:
    validation_side = _side_sql("e.payload->>'side'")
    failed_side = _side_sql("o.value->>'side'")
    validation_price = _numeric_sql("e.payload->>'price'", default="NULL")
    validation_qty = _numeric_sql("e.payload->>'qty'")
    failed_price = _numeric_sql("o.value->>'price'", default="NULL")
    failed_qty = _numeric_sql("o.value->>'qty'")
    return f"""WITH rejected AS (
  SELECT ('validation: ' || COALESCE(NULLIF(e.payload->>'reason', ''), 'unknown') || ' ' || {validation_side}) AS series,
         {validation_price} AS price,
         {validation_qty} AS qty
  FROM mm_events e
  WHERE e.event_type = 'OrderValidationRejected'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
    AND $__timeFilter(e.ts)
  UNION ALL
  SELECT ('router: ' || COALESCE(NULLIF(e.payload->>'error', ''), 'unknown') || ' ' || {failed_side}) AS series,
         {failed_price} AS price,
         {failed_qty} AS qty
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE e.event_type = 'PlaceOrdersFailed'
    AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
)
SELECT series,
       price,
       SUM(qty) AS value
FROM rejected
WHERE price IS NOT NULL
  AND qty > 0
GROUP BY 1, 2
ORDER BY 1, 2"""


def _placed_vs_canceled_qty_price_sql() -> str:
    placed_price = _numeric_sql("o.value->>'price'", default="NULL")
    placed_qty = _numeric_sql("o.value->>'qty'")
    canceled_price = _numeric_sql("live_order.value #>> ARRAY['order','price']", default="NULL")
    canceled_qty = _numeric_sql("COALESCE(live_order.value->>'remaining_qty', live_order.value #>> ARRAY['order','qty'])")
    return f"""WITH placed AS (
  SELECT 'placed orders'::text AS series,
         {placed_price} AS price,
         {placed_qty} AS qty
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE e.event_type = 'PlaceOrdersAccepted'
    AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
),
cancel_ids AS (
  SELECT e.ts,
         e.exchange,
         order_id.value AS order_id
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(e.payload->'order_ids', '[]'::jsonb)) AS order_id(value)
  WHERE e.event_type = 'CancelOrdersAccepted'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
),
canceled AS (
  SELECT 'canceled order ids'::text AS series,
         {canceled_price} AS price,
         {canceled_qty} AS qty
  FROM cancel_ids c
  JOIN LATERAL (
    SELECT open_order.value
    FROM mm_events s
    CROSS JOIN LATERAL jsonb_each(COALESCE(s.payload->'live_orders', '{{}}'::jsonb)) AS open_order(key, value)
    WHERE s.topic LIKE 'execution:state:%'
      AND s.exchange = c.exchange
      AND s.ts <= c.ts
      AND open_order.key = c.order_id
      AND COALESCE(open_order.value #>> ARRAY['order','symbol'], '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    ORDER BY s.ts DESC
    LIMIT 1
  ) AS live_order ON TRUE
),
commands AS (
  SELECT * FROM placed
  UNION ALL
  SELECT * FROM canceled
),
grouped AS (
SELECT series,
       price,
       SUM(qty) AS value
FROM commands
WHERE price IS NOT NULL
  AND qty > 0
GROUP BY 1, 2
)
SELECT series,
       price,
       value
FROM grouped
ORDER BY grouped.price, series"""


def _previous_live_vs_canceled_qty_price_sql() -> str:
    side = _side_sql("live_order.value #>> ARRAY['order','side']")
    price = _numeric_sql("live_order.value #>> ARRAY['order','price']", default="NULL")
    qty = _numeric_sql("COALESCE(live_order.value->>'remaining_qty', live_order.value #>> ARRAY['order','qty'])")
    return f"""WITH latest_cancel AS (
  SELECT e.ts,
         e.exchange,
         e.payload
  FROM mm_events e
  WHERE e.event_type = 'CancelOrdersAccepted'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
),
cancel_ids AS (
  SELECT order_id.value AS order_id
  FROM latest_cancel c
  CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(c.payload->'order_ids', '[]'::jsonb)) AS order_id(value)
),
previous_state AS (
  SELECT s.ts,
         s.exchange,
         s.payload
  FROM latest_cancel c
  JOIN LATERAL (
    SELECT s.ts, s.exchange, s.payload
    FROM mm_events s
    WHERE s.topic LIKE 'execution:state:%'
      AND s.exchange = c.exchange
      AND s.ts <= c.ts
      AND $__timeFilter(s.ts)
    ORDER BY s.ts DESC
    LIMIT 1
  ) s ON TRUE
),
live_orders AS (
  SELECT live_order.key AS order_id,
         {side} AS side,
         {price} AS price,
         {qty} AS qty
  FROM previous_state s
  CROSS JOIN LATERAL jsonb_each(COALESCE(s.payload->'live_orders', '{{}}'::jsonb)) AS live_order(key, value)
  WHERE COALESCE(live_order.value #>> ARRAY['order','symbol'], '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
),
series AS (
  SELECT CASE
           WHEN cancel_ids.order_id IS NOT NULL THEN 'canceled ' || live_orders.side
           ELSE 'previously live ' || live_orders.side
         END AS series,
         live_orders.price,
         live_orders.qty AS value
  FROM live_orders
  LEFT JOIN cancel_ids ON cancel_ids.order_id = live_orders.order_id
  WHERE live_orders.side IN ('bid', 'ask')
    AND live_orders.price IS NOT NULL
),
{_stem_axis_ctes()}
SELECT series,
       price,
       SUM(value) AS value
FROM series
WHERE value > 0
GROUP BY 1, 2
UNION ALL
SELECT series, price, value
FROM axis_rows
ORDER BY price, series"""


def _stem_axis_ctes() -> str:
    raw_side = _side_sql("axis_o.value->>'side'")
    raw_price = _numeric_sql("axis_o.value->>'price'", default="NULL")
    raw_qty = _numeric_sql("axis_o.value->>'qty'")
    target_side = _side_sql("axis_o.value->>'side'")
    target_price = _numeric_sql("axis_o.value->>'price'", default="NULL")
    target_qty = _numeric_sql("axis_o.value->>'qty'")
    return f"""axis_latest_book AS (
  SELECT e.ts,
         e.exchange,
         e.symbol,
         e.actor_id,
         e.payload
  FROM mm_events e
  WHERE e.topic = '${{target_books_topic}}'
    AND e.event_type = 'JointTargetBook'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
),
axis_latest_decision AS (
  SELECT e.ts,
         e.payload
  FROM mm_events e
  WHERE e.event_type = 'StrategyDecision'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
),
axis_raw_orders AS (
  SELECT {raw_side} AS side,
         {raw_price} AS price,
         SUM({raw_qty}) AS qty
  FROM axis_latest_decision d
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(d.payload->'ladder_orders', '[]'::jsonb)) AS axis_o(value)
  GROUP BY 1, 2
),
axis_target_orders AS (
  SELECT {target_side} AS side,
         {target_price} AS price,
         SUM({target_qty}) AS qty
  FROM axis_latest_book lb
  CROSS JOIN LATERAL (
    VALUES ('local', lb.payload->'local'), ('global', lb.payload->'global')
  ) AS book(role, value)
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(book.value->'orders', '[]'::jsonb)) AS axis_o(value)
  WHERE book.value IS NOT NULL
    AND ('${{target_role}}' = 'all' OR book.role = '${{target_role}}')
  GROUP BY 1, 2
),
axis_points AS (
  SELECT axis_target_orders.price,
         axis_target_orders.qty AS value
  FROM axis_target_orders
  WHERE axis_target_orders.side IN ('bid', 'ask')
    AND axis_target_orders.price IS NOT NULL
  UNION ALL
  SELECT axis_raw_orders.price,
         GREATEST(axis_raw_orders.qty - COALESCE(axis_target_orders.qty, 0.0), 0.0) AS value
  FROM axis_raw_orders
  LEFT JOIN axis_target_orders
    ON axis_target_orders.side = axis_raw_orders.side
   AND axis_target_orders.price = axis_raw_orders.price
  WHERE axis_raw_orders.side IN ('bid', 'ask')
    AND axis_raw_orders.price IS NOT NULL
),
axis_frame AS (
  SELECT MIN(price) AS price_min,
         MAX(price) AS price_max,
         GREATEST(MAX(value), 0.0) AS qty_max
  FROM axis_points
  WHERE value > 0
),
axis_rows AS (
  SELECT 'axis range' AS series, price_min AS price, 0.0 AS value
  FROM axis_frame
  WHERE price_min IS NOT NULL
  UNION ALL
  SELECT 'axis range' AS series, price_max AS price, 0.0 AS value
  FROM axis_frame
  WHERE price_max IS NOT NULL
  UNION ALL
  SELECT 'axis range' AS series, price_min AS price, qty_max AS value
  FROM axis_frame
  WHERE price_min IS NOT NULL
  UNION ALL
  SELECT 'axis range' AS series, price_max AS price, qty_max AS value
  FROM axis_frame
  WHERE price_max IS NOT NULL
)"""


def _pipeline_exchange_filter(order_exchange_sql: str, *, role: str | None) -> str:
    if role is None:
        return f"""
    AND COALESCE({order_exchange_sql}, l.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])"""
    return f"""
    AND COALESCE({order_exchange_sql}, '') IN (
      SELECT exchange FROM role_exchanges WHERE role = '{role}'
    )"""


def _role_exchanges_cte() -> str:
    venue_gates_expr = "l.payload #> ARRAY['execution_gates', 'venue_gates']"
    return f""",
role_exchanges AS (
  SELECT 'local'::text AS role,
         l.exchange AS exchange
  FROM latest l
  WHERE l.exchange IS NOT NULL
  UNION
  SELECT gates.value->>'role' AS role,
         gates.key AS exchange
  FROM latest l
  CROSS JOIN LATERAL jsonb_each(
    {_jsonb_object_or_empty_sql(venue_gates_expr)}
  ) AS gates(key, value)
  WHERE gates.value->>'role' IN ('local', 'global')
)"""


def _latest_action_pipeline_stage_orders_sql(
    stage: str,
    *,
    field: str = "orders",
    reason: str | None = None,
    include_reason_in_series: bool = False,
    role: str | None = None,
) -> str:
    if role is not None and role not in {"local", "global"}:
        raise ValueError(f"unknown action pipeline role: {role}")
    side = _side_sql("o.value->>'side'")
    series = side
    if include_reason_in_series:
        series = f"(COALESCE(NULLIF(o.value->>'reason', ''), 'unknown') || ' ' || {side})"
    price = _numeric_sql("o.value->>'price'", default="NULL")
    qty = _numeric_sql("o.value->>'qty'")
    axis_price = _numeric_sql("axis_o.value->>'price'", default="NULL")
    axis_qty = _numeric_sql("axis_o.value->>'qty'")
    orders_expr = _action_pipeline_orders_array_sql(stage, field=field)
    axis_orders_expr = _action_pipeline_orders_array_sql("ladder")
    reason_filter = ""
    if reason is not None:
        reason_filter = f"\n    AND COALESCE(o.value->>'reason', '') = '{reason}'"
    role_filter = _pipeline_exchange_filter("o.value->>'exchange'", role=role)
    axis_role_filter = _pipeline_exchange_filter("axis_o.value->>'exchange'", role=role)
    grid_price_filter = ""
    if stage == "ladder":
        grid_price_filter = f"""
    AND {price} IN (
      SELECT grid.price
      FROM (
        SELECT CASE
          WHEN trim(COALESCE(elem->>'price', '')) ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$'
          THEN trim(COALESCE(elem->>'price', ''))::double precision
          ELSE NULL
        END AS price
        FROM latest l
        CROSS JOIN LATERAL jsonb_array_elements(
          COALESCE(l.payload #> ARRAY['action_pipeline', 'continuous', 'bid'], '[]'::jsonb)
          || COALESCE(l.payload #> ARRAY['action_pipeline', 'continuous', 'ask'], '[]'::jsonb)
        ) AS elem
      ) AS grid
      WHERE grid.price IS NOT NULL
    )"""
    role_exchanges_cte = _role_exchanges_cte() if role is not None else ""
    return f"""WITH latest AS (
  SELECT e.exchange,
         e.payload
  FROM mm_events e
  WHERE e.topic = '${{diagnostics_topic}}'
    AND e.event_type = 'ControllerDiagnostics'
    AND (
      '${{target_frame_mode}}' = 'latest'
      OR e.event_id = ${{target_frame_event_id:sqlstring}}
    )
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
){role_exchanges_cte},
orders AS (
  SELECT {series} AS series,
         {price} AS price,
         {qty} AS value
  FROM latest l
  CROSS JOIN LATERAL jsonb_array_elements(
    {orders_expr}
  ) AS o(value)
  WHERE {side} IN ('bid', 'ask')
    AND {price} IS NOT NULL{reason_filter}{role_filter}{grid_price_filter}
),
axis_points AS (
  SELECT {axis_price} AS price,
         {axis_qty} AS value
  FROM latest l
  CROSS JOIN LATERAL jsonb_array_elements(
    {axis_orders_expr}
  ) AS axis_o(value)
  WHERE {axis_price} IS NOT NULL{axis_role_filter}
),
axis_frame AS (
  SELECT MIN(price) AS price_min,
         MAX(price) AS price_max,
         GREATEST(MAX(value), 0.0) AS qty_max
  FROM axis_points
  WHERE value > 0
),
axis_rows AS (
  SELECT 'axis range' AS series, price_min AS price, 0.0 AS value
  FROM axis_frame
  WHERE price_min IS NOT NULL
  UNION ALL
  SELECT 'axis range' AS series, price_max AS price, 0.0 AS value
  FROM axis_frame
  WHERE price_max IS NOT NULL
  UNION ALL
  SELECT 'axis range' AS series, price_min AS price, qty_max AS value
  FROM axis_frame
  WHERE price_min IS NOT NULL
  UNION ALL
  SELECT 'axis range' AS series, price_max AS price, qty_max AS value
  FROM axis_frame
  WHERE price_max IS NOT NULL
),
grouped AS (
  SELECT series, price, SUM(value) AS value
  FROM orders
  WHERE value > 0
  GROUP BY 1, 2
)
SELECT series, price, value
FROM grouped
UNION ALL
SELECT series, price, value
FROM axis_rows
ORDER BY price, series"""


def _latest_action_pipeline_stage_depth_sql(stage: str, *, role: str | None = "local") -> str:
    if role is not None and role not in {"local", "global"}:
        raise ValueError(f"unknown action pipeline depth role: {role}")
    side = _side_sql("o.value->>'side'")
    price = _numeric_sql("o.value->>'price'", default="NULL")
    qty = _numeric_sql("o.value->>'qty'")
    orders_expr = _action_pipeline_orders_array_sql(stage)
    clean_payload_path = "global_cleared_orderbook" if role == "global" else "cleared_orderbook"
    state_mid_path = "global_mid" if role == "global" else "mid"
    clean_bid = _numeric_sql(
        f"l.payload #>> ARRAY['{clean_payload_path}', 'bids', '0', 'price']",
        default="NULL",
    )
    clean_ask = _numeric_sql(
        f"l.payload #>> ARRAY['{clean_payload_path}', 'asks', '0', 'price']",
        default="NULL",
    )
    exchange_filter = _pipeline_exchange_filter("o.value->>'exchange'", role=role)
    role_exchanges_cte = _role_exchanges_cte() if role is not None else ""
    return f"""WITH latest AS (
  SELECT e.exchange,
         e.payload
  FROM mm_events e
  WHERE e.topic = '${{diagnostics_topic}}'
    AND e.event_type = 'ControllerDiagnostics'
    AND (
      '${{target_frame_mode}}' = 'latest'
      OR e.event_id = ${{target_frame_event_id:sqlstring}}
    )
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
){role_exchanges_cte},
selected AS (
  SELECT COALESCE(
           {_numeric_sql(f"l.payload #>> ARRAY['state', '{state_mid_path}']", default="NULL")},
           ({clean_bid} + {clean_ask}) / 2.0,
           0.0
         ) AS mid,
         GREATEST({_numeric_sql("'${spread_max}'", default="100.0")}, 1.0) AS spread_max
  FROM latest l
),
bid_levels AS (
  SELECT {price} AS price,
         {qty} AS qty
  FROM latest l
  CROSS JOIN LATERAL jsonb_array_elements(
    {orders_expr}
  ) AS o(value)
  WHERE {side} = 'bid'
    AND {price} IS NOT NULL{exchange_filter}
),
ask_levels AS (
  SELECT {price} AS price,
         {qty} AS qty
  FROM latest l
  CROSS JOIN LATERAL jsonb_array_elements(
    {orders_expr}
  ) AS o(value)
  WHERE {side} = 'ask'
    AND {price} IS NOT NULL{exchange_filter}
),
{_orderbook_depth_plot_ctes()}
SELECT 'bid'::text AS side,
       spread,
       value
FROM bid_step_plot
UNION ALL
SELECT 'ask'::text AS side,
       spread,
       value
FROM ask_step_plot
ORDER BY side, spread"""


def _stem_raw_to_target_sql() -> str:
    raw_side = _side_sql("o.value->>'side'")
    raw_price = _numeric_sql("o.value->>'price'", default="NULL")
    raw_qty = _numeric_sql("o.value->>'qty'")
    target_side = _side_sql("o.value->>'side'")
    target_price = _numeric_sql("o.value->>'price'", default="NULL")
    target_qty = _numeric_sql("o.value->>'qty'")
    return f"""WITH latest_book AS (
  SELECT e.ts,
         e.exchange,
         e.symbol,
         e.actor_id,
         e.payload
  FROM mm_events e
  WHERE e.topic = '${{target_books_topic}}'
    AND e.event_type = 'JointTargetBook'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
),
latest_decision AS (
  SELECT e.ts,
         e.payload
  FROM mm_events e
  WHERE e.event_type = 'StrategyDecision'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
),
raw_orders AS (
  SELECT {raw_side} AS side,
         {raw_price} AS price,
         SUM({raw_qty}) AS qty
  FROM latest_decision d
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(d.payload->'ladder_orders', '[]'::jsonb)) AS o(value)
  GROUP BY 1, 2
),
target_orders AS (
  SELECT {target_side} AS side,
         {target_price} AS price,
         SUM({target_qty}) AS qty
  FROM latest_book lb
  CROSS JOIN LATERAL (
    VALUES ('local', lb.payload->'local'), ('global', lb.payload->'global')
  ) AS book(role, value)
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(book.value->'orders', '[]'::jsonb)) AS o(value)
  WHERE book.value IS NOT NULL
    AND ('${{target_role}}' = 'all' OR book.role = '${{target_role}}')
  GROUP BY 1, 2
),
remaining AS (
  SELECT side AS series,
         price,
         qty AS value
  FROM target_orders
  WHERE side IN ('bid', 'ask')
    AND price IS NOT NULL
)
,
{_stem_axis_ctes()},
grouped AS (
  SELECT series, price, SUM(value) AS value
  FROM (
  SELECT * FROM remaining
  ) stem
  WHERE value > 0
  GROUP BY 1, 2
)
SELECT series, price, value
FROM grouped
UNION ALL
SELECT series, price, value
FROM axis_rows
ORDER BY price, series"""


def _stem_target_to_place_sql() -> str:
    target_side = _side_sql("o.value->>'side'")
    target_price = _numeric_sql("o.value->>'price'", default="NULL")
    target_qty = _numeric_sql("o.value->>'qty'")
    request_side = _side_sql("o.value->>'side'")
    request_price = _numeric_sql("o.value->>'price'", default="NULL")
    request_qty = _numeric_sql("o.value->>'qty'")
    return f"""WITH latest_book AS (
  SELECT e.ts,
         e.exchange,
         e.symbol,
         e.actor_id,
         e.payload
  FROM mm_events e
  WHERE e.topic = '${{target_books_topic}}'
    AND e.event_type = 'JointTargetBook'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 1
),
latest_request AS (
  SELECT e.ts,
         e.exchange,
         e.symbol,
         e.payload
  FROM mm_events e
  WHERE e.event_type = 'PlaceOrdersRequested'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
    AND EXISTS (
      SELECT 1
      FROM jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS eo(value)
      WHERE COALESCE(eo.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    )
  ORDER BY e.ts DESC
  LIMIT 1
),
target_orders AS (
  SELECT {target_side} AS side,
         {target_price} AS price,
         SUM({target_qty}) AS qty
  FROM latest_book lb
  CROSS JOIN LATERAL (
    VALUES ('local', lb.payload->'local'), ('global', lb.payload->'global')
  ) AS book(role, value)
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(book.value->'orders', '[]'::jsonb)) AS o(value)
  WHERE book.value IS NOT NULL
    AND ('${{target_role}}' = 'all' OR book.role = '${{target_role}}')
  GROUP BY 1, 2
),
requested AS (
  SELECT {request_side} AS side,
         {request_price} AS price,
         SUM({request_qty}) AS qty
  FROM latest_request e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  GROUP BY 1, 2
),
sent AS (
  SELECT side AS series,
         price,
         qty AS value
  FROM requested
  WHERE side IN ('bid', 'ask')
    AND price IS NOT NULL
)
,
{_stem_axis_ctes()},
grouped AS (
  SELECT series, price, SUM(value) AS value
  FROM (
  SELECT * FROM sent
  ) stem
  WHERE value > 0
  GROUP BY 1, 2
)
SELECT series, price, value
FROM grouped
UNION ALL
SELECT series, price, value
FROM axis_rows
ORDER BY price, series"""


def _stem_place_to_accept_reject_sql() -> str:
    request_side = _side_sql("o.value->>'side'")
    request_price = _numeric_sql("o.value->>'price'", default="NULL")
    request_qty = _numeric_sql("o.value->>'qty'")
    accepted_side = _side_sql("o.value->>'side'")
    accepted_price = _numeric_sql("o.value->>'price'", default="NULL")
    accepted_qty = _numeric_sql("o.value->>'qty'")
    validation_side = _side_sql("e.payload->>'side'")
    validation_price = _numeric_sql("e.payload->>'price'", default="NULL")
    validation_qty = _numeric_sql("e.payload->>'qty'")
    failed_side = _side_sql("o.value->>'side'")
    failed_price = _numeric_sql("o.value->>'price'", default="NULL")
    failed_qty = _numeric_sql("o.value->>'qty'")
    return f"""WITH latest_request AS (
  SELECT e.ts,
         e.exchange,
         e.symbol,
         e.payload
  FROM mm_events e
  WHERE e.event_type = 'PlaceOrdersRequested'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
    AND EXISTS (
      SELECT 1
      FROM jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS eo(value)
      WHERE COALESCE(eo.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    )
  ORDER BY e.ts DESC
  LIMIT 1
),
requested AS (
  SELECT {request_side} AS side,
         {request_price} AS price,
         SUM({request_qty}) AS qty
  FROM latest_request e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  GROUP BY 1, 2
),
accepted AS (
  SELECT {accepted_side} AS side,
         {accepted_price} AS price,
         SUM({accepted_qty}) AS qty
  FROM latest_request r
  JOIN mm_events e
    ON e.event_type = 'PlaceOrdersAccepted'
   AND e.exchange = r.exchange
   AND e.ts >= r.ts
   AND e.ts <= r.ts + interval '5 minutes'
   AND $__timeFilter(e.ts)
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  GROUP BY 1, 2
),
rejected AS (
  SELECT ('rejected: ' || COALESCE(NULLIF(e.payload->>'reason', ''), 'unknown') || ' ' || {validation_side}) AS series,
         {validation_side} AS side,
         {validation_price} AS price,
         SUM({validation_qty}) AS qty
  FROM latest_request r
  JOIN mm_events e
    ON e.event_type = 'OrderValidationRejected'
   AND e.exchange = r.exchange
   AND e.ts >= r.ts
   AND e.ts <= r.ts + interval '5 minutes'
   AND $__timeFilter(e.ts)
  WHERE e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  GROUP BY 1, 2, 3
  UNION ALL
  SELECT ('rejected: ' || COALESCE(NULLIF(e.payload->>'error', ''), 'unknown') || ' ' || {failed_side}) AS series,
         {failed_side} AS side,
         {failed_price} AS price,
         SUM({failed_qty}) AS qty
  FROM latest_request r
  JOIN mm_events e
    ON e.event_type = 'PlaceOrdersFailed'
   AND e.exchange = r.exchange
   AND e.ts >= r.ts
   AND e.ts <= r.ts + interval '5 minutes'
   AND $__timeFilter(e.ts)
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  GROUP BY 1, 2, 3
),
accepted_series AS (
  SELECT side AS series,
         price,
         qty AS value
  FROM accepted
  WHERE side IN ('bid', 'ask')
    AND price IS NOT NULL
),
rejected_series AS (
  SELECT series,
         price,
         qty AS value
  FROM rejected
  WHERE side IN ('bid', 'ask')
    AND price IS NOT NULL
)
,
{_stem_axis_ctes()},
grouped AS (
  SELECT series, price, SUM(value) AS value
  FROM (
  SELECT * FROM accepted_series
  UNION ALL
  SELECT * FROM rejected_series
  ) stem
  WHERE value > 0
  GROUP BY 1, 2
)
SELECT series, price, value
FROM grouped
UNION ALL
SELECT series, price, value
FROM axis_rows
ORDER BY price, series"""

def _action_pipeline_side_stage_counts_sql(
    side: str,
    stages: tuple[tuple[str, str], ...],
) -> str:
    side = "ask" if side == "ask" else "bid"
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    values_sql = ",\n    ".join(f"('{metric}', '{stage}')" for metric, stage in stages)
    metrics_sql = "  SELECT metric\n  FROM (\n    VALUES\n    " + values_sql + "\n  ) AS m(metric, stage)"
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  stages.metric,
  COUNT(*)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES
    {values_sql}
) AS stages(metric, stage)
CROSS JOIN LATERAL jsonb_array_elements(
  COALESCE(e.payload #> ARRAY['action_pipeline', stages.stage, 'orders'], '[]'::jsonb)
) AS o(value)
WHERE {_diagnostics_filter()}
  AND {_side_sql("o.value->>'side'")} = '{side}'
GROUP BY 1, 2"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _action_pipeline_placement_side_summary_sql(measure: str, stage: str, label: str = "placement") -> str:
    if measure not in {"count", "qty"}:
        raise ValueError(f"unknown placement side summary measure: {measure}")
    if stage not in {"planned_place", "router_accepted"}:
        raise ValueError(f"unknown placement side summary stage: {stage}")
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    value_expr = "1.0" if measure == "count" else _numeric_sql("o.value->>'qty'")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  ({_side_sql("o.value->>'side'")} || ' {label}') AS metric,
  SUM({value_expr}) AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements(
  COALESCE(e.payload #> ARRAY['action_pipeline', '{stage}', 'orders'], '[]'::jsonb)
) AS o(value)
WHERE {_diagnostics_filter()}
  AND {_side_sql("o.value->>'side'")} IN ('bid', 'ask')
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT metric
  FROM (
    VALUES ('bid {label}'), ('ask {label}')
  ) AS m(metric)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _action_pipeline_side_removed_reason_counts_sql(side: str) -> str:
    side = "ask" if side == "ask" else "bid"
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    stages_sql = """VALUES
    ('risk_approved_target'),
    ('rules_valid'),
    ('router_accepted')"""
    reason_expr = "COALESCE(NULLIF(o.value->>'reason', ''), 'unknown')"
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  {reason_expr} AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL (
  {stages_sql}
) AS stages(stage)
CROSS JOIN LATERAL jsonb_array_elements(
  COALESCE(e.payload #> ARRAY['action_pipeline', stages.stage, 'rejections'], '[]'::jsonb)
) AS o(value)
WHERE {_diagnostics_filter()}
  AND {_side_sql("o.value->>'side'")} = '{side}'
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT DISTINCT {reason_expr} AS metric
  FROM mm_events e
  CROSS JOIN LATERAL (
    {stages_sql}
  ) AS stages(stage)
  CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(e.payload #> ARRAY['action_pipeline', stages.stage, 'rejections'], '[]'::jsonb)
  ) AS o(value)
  WHERE {_diagnostics_filter()}
    AND {_side_sql("o.value->>'side'")} = '{side}'
  UNION
  SELECT 'no removals'::text AS metric"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _placed_vs_canceled_step_sql() -> str:
    event_step = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT "time", metric, SUM(value) AS value
FROM (
  SELECT {_approx_step_time_alias_sql(event_step)},
         'placed orders'::text AS metric,
         COUNT(*)::double precision AS value
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE e.event_type = 'PlaceOrdersAccepted'
    AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
  UNION ALL
  SELECT {_approx_step_time_alias_sql(event_step)},
         'canceled order ids'::text AS metric,
         SUM(jsonb_array_length(COALESCE(e.payload->'order_ids', '[]'::jsonb)))::double precision AS value
  FROM mm_events e
  WHERE e.event_type = 'CancelOrdersAccepted'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
) commands
GROUP BY 1, 2"""
    metrics_sql = """  SELECT metric
  FROM (
    VALUES ('placed orders'), ('canceled order ids')
  ) AS m(metric)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _operator_failure_step_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  metric,
  SUM(value) AS value
FROM (
  SELECT e.ts,
         e.payload,
         CASE e.event_type
           WHEN 'CancelOrdersFailed' THEN 'cancel failed'
           WHEN 'PlaceOrdersFailed' THEN 'place failed'
           ELSE 'validation rejected'
         END AS metric,
         1.0 AS value
  FROM mm_events e
  WHERE e.event_type IN ('CancelOrdersFailed', 'PlaceOrdersFailed', 'OrderValidationRejected')
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
    AND $__timeFilter(e.ts)
) e
GROUP BY 1, 2"""
    metrics_sql = """  SELECT metric
  FROM (
    VALUES ('cancel failed'), ('place failed'), ('validation rejected')
  ) AS m(metric)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def runtime_overview() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Runtime health from persisted events. Prometheus scrape status is shown separately so a failed `/metrics` socket does not make the whole overview blank.", y=0),
        _panel(2, "Prometheus scrape up", "stat", PROM_DS, 'max(up{job="market_maker"})', x=0, y=3, w=6, h=4, legend="{{job}}"),
        _panel(3, "Frames / step", "timeseries", PG_DS, _runtime_frame_rate_sql(), x=6, y=3),
        _panel(4, "Decisions / step", "timeseries", PG_DS, _runtime_decision_rate_sql(), x=18, y=3, w=6),
        _panel(5, "Controller stage p95 ms", "timeseries", PG_DS, _runtime_controller_stage_p95_sql(), x=0, y=11, unit="ms"),
        _panel(6, "Order sync p95 ms", "timeseries", PG_DS, _runtime_order_sync_p95_sql(), x=12, y=11, w=6, unit="ms"),
        _panel(15, "Sync requests / step", "timeseries", PG_DS, _runtime_execution_event_rate_sql("ExecutionSyncRequested"), x=18, y=11, w=6),
        _panel(7, "Placed orders / step", "timeseries", PG_DS, _runtime_order_payload_rate_sql("PlaceOrdersAccepted", "count"), x=0, y=19, w=6),
        _panel(18, "Cancelled orders / step", "timeseries", PG_DS, _runtime_order_id_event_sql("CancelOrdersAccepted"), x=6, y=19, w=6),
        _panel(8, "Failed orders / step", "timeseries", PG_DS, _runtime_order_payload_rate_sql("PlaceOrdersFailed", "count", include_reason=True), x=12, y=19, w=6),
        _panel(9, "Fill events / step (private feed)", "timeseries", PG_DS, _runtime_fill_rate_sql("count"), x=18, y=19, w=6),
        _panel(10, "Filled qty / step (private feed)", "timeseries", PG_DS, _runtime_fill_rate_sql("qty"), x=0, y=27, w=8),
        _panel(11, "Execution live orders / step (latest snapshot)", "timeseries", PG_DS, _runtime_live_orders_sql("count"), x=8, y=27, w=8),
        _panel(12, "Execution live qty / step (latest snapshot)", "timeseries", PG_DS, _runtime_live_orders_sql("qty"), x=16, y=27, w=8),
        _panel(16, "Latest private vs execution order counts", "table", PG_DS, _runtime_latest_order_counts_sql(), x=0, y=35, w=12, fmt="table"),
        _panel(17, "Recent private fills", "table", PG_DS, _runtime_recent_fills_sql(), x=12, y=35, w=12, fmt="table"),
        _panel(19, "Latest execution gates", "table", PG_DS, _venue_gates_sql(), x=0, y=43, w=12, fmt="table"),
        _panel(14, "Latest controller warnings", "table", PG_DS, _latest_warnings_sql(), x=12, y=43, w=12, fmt="table"),
        _panel(13, "Persistence errors", "timeseries", LOKI_DS, 'sum by (writer,error) (count_over_time({job="market-maker-runtime"} | json | event="persistence_batch_failed" [$__interval]))', x=0, y=51, w=24, legend="{{writer}} {{error}}"),
    ]
    return _dashboard("mm-obs-v2-runtime-overview", "MM Observability v2 / 00 Runtime Overview", panels, tags=["runtime"])


def _raw_feed_metrics_guide_markdown() -> str:
    return """### Raw feed metrics (ingestor Prometheus, pre-aggregation)

These panels track **raw** orderbook/trade payloads as they arrive at the ingestor (`market:raw:*` topics), **before** interval `MarketEvent` aggregation. Each series is labeled by `exchange`, `symbol`, and `kind` (`orderbook` or `trade`).

| Panel | What it measures | How to read it |
|-------|----------------|----------------|
| **Raw feed freshness ms** | `now − latest_event_timestamp_ms` at scrape time | **Staleness of the last seen upstream timestamp.** Flat and low → feed is updating. **Steadily climbing** while the market is open → no new raw events (stuck poll, dead WS, or ingestor not subscribed). Between REST polls this naturally ramps up until the next snapshot resets it. |
| **Raw receive lag latest ms** | `wall_clock_at_receive − event.timestamp_ms` on the **last** raw event | **Exchange-timestamp vs when we received it.** Captures API latency, network delay, and clock skew. Spikes on each REST poll are normal; a persistently high floor suggests slow venue responses or timestamp drift. Updates only when a new raw event arrives. |
| **Raw event gap latest ms** | `current_event.timestamp_ms − previous_event.timestamp_ms` (per exchange/symbol/kind) | **Spacing between consecutive upstream timestamps.** For REST orderbooks, expect gaps ≈ `poll_interval_ms`. For trades, smaller irregular gaps during active tape; **sudden jumps** can mean missed polls or an API gap. Does not measure wall-clock silence by itself. |

**Healthy patterns:** freshness rises and falls with your poll cadence; receive lag stays within a stable band; orderbook gaps track `poll_interval_ms` from the market-feed config.

**Warning signs:** freshness monotonically increasing for minutes; receive lag far above poll interval + network budget; orderbook gaps many multiples of `poll_interval_ms` without an explanation (outage, rate limit, feed disabled)."""


def market_data_quality() -> dict[str, Any]:
    panels = [
        _text(
            1,
            "Scope",
            "Per-exchange prices/volumes from persisted `MarketEvent` rows (post-aggregation). "
            "Raw ingest health is covered below and in the three Prometheus panels. "
            "Use **Price series** / **Volume series** to toggle bid, ask, VWAP, and volume lines.",
            y=0,
        ),
        _text(2, "Raw feed metrics guide", _raw_feed_metrics_guide_markdown(), y=3, h=9),
        _panel(
            3,
            "Raw feed freshness ms",
            "timeseries",
            PROM_DS,
            '(time() * 1000) - max by (exchange,symbol,kind)(mm_raw_market_event_timestamp_ms{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"})',
            x=0,
            y=12,
            w=8,
            unit="ms",
            legend="{{exchange}} {{kind}}",
            description="Staleness of the latest upstream `timestamp_ms` at scrape time. Rises between polls; climbing without resets means the feed stopped updating.",
        ),
        _panel(
            4,
            "Raw receive lag latest ms",
            "timeseries",
            PROM_DS,
            'max by (exchange,symbol,kind)(mm_raw_market_ingest_lag_latest_ms{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"})',
            x=8,
            y=12,
            w=8,
            unit="ms",
            legend="{{exchange}} {{kind}}",
            description="Wall-clock lag recorded when the last raw event arrived: `receive_time − event.timestamp_ms`. Reflects API/network latency and clock skew.",
        ),
        _panel(
            5,
            "Raw event gap latest ms",
            "timeseries",
            PROM_DS,
            'max by (exchange,symbol,kind)(mm_raw_market_event_gap_latest_ms{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"})',
            x=16,
            y=12,
            w=8,
            unit="ms",
            legend="{{exchange}} {{kind}}",
            description="Timestamp delta between the last two raw events for the same exchange/symbol/kind. Orderbook gaps should track `poll_interval_ms`.",
        ),
        _panel(6, "Best bid / ask / VWAP by exchange", "timeseries", PG_DS, _exchange_price_vwap_sql(), x=0, y=20, w=24),
        _panel(8, "Buy / sell volume by exchange", "timeseries", PG_DS, _exchange_volume_sql(), x=0, y=28, w=24),
    ]
    return _dashboard(
        "mm-obs-v2-market-data",
        "MM Observability v2 / 01 Market Data Quality",
        panels,
        tags=["market-data"],
        extra_templating=[
            _custom_var_multi(
                "price_series",
                "Price series",
                "best_bid,best_ask,buy_vwap,sell_vwap,trade_vwap",
                default="best_bid,best_ask,buy_vwap,sell_vwap",
            ),
            _custom_var_multi(
                "volume_series",
                "Volume series",
                "buy_volume,sell_volume",
            ),
        ],
    )


def orderbook_depth() -> dict[str, Any]:
    panels = [
        _text(
            1,
            "Scope",
            "Per-**Exchange** orderbook and MPC ladder views for the selected diagnostics frame. "
            "Depth curves use `price - mid` on the x-axis. **Marginal ladder** plots one point per "
            "`orderbook_grid` rung with positive marginal qty (at most `ceil(spread_max / spread_step)` per side).",
            y=0,
            h=4,
        ),
        _xy_depth_panel(2, "Orderbook depth", _orderbook_depth_sql("raw", role=None), x=0, y=4, w=24),
        _xy_depth_panel(3, "Clean orderbook depth", _orderbook_depth_sql("clean", role=None), x=0, y=12, w=24),
        _xy_depth_panel(4, "Action ladder depth", _latest_action_pipeline_stage_depth_sql("ladder", role=None), x=0, y=20, w=24),
        _stem_order_qty_price_panel(
            5,
            "Marginal ladder",
            _latest_action_pipeline_stage_orders_sql("ladder", role=None),
            x=0,
            y=28,
            w=24,
            description="Marginal ladder orders on the controller spread grid (pre-cap) for the selected exchange.",
        ),
        _stem_order_qty_price_panel(
            6,
            "After venue rounding",
            _latest_action_pipeline_stage_orders_sql("rules_valid", role=None),
            x=0,
            y=38,
            w=12,
        ),
        _stem_order_qty_price_panel(
            7,
            "Exchange-accepted placements",
            _latest_action_pipeline_stage_orders_sql("router_accepted", role=None),
            x=12,
            y=38,
            w=12,
        ),
    ]
    return _dashboard(
        "mm-obs-v2-orderbook-depth",
        "MM Observability v2 / 01B Orderbook Depth",
        panels,
        tags=["orderbook"],
        extra_templating=[
            _custom_var("spread_max", "Spread max", "2000,5000,10000,20000,500,200,100"),
            _custom_var("target_frame_mode", "Target frame mode", "latest,pinned"),
            _query_var("target_frame_event_id", "Target frame", _target_stems_frame_query(), refresh=2),
        ],
    )


def controller_health() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Controller frame freshness, readiness, warnings, and stage timings.", y=0),
        _panel(2, "Frame age", "timeseries", PROM_DS, 'mm_frame_age_ms{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"}', x=0, y=3, legend="{{exchange}} {{symbol}}"),
        _panel(3, "Frame readiness", "timeseries", PROM_DS, 'mm_frame_sync_ready{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"}', x=12, y=3, legend="{{field}} {{exchange}} {{symbol}}"),
        _panel(4, "Stage latency / step", "timeseries", PG_DS, _controller_stage_step_sql(), x=0, y=11, unit="ms"),
        _panel(5, "Warnings / step", "timeseries", PG_DS, _controller_warning_step_sql(), x=12, y=11),
        _panel(6, "Recent diagnostics warnings", "table", PG_DS, _latest_warnings_sql(limit=100), x=0, y=19, w=24, fmt="table"),
    ]
    return _dashboard("mm-obs-v2-controller-health", "MM Observability v2 / 02 Frame And Controller Health", panels, tags=["controller"])


def pipeline_time_profile() -> dict[str, Any]:
    panels = [
        _text(
            1,
            "Scope",
            "Per-frame stage durations plus cumulative milestone offsets. "
            "Cumulative charts anchor at local orderbook timestamp or frame close (`t_end_ms`) and plot "
            "when each milestone occurred relative to that anchor. "
            "Strategy milestones start at diagnostics `created_at_ms` (StrategyRunner entry) and add "
            "monotonic stage timings; `+pipeline_end` is `created_at_ms + controller_total_ms`. "
            "Raw adapter receive lag stays on Prometheus because raw market events are not persisted by default.",
            y=0,
        ),
        _panel(
            2,
            "End-to-end pipeline latency / frame",
            "timeseries",
            PG_DS,
            _pipeline_profile_sql(),
            x=0,
            y=3,
            w=24,
            h=10,
            unit="ms",
        ),
        _pipeline_cumulative_panel(
            6,
            "Cumulative pipeline offset vs orderbook timestamp",
            _pipeline_cumulative_profile_sql(anchor="orderbook"),
            x=0,
            y=13,
        ),
        _pipeline_cumulative_panel(
            7,
            "Cumulative pipeline offset vs frame close (t_end_ms)",
            _pipeline_cumulative_profile_sql(anchor="t_end"),
            x=0,
            y=23,
        ),
        _panel(
            3,
            "Raw feed receive lag latest",
            "timeseries",
            PROM_DS,
            'max by (exchange,symbol,kind)(mm_raw_market_ingest_lag_latest_ms{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}",kind="orderbook"})',
            x=0,
            y=33,
            w=12,
            unit="ms",
            legend="{{exchange}} {{symbol}} {{kind}}",
        ),
        _panel(
            4,
            "Optimizer substage latency / frame",
            "timeseries",
            PG_DS,
            _optimizer_substage_profile_sql(),
            x=12,
            y=33,
            w=12,
            unit="ms",
        ),
        _panel(
            5,
            "Slowest recent frames",
            "table",
            PG_DS,
            _slowest_frames_sql(),
            x=0,
            y=41,
            w=24,
            h=9,
            fmt="table",
            unit="ms",
        ),
    ]
    return _dashboard(
        "mm-obs-v2-pipeline-time-profile",
        "MM Observability v2 / 02B Pipeline Time Profile",
        panels,
        tags=["pipeline", "latency"],
    )


def model_quality() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Model NLL and latest model scores emitted from `ControllerDiagnostics.models`.", y=0),
        _panel(2, "Model NLL / step", "timeseries", PG_DS, _model_nll_step_sql(), x=0, y=3, w=24),
        _panel(3, "Latest model scores", "table", PG_DS, _latest_model_scores_sql(), x=0, y=11, w=24, fmt="table"),
    ]
    return _dashboard("mm-obs-v2-model-quality", "MM Observability v2 / 03 Strategy Model Quality", panels, tags=["models"])


def model_parameters() -> dict[str, Any]:
    formula_note = (
        "### Conditional mechanical-mixture price model\n\n"
        "`q_market(t) = sigmoid(gate_beta[0] + gate_beta[1] * log1p(total_flow_t))`\n\n"
        "For each component `c in {mid, log_spread, log_buy_shape, log_sell_shape}`:\n\n"
        "`market_mean_c = market_beta.c[0] + market_beta.c[1] * mechanical_delta_c`\n\n"
        "`refresh_mean_c = refresh_mean.c`\n\n"
        "`component_delta_c ~ q_market * N(market_mean_c, market_sigma.c^2) + "
        "(1 - q_market) * StudentT(refresh_mean_c, refresh_sigma.c, student_df)`\n\n"
        "`mid_t = mid_{t-1} + delta_mid`\n\n"
        "`spread_t = spread_{t-1} * exp(delta_log_spread)`\n\n"
        "`buy_shape_t = buy_shape_{t-1} * exp(delta_log_buy_shape)`\n\n"
        "`sell_shape_t = sell_shape_{t-1} * exp(delta_log_sell_shape)`\n\n"
        "`buy_vwap(depth) = mid_t + spread_t / 2 + buy_shape_t * depth_ratio(depth, reference_vwap_vol, shape_alpha)`\n\n"
        "`sell_vwap(depth) = mid_t - spread_t / 2 - sell_shape_t * depth_ratio(depth, reference_vwap_vol, shape_alpha)`\n\n"
        "Shapes are defined from the fixed reference depth: "
        "`buy_shape = buy_vwap_ref - mid - spread/2`, "
        "`sell_shape = mid - sell_vwap_ref - spread/2`.\n\n"
        "### Market-flow models\n\n"
        "Market-flow params live under `local_market_flow_model.params` and "
        "`global_market_flow_model.params`. For hurdle/lognormal-style models, read "
        "`p_*`/gate parameters as event probability controls, `mu_*` as log-size means, "
        "and `sigma_*` as log-size uncertainty."
    )
    panels = [
        _text(
            1,
            "Scope",
            "Strategy model parameter time series from `StrategyDecision` snapshots. Nested JSON params are flattened recursively, so arrays and objects become stable metric names. Model NLL is included from `ControllerDiagnostics.models` for quick parameter-vs-score inspection.",
            y=0,
        ),
        _text(2, "Formula Reminder", formula_note, y=3, h=13),
        _panel(
            3,
            "Latest model snapshot",
            "table",
            PG_DS,
            _latest_strategy_model_params_sql(),
            x=0,
            y=16,
            w=24,
            h=7,
            fmt="table",
        ),
        _panel(
            4,
            "Model NLL / step",
            "timeseries",
            PG_DS,
            _model_nll_step_sql(),
            x=0,
            y=23,
            w=24,
            h=8,
            decimals=4,
        ),
        _row(5, "Conditional Price Models", y=31),
        _panel(
            6,
            "Local price gate",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(
                scope="local",
                model_type="price",
                param_regex=r"\.(market_prob|market_resp)$|\.gate_beta\[",
            ),
            x=0,
            y=32,
            w=12,
            h=8,
            decimals=4,
        ),
        _panel(
            7,
            "Global price gate",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(
                scope="global",
                model_type="price",
                param_regex=r"\.(market_prob|market_resp)$|\.gate_beta\[",
            ),
            x=12,
            y=32,
            w=12,
            h=8,
            decimals=4,
        ),
        _panel(
            8,
            "Local price coefficients and means",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(
                scope="local",
                model_type="price",
                param_regex=r"\.(market_beta|refresh_mean|mu_|beta|mean)",
                exclude_regex=r"\.(market_prob|market_resp)$",
            ),
            x=0,
            y=40,
            w=12,
            h=8,
            decimals=4,
        ),
        _panel(
            9,
            "Global price coefficients and means",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(
                scope="global",
                model_type="price",
                param_regex=r"\.(market_beta|refresh_mean|mu_|beta|mean)",
                exclude_regex=r"\.(market_prob|market_resp)$",
            ),
            x=12,
            y=40,
            w=12,
            h=8,
            decimals=4,
        ),
        _panel(
            10,
            "Local price uncertainty",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(
                scope="local",
                model_type="price",
                param_regex=r"\.(market_sigma|refresh_sigma|sigma_|var|variance|std)",
            ),
            x=0,
            y=48,
            w=12,
            h=8,
            decimals=4,
        ),
        _panel(
            11,
            "Global price uncertainty",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(
                scope="global",
                model_type="price",
                param_regex=r"\.(market_sigma|refresh_sigma|sigma_|var|variance|std)",
            ),
            x=12,
            y=48,
            w=12,
            h=8,
            decimals=4,
        ),
        _row(12, "Market Flow Models", y=56),
        _panel(
            13,
            "Local market-flow params",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(scope="local", model_type="market_flow"),
            x=0,
            y=57,
            w=12,
            h=8,
            decimals=4,
        ),
        _panel(
            14,
            "Global market-flow params",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(scope="global", model_type="market_flow"),
            x=12,
            y=57,
            w=12,
            h=8,
            decimals=4,
        ),
        _row(15, "All Flattened Params", y=65),
        _panel(
            16,
            "Local price scale/config",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(
                scope="local",
                model_type="price",
                param_regex=_PRICE_SCALE_CONFIG_PARAM_REGEX,
            ),
            x=0,
            y=66,
            w=12,
            h=7,
            decimals=4,
        ),
        _panel(
            17,
            "Global price scale/config",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(
                scope="global",
                model_type="price",
                param_regex=_PRICE_SCALE_CONFIG_PARAM_REGEX,
            ),
            x=12,
            y=66,
            w=12,
            h=7,
            decimals=4,
        ),
        _panel(
            18,
            "All local params",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(scope="local", exclude_regex=_PRICE_SCALE_CONFIG_PARAM_REGEX),
            x=0,
            y=73,
            w=12,
            h=10,
            decimals=4,
        ),
        _panel(
            19,
            "All global params",
            "timeseries",
            PG_DS,
            _strategy_model_params_sql(scope="global", exclude_regex=_PRICE_SCALE_CONFIG_PARAM_REGEX),
            x=12,
            y=73,
            w=12,
            h=10,
            decimals=4,
        ),
    ]
    return _dashboard(
        "mm-obs-v2-model-parameters",
        "MM Observability v2 / 03B Model Parameters",
        panels,
        tags=["models", "parameters"],
    )


def optimizer_diagnostics() -> dict[str, Any]:
    panels = [
        _text(
            1,
            "Scope",
            "Optimizer objective J, weighted-term contribution fractions (CE, book quality, inventory zone), and timing.",
            y=0,
        ),
        _panel(2, "Objective value", "timeseries", PG_DS, _optimizer_objective_sql(), x=0, y=3, w=24, unit="sci", decimals=4),
        _panel(
            3,
            "Contribution fractions",
            "timeseries",
            PG_DS,
            _optimizer_contrib_sql(),
            x=0,
            y=11,
            w=12,
            unit="percentunit",
            decimals=2,
        ),
        _panel(4, "Optimizer timings", "timeseries", PG_DS, _optimizer_timing_sql(), x=12, y=11, w=12),
        _panel(
            5,
            "Objective components (weighted J terms)",
            "timeseries",
            PG_DS,
            _optimizer_weighted_components_sql(),
            x=0,
            y=19,
            w=12,
            unit="sci",
            decimals=4,
        ),
        _panel(
            6,
            "Book quality breakdown",
            "timeseries",
            PG_DS,
            _optimizer_book_breakdown_sql(),
            x=12,
            y=19,
            w=12,
            unit="percentunit",
            decimals=2,
        ),
        _panel(
            7,
            "Inventory zone breakdown",
            "timeseries",
            PG_DS,
            _optimizer_inventory_zone_breakdown_sql(),
            x=0,
            y=27,
            w=12,
            decimals=4,
        ),
        _panel(
            8,
            "Inventory zone config",
            "timeseries",
            PG_DS,
            _optimizer_inventory_zone_config_sql(),
            x=12,
            y=27,
            w=12,
            decimals=4,
        ),
    ]
    return _dashboard("mm-obs-v2-optimizer", "MM Observability v2 / 04 Optimizer Diagnostics", panels, tags=["optimizer"])


def target_order_stems() -> dict[str, Any]:
    stem_h = 7
    stem_gap = 7
    stages = [
        ("01 Strategy raw ladder", "ladder", "orders", None, False),
        ("02 Strategy cap -> target orderbook", "capped", "orders", None, False),
        ("03 Risk-approved target", "risk_approved_target", "orders", None, False),
        ("04 Planner new place orders", "planned_place", "orders", None, False),
        ("05 Venue rules valid after rounding", "rules_valid", "orders", None, False),
        ("06 Exchange accepted placed orders", "router_accepted", "orders", None, False),
        ("07 Planner kept live orders", "kept", "orders", None, False),
        ("08 Planner cancel orders", "planned_cancel", "orders", None, False),
        ("09 Removed by risk / collateral", "risk_approved_target", "rejections", None, True),
        ("10 Removed by venue adapter: invalid price/qty", "rules_valid", "rejections", "rules_invalid_price_or_qty", False),
        ("11 Removed by venue adapter: min qty", "rules_valid", "rejections", "rules_min_qty", False),
        ("12 Removed by venue adapter: min notional", "rules_valid", "rejections", "rules_min_notional", False),
    ]
    panels = [
        _text(
            1,
            "Scope",
            "Live or pinned strategy-to-execution ladder, split local left and global right. Latest mode follows the newest ControllerDiagnostics event; pinned mode uses the selected target frame. Every chart uses its venue raw-ladder price/qty frame; bid is green, ask is red, and hidden black axis anchors keep scales aligned.",
            y=0,
        ),
    ]
    panel_id = 2
    for row, (title, stage, field, reason, include_reason) in enumerate(stages):
        y = 3 + stem_gap * row
        for role, x in (("local", 0), ("global", 12)):
            role_title = title
            if title == "09 Removed by risk / collateral":
                role_title = f"09 Removed by {role} risk / collateral"
            panels.append(
                _stem_order_qty_price_panel(
                    panel_id,
                    f"{role.title()} {role_title}",
                    _latest_action_pipeline_stage_orders_sql(
                        stage,
                        field=field,
                        reason=reason,
                        include_reason_in_series=include_reason,
                        role=role,
                    ),
                    x=x,
                    y=y,
                    w=12,
                    h=stem_h,
                )
            )
            panel_id += 1
    return _dashboard(
        "mm-obs-v2-target-order-stems",
        "MM Observability v2 / 05C Target Order Stems",
        panels,
        tags=["actions", "execution"],
        extra_templating=[
            _custom_var("target_frame_mode", "Target frame mode", "latest,pinned"),
            _query_var("target_frame_event_id", "Target frame", _target_stems_frame_query(), refresh=2),
        ],
    )


def action_pipeline_side_counts() -> dict[str, Any]:
    pipeline_stages = (
        ("raw ladder", "ladder"),
        ("strategy capped", "capped"),
        ("risk approved", "risk_approved_target"),
        ("venue valid", "rules_valid"),
    )
    placement_stages = (
        ("planner placed", "planned_place"),
        ("exchange accepted", "router_accepted"),
    )
    live_stages = (
        ("kept live", "kept"),
        ("planner canceled", "planned_cancel"),
    )
    panels = [
        _text(
            1,
            "Scope",
            "Bid and ask order counts per controller step through the strategy, risk, planner, venue-adapter, and exchange-acceptance path.",
            y=0,
        ),
        _panel(2, "Bid/ask exchange accepted placement counts / step", "timeseries", PG_DS, _action_pipeline_placement_side_summary_sql("count", "router_accepted", "accepted placement"), x=0, y=3, w=24),
        _panel(3, "Bid/ask exchange accepted placement volume / step", "timeseries", PG_DS, _action_pipeline_placement_side_summary_sql("qty", "router_accepted", "accepted placement"), x=0, y=11, w=24),
        _panel(4, "Bid/ask planner placement counts / step", "timeseries", PG_DS, _action_pipeline_placement_side_summary_sql("count", "planned_place"), x=0, y=19, w=24),
        _panel(5, "Bid/ask planner placement volume / step", "timeseries", PG_DS, _action_pipeline_placement_side_summary_sql("qty", "planned_place"), x=0, y=27, w=24),
        _panel(6, "Bid target survival counts / step", "timeseries", PG_DS, _action_pipeline_side_stage_counts_sql("bid", pipeline_stages), x=0, y=35),
        _panel(7, "Ask target survival counts / step", "timeseries", PG_DS, _action_pipeline_side_stage_counts_sql("ask", pipeline_stages), x=12, y=35),
        _panel(8, "Bid placement counts / step", "timeseries", PG_DS, _action_pipeline_side_stage_counts_sql("bid", placement_stages), x=0, y=43),
        _panel(9, "Ask placement counts / step", "timeseries", PG_DS, _action_pipeline_side_stage_counts_sql("ask", placement_stages), x=12, y=43),
        _panel(10, "Bid live/cancel counts / step", "timeseries", PG_DS, _action_pipeline_side_stage_counts_sql("bid", live_stages), x=0, y=51),
        _panel(11, "Ask live/cancel counts / step", "timeseries", PG_DS, _action_pipeline_side_stage_counts_sql("ask", live_stages), x=12, y=51),
        _panel(12, "Bid removed by reason / step", "timeseries", PG_DS, _action_pipeline_side_removed_reason_counts_sql("bid"), x=0, y=59),
        _panel(13, "Ask removed by reason / step", "timeseries", PG_DS, _action_pipeline_side_removed_reason_counts_sql("ask"), x=12, y=59),
    ]
    return _dashboard(
        "mm-obs-v2-action-pipeline-side-counts",
        "MM Observability v2 / 05D Action Pipeline Side Counts",
        panels,
        tags=["actions", "execution"],
    )


def action_pipeline() -> dict[str, Any]:
    panels = [
        _text(
            1,
            "Scope",
            "Chart view of the strategy target: target ladder levels, frame order flow, and why target orders did not become executable commands.",
            y=0,
        ),
        _panel(2, "Target orders / step", "timeseries", PG_DS, _target_book_count_sql(), x=0, y=3, w=12),
        _panel(3, "Target quantity / step", "timeseries", PG_DS, _target_book_qty_sql(), x=12, y=3, w=12),
        _panel(4, "Target ladder prices / step", "timeseries", PG_DS, _target_ladder_level_sql("price"), x=0, y=11, w=12),
        _panel(5, "Target ladder qty / step", "timeseries", PG_DS, _target_ladder_level_sql("qty"), x=12, y=11, w=12),
        _panel(6, "Frame order flow / step", "timeseries", PG_DS, _action_frame_order_flow_sql(), x=0, y=19, w=24),
        _panel(7, "Local orders removed before target / step", "timeseries", PG_DS, _local_stage_drop_sql(), x=0, y=27, w=12),
        _panel(8, "Global risk rejects / step", "timeseries", PG_DS, _role_rejection_reason_sql("global"), x=12, y=27, w=12),
        _panel(9, "Local gate blocks / frame", "timeseries", PG_DS, _role_gate_block_reason_sql("local"), x=0, y=35, w=12),
        _panel(10, "Global gate blocks / frame", "timeseries", PG_DS, _role_gate_block_reason_sql("global"), x=12, y=35, w=12),
    ]
    return _dashboard("mm-obs-v2-action-pipeline", "MM Observability v2 / 05 Action Pipeline Funnel", panels, tags=["actions"])


def risk_execution() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Chart view of execution: what was sent to the exchange, what was accepted, what was canceled, what was rejected, and what filled.", y=0),
        _panel(2, "Order flow / step", "timeseries", PG_DS, _operator_order_flow_step_sql(), x=0, y=3, w=24),
        _panel(3, "Sync plan counts / step", "timeseries", PG_DS, _execution_sync_counts_step_sql(), x=0, y=11, w=24),
        _panel(4, "Orders sent to exchange / step", "timeseries", PG_DS, _execution_order_payload_step_sql("PlaceOrdersRequested"), x=0, y=19, w=12),
        _panel(5, "Orders accepted by exchange / step", "timeseries", PG_DS, _execution_order_payload_step_sql("PlaceOrdersAccepted"), x=12, y=19, w=12),
        _panel(6, "Canceled order ids / step", "timeseries", PG_DS, _runtime_order_id_event_sql("CancelOrdersAccepted"), x=0, y=27, w=12),
        _panel(7, "Rejected order commands / step", "timeseries", PG_DS, _rejected_order_commands_step_sql(), x=12, y=27, w=12),
        _panel(8, "Fill events / step", "timeseries", PG_DS, _runtime_fill_rate_sql("count"), x=0, y=35, w=12),
        _panel(9, "Venue gate status / step", "timeseries", PG_DS, _venue_gate_status_step_sql(), x=12, y=35, w=12),
        _panel(10, "Risk reject reasons / step", "timeseries", PG_DS, _risk_reject_step_sql(), x=0, y=43, w=12),
        _panel(11, "Gate block reasons / step", "timeseries", PG_DS, _execution_gate_block_step_sql(), x=12, y=43, w=12),
        _panel(12, "Router failures / step", "timeseries", PG_DS, _router_failure_step_sql(), x=0, y=51, w=12),
    ]
    return _dashboard("mm-obs-v2-risk-execution", "MM Observability v2 / 06 Risk And Execution", panels, tags=["risk", "execution"])


def execution_errors() -> dict[str, Any]:
    panels = [
        _text(
            1,
            "Scope",
            "Structured execution failures from persisted events: adapter/place/cancel failures, validation rejects, and sync failures. Use the tables to inspect exact exchange messages.",
            y=0,
        ),
        _panel(2, "Execution error totals", "stat", PG_DS, _execution_error_totals_sql(), x=0, y=3, w=8, h=5),
        _panel(3, "Errors by exchange", "stat", PG_DS, _execution_error_by_exchange_sql(), x=8, y=3, w=8, h=5),
        _panel(4, "Execution error flow / step", "timeseries", PG_DS, _execution_error_flow_sql(), x=0, y=8, w=24),
        _panel(5, "Errors by reason / step", "timeseries", PG_DS, _execution_error_reason_step_sql(), x=0, y=16, w=24),
        _panel(6, "Grouped exchange error messages", "table", PG_DS, _execution_error_message_table_sql(), x=0, y=24, w=24, h=9, fmt="table"),
        _panel(7, "Recent execution errors", "table", PG_DS, _recent_execution_errors_sql(), x=0, y=33, w=24, h=11, fmt="table"),
        _panel(
            8,
            "Runtime error logs",
            "logs",
            LOKI_DS,
            '{job="market-maker-runtime"} | json |~ "PlaceOrdersFailed|CancelOrdersFailed|OrderValidationRejected|ExecutionSyncFailed|adapter_failure|failed|error"',
            x=0,
            y=44,
            w=24,
            h=8,
        ),
    ]
    return _dashboard(
        "mm-obs-v2-execution-errors",
        "MM Observability v2 / 06B Execution Errors",
        panels,
        tags=["execution", "errors"],
    )


def portfolio_pnl() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Inventory, cash, total marked value, wealth, and step economics.", y=0),
        _panel(2, "Inventory", "timeseries", PG_DS, _portfolio_field_sql("local_inventory", "global_inventory"), x=0, y=3),
        _panel(3, "Cash", "timeseries", PG_DS, _portfolio_field_sql("local_cash", "global_cash"), x=12, y=3),
        _panel(4, "Wealth", "timeseries", PG_DS, _portfolio_field_sql("local_wealth", "global_wealth"), x=0, y=11),
        _panel(5, "Total cash / inventory value / wealth", "timeseries", PG_DS, _portfolio_total_value_sql(), x=12, y=11),
        _panel(6, "Local cash / inventory value / wealth", "timeseries", PG_DS, _portfolio_venue_value_sql("local"), x=0, y=19),
        _panel(7, "Global cash / inventory value / wealth", "timeseries", PG_DS, _portfolio_venue_value_sql("global"), x=12, y=19),
        _panel(
            8,
            "Cumulative net spread capture PnL",
            "timeseries",
            PG_DS,
            _portfolio_cumulative_pnl_component_sql("net_spread_capture"),
            x=0,
            y=27,
            w=24,
        ),
        _panel(
            9,
            "Cumulative inventory drift PnL",
            "timeseries",
            PG_DS,
            _portfolio_cumulative_pnl_component_sql("inventory_drift"),
            x=0,
            y=35,
            w=24,
        ),
        _panel(
            10,
            "Cumulative adverse selection PnL",
            "timeseries",
            PG_DS,
            _portfolio_cumulative_pnl_component_sql("adverse_selection"),
            x=0,
            y=43,
            w=24,
        ),
    ]
    return _dashboard("mm-obs-v2-portfolio-pnl", "MM Observability v2 / 07 Portfolio And PnL", panels, tags=["portfolio"])


def fill_spread_pnl() -> dict[str, Any]:
    panels = [
        _text(
            1,
            "Scope",
            "Private fill totals over the selected time range. Average prices are quantity-weighted; spread PnL uses paired bid/ask volume times average spread.",
            y=0,
        ),
        _panel(2, "Total filled bid volume", "stat", PG_DS, _fill_interval_stat_sql("bid_volume"), x=0, y=3, w=6, h=5),
        _panel(3, "Local filled bid volume", "stat", PG_DS, _fill_interval_stat_sql("bid_volume", role="local"), x=6, y=3, w=6, h=5),
        _panel(4, "Global filled bid volume", "stat", PG_DS, _fill_interval_stat_sql("bid_volume", role="global"), x=12, y=3, w=6, h=5),
        _panel(5, "Average bid fill price", "stat", PG_DS, _fill_interval_stat_sql("bid_avg_price"), x=18, y=3, w=6, h=5, unit="currency"),
        _panel(6, "Total filled ask volume", "stat", PG_DS, _fill_interval_stat_sql("ask_volume"), x=0, y=8, w=6, h=5),
        _panel(7, "Local filled ask volume", "stat", PG_DS, _fill_interval_stat_sql("ask_volume", role="local"), x=6, y=8, w=6, h=5),
        _panel(8, "Global filled ask volume", "stat", PG_DS, _fill_interval_stat_sql("ask_volume", role="global"), x=12, y=8, w=6, h=5),
        _panel(9, "Average ask fill price", "stat", PG_DS, _fill_interval_stat_sql("ask_avg_price"), x=18, y=8, w=6, h=5, unit="currency"),
        _panel(10, "Spread PnL", "stat", PG_DS, _fill_interval_stat_sql("spread_pnl"), x=0, y=13, w=6, h=5, unit="currency"),
        _panel(11, "Total fill fee", "stat", PG_DS, _fill_interval_stat_sql("total_fee"), x=6, y=13, w=6, h=5, unit="currency"),
        _panel(12, "Total delta cash", "stat", PG_DS, _portfolio_interval_delta_sql("delta_cash"), x=0, y=18, w=6, h=5, unit="currency"),
        _panel(13, "Total delta inventory", "stat", PG_DS, _portfolio_interval_delta_sql("delta_inventory"), x=6, y=18, w=6, h=5),
        _panel(14, "Total delta wealth", "stat", PG_DS, _portfolio_interval_delta_sql("delta_wealth"), x=12, y=18, w=6, h=5, unit="currency"),
        _panel(15, "Total delta ref price", "stat", PG_DS, _portfolio_interval_delta_sql("delta_price"), x=18, y=18, w=6, h=5, unit="currency"),
        _panel(16, "Latest total cash", "stat", PG_DS, _portfolio_latest_total_sql("cash"), x=0, y=23, w=6, h=5, unit="currency"),
        _panel(17, "Latest total inventory", "stat", PG_DS, _portfolio_latest_total_sql("inventory"), x=6, y=23, w=6, h=5),
    ]
    return _dashboard(
        "mm-obs-v2-fill-spread-pnl",
        "MM Observability v2 / 07B Fill Spread PnL",
        panels,
        tags=["portfolio", "fills"],
    )


def liquidity_sourcing() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Local/global readiness and joint target-book routing.", y=0),
        _panel(2, "Readiness flags", "timeseries", PROM_DS, 'mm_frame_sync_ready{exchange=~"${exchange:regex}",symbol=~"${symbol:regex}"}', x=0, y=3, legend="{{field}} {{exchange}} {{symbol}}"),
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
  AND e.exchange = ANY(ARRAY[${exchange:sqlstring}]::text[])
  AND e.symbol = ANY(ARRAY[${symbol:sqlstring}]::text[])
  AND e.actor_id = ANY(ARRAY[${actor_id:sqlstring}]::text[])
  AND ($__timeFilter(e.ts))"""


def _latest_warnings_sql(limit: int = 25) -> str:
    return f"""SELECT e.ts, e.actor_id, e.exchange, e.symbol, w.warning
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(e.payload->'warnings', '[]'::jsonb)) AS w(warning)
WHERE {_diagnostics_filter()}
ORDER BY e.ts DESC
LIMIT {limit}"""


def _orderbook_depth_sql(kind: str, *, role: str | None = "local") -> str:
    if role is not None and role not in {"local", "global"}:
        raise ValueError(f"unknown orderbook depth role: {role}")
    if kind == "raw":
        raw_bid_price = _numeric_sql("m.payload #>> ARRAY['orderbook','bids','0','price']", default="NULL")
        raw_ask_price = _numeric_sql("m.payload #>> ARRAY['orderbook','asks','0','price']", default="NULL")
        raw_t_end_ms = _numeric_sql("m.payload->>'t_end_ms'", default="NULL")
        selected = f"""selected AS (
  SELECT COALESCE(
           (
             {raw_bid_price} +
             {raw_ask_price}
           ) / 2.0,
           NULLIF(ctx.mid, 0.0),
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
      AND ({raw_t_end_ms}) <= ctx.t_end_ms
      AND $__timeFilter(m.ts)
    ORDER BY ({raw_t_end_ms}) DESC, m.ts DESC
    LIMIT 1
  ) m ON TRUE
)"""
        bid_levels = f"""bid_levels AS (
  SELECT {_numeric_sql("b->>'price'", default="NULL")} AS price,
         {_numeric_sql("b->>'qty'")} AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'orderbook'->'bids', '[]'::jsonb)) AS b
)"""
        ask_levels = f"""ask_levels AS (
  SELECT {_numeric_sql("a->>'price'", default="NULL")} AS price,
         {_numeric_sql("a->>'qty'")} AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'orderbook'->'asks', '[]'::jsonb)) AS a
)"""
    elif kind == "clean":
        clean_payload_path = "global_cleared_orderbook" if role == "global" else "cleared_orderbook"
        selected = """selected AS (
  SELECT ctx.mid, ctx.spread_max, ctx.diag_payload AS payload
  FROM ctx
)"""
        bid_levels = f"""bid_levels AS (
  SELECT {_numeric_sql("b->>'price'", default="NULL")} AS price,
         {_numeric_sql("b->>'qty'")} AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'{clean_payload_path}'->'bids', '[]'::jsonb)) AS b
)"""
        ask_levels = f"""ask_levels AS (
  SELECT {_numeric_sql("a->>'price'", default="NULL")} AS price,
         {_numeric_sql("a->>'qty'")} AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'{clean_payload_path}'->'asks', '[]'::jsonb)) AS a
)"""
    elif kind == "ladder":
        decision_mid = _numeric_sql("d.payload->>'mid'", default="NULL")
        decision_t_end_ms = _numeric_sql("d.payload->>'t_end_ms'", default="NULL")
        selected = f"""selected AS (
  SELECT COALESCE(
           NULLIF(ctx.mid, 0.0),
           {decision_mid},
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
      AND ({decision_t_end_ms}) <= ctx.t_end_ms
      AND $__timeFilter(d.ts)
    ORDER BY ({decision_t_end_ms}) DESC, d.ts DESC
    LIMIT 1
  ) d ON TRUE
)"""
        bid_levels = f"""bid_levels AS (
  SELECT {_numeric_sql("o->>'price'", default="NULL")} AS price,
         {_numeric_sql("o->>'qty'")} AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'ladder_orders', '[]'::jsonb)) AS o
  WHERE lower(trim(o->>'side')) = 'buy'
)"""
        ask_levels = f"""ask_levels AS (
  SELECT {_numeric_sql("o->>'price'", default="NULL")} AS price,
         {_numeric_sql("o->>'qty'")} AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'ladder_orders', '[]'::jsonb)) AS o
  WHERE lower(trim(o->>'side')) = 'sell'
)"""
    else:
        raise ValueError(f"unknown orderbook depth kind: {kind}")
    return f"""WITH {_orderbook_context_cte(role)},
{selected},
{bid_levels},
{ask_levels},
{_orderbook_depth_plot_ctes()}
SELECT 'bid'::text AS side,
       spread,
       value
FROM bid_step_plot
UNION ALL
SELECT 'ask'::text AS side,
       spread,
       value
FROM ask_step_plot
ORDER BY side, spread"""


def _orderbook_context_cte(role: str | None = "local") -> str:
    t_end_ms = _numeric_sql("e.payload->>'t_end_ms'")
    if role == "global":
        state_mid_path = "global_mid"
        clean_payload_path = "global_cleared_orderbook"
        exchange_expr = "gates.key"
        role_join = """
  LEFT JOIN LATERAL (
    SELECT gates.key
    FROM jsonb_each(COALESCE(e.payload #> ARRAY['execution_gates','venue_gates'], '{}'::jsonb)) AS gates(key, value)
    WHERE gates.value->>'role' = 'global'
    LIMIT 1
  ) gates ON TRUE"""
        role_where = " AND gates.key IS NOT NULL"
    else:
        state_mid_path = "mid"
        clean_payload_path = "cleared_orderbook"
        exchange_expr = "e.exchange"
        role_join = ""
        role_where = ""
    state_mid = _numeric_sql(f"e.payload #>> ARRAY['state','{state_mid_path}']", default="NULL")
    clean_bid = _numeric_sql(f"e.payload #>> ARRAY['{clean_payload_path}','bids','0','price']", default="NULL")
    clean_ask = _numeric_sql(f"e.payload #>> ARRAY['{clean_payload_path}','asks','0','price']", default="NULL")
    spread_max = _numeric_sql("'${spread_max}'", default="100.0")
    return f"""ctx AS (
  SELECT e.ts AS diag_ts,
         e.payload AS diag_payload,
         {exchange_expr} AS exchange,
         e.symbol,
         e.actor_id,
         {t_end_ms} AS t_end_ms,
         COALESCE(
           (
             {clean_bid} +
             {clean_ask}
           ) / 2.0,
           {state_mid},
           0.0
         ) AS mid,
         GREATEST({spread_max}, 1.0) AS spread_max
  FROM mm_events e
  {role_join}
  WHERE {_diagnostics_filter()}
    {role_where}
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
  SELECT spread, value FROM bid_step
  UNION ALL
  SELECT -selected.spread_max,
         (SELECT value FROM bid_step ORDER BY spread ASC LIMIT 1)
  FROM selected
  WHERE EXISTS (SELECT 1 FROM bid_step)
),
ask_step_plot AS (
  SELECT spread, value FROM ask_step
  UNION ALL
  SELECT selected.spread_max,
         (SELECT value FROM ask_step ORDER BY spread DESC LIMIT 1)
  FROM selected
  WHERE EXISTS (SELECT 1 FROM ask_step)
)"""


def _exchange_price_vwap_sql() -> str:
    best_bid = _numeric_sql("payload #>> ARRAY['orderbook','bids','0','price']", default="NULL")
    best_ask = _numeric_sql("payload #>> ARRAY['orderbook','asks','0','price']", default="NULL")
    book_buy_vwap = _numeric_sql("payload->>'book_buy_vwap'", default="NULL")
    book_sell_vwap = _numeric_sql("payload->>'book_sell_vwap'", default="NULL")
    trade_vwap = _numeric_sql("NULLIF(payload->>'vwap', '0')", default="NULL")
    filters = """event_type = 'MarketEvent'
  AND exchange = ANY(ARRAY[${exchange:sqlstring}]::text[])
  AND symbol = ANY(ARRAY[${symbol:sqlstring}]::text[])
  AND $__timeFilter(ts)"""
    return f"""WITH all_series AS (
  SELECT ts AS "time", (exchange || ' best_bid') AS metric, {best_bid} AS value, 'best_bid'::text AS series
  FROM mm_events
  WHERE {filters}
    AND (payload->'orderbook'->'bids'->0->>'price') IS NOT NULL
  UNION ALL
  SELECT ts AS "time", (exchange || ' best_ask') AS metric, {best_ask} AS value, 'best_ask'::text AS series
  FROM mm_events
  WHERE {filters}
    AND (payload->'orderbook'->'asks'->0->>'price') IS NOT NULL
  UNION ALL
  SELECT ts AS "time", (exchange || ' buy_vwap') AS metric, NULLIF({book_buy_vwap}, 0.0) AS value, 'buy_vwap'::text AS series
  FROM mm_events
  WHERE {filters}
    AND (payload->>'book_buy_vwap') IS NOT NULL
  UNION ALL
  SELECT ts AS "time", (exchange || ' sell_vwap') AS metric, NULLIF({book_sell_vwap}, 0.0) AS value, 'sell_vwap'::text AS series
  FROM mm_events
  WHERE {filters}
    AND (payload->>'book_sell_vwap') IS NOT NULL
  UNION ALL
  SELECT ts AS "time", (exchange || ' trade_vwap') AS metric, {trade_vwap} AS value, 'trade_vwap'::text AS series
  FROM mm_events
  WHERE {filters}
    AND NULLIF(payload->>'vwap', '0') IS NOT NULL
)
SELECT "time", metric, value
FROM all_series
WHERE series = ANY(ARRAY[${{price_series:sqlstring}}]::text[])
ORDER BY 1"""


def _exchange_volume_sql() -> str:
    buy_volume = _numeric_sql("payload->>'buy_volume'", default="NULL")
    sell_volume = _numeric_sql("payload->>'sell_volume'", default="NULL")
    filters = """event_type = 'MarketEvent'
  AND exchange = ANY(ARRAY[${exchange:sqlstring}]::text[])
  AND symbol = ANY(ARRAY[${symbol:sqlstring}]::text[])
  AND $__timeFilter(ts)"""
    return f"""WITH all_series AS (
  SELECT ts AS "time", (exchange || ' buy_volume') AS metric, {buy_volume} AS value, 'buy_volume'::text AS series
  FROM mm_events
  WHERE {filters}
    AND (payload->>'buy_volume') IS NOT NULL
  UNION ALL
  SELECT ts AS "time", (exchange || ' sell_volume') AS metric, {sell_volume} AS value, 'sell_volume'::text AS series
  FROM mm_events
  WHERE {filters}
    AND (payload->>'sell_volume') IS NOT NULL
)
SELECT "time", metric, value
FROM all_series
WHERE series = ANY(ARRAY[${{volume_series:sqlstring}}]::text[])
ORDER BY 1"""


def _best_bid_ask_sql() -> str:
    best_bid = _numeric_sql("payload #>> ARRAY['orderbook','bids','0','price']", default="NULL")
    best_ask = _numeric_sql("payload #>> ARRAY['orderbook','asks','0','price']", default="NULL")
    return f"""SELECT ts AS "time", (exchange || ' best_bid') AS metric,
       {best_bid} AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[]) AND symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) AND $__timeFilter(ts)
UNION ALL
SELECT ts AS "time", (exchange || ' best_ask') AS metric,
       {best_ask} AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[]) AND symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) AND $__timeFilter(ts)
ORDER BY 1"""


def _role_exchange_cte() -> str:
    return """role_exchange AS (
  SELECT DISTINCT role, exchange, symbol
  FROM (
    SELECT 'local'::text AS role, e.exchange AS exchange, e.symbol AS symbol
    FROM mm_events e
    WHERE e.topic = '${diagnostics_topic}'
      AND e.event_type = 'ControllerDiagnostics'
      AND e.symbol = ANY(ARRAY[${symbol:sqlstring}]::text[])
      AND e.actor_id = ANY(ARRAY[${actor_id:sqlstring}]::text[])
      AND $__timeFilter(e.ts)
    UNION ALL
    SELECT gates.value->>'role' AS role, gates.key AS exchange, e.symbol AS symbol
    FROM mm_events e
    CROSS JOIN LATERAL jsonb_each(
      COALESCE(e.payload #> ARRAY['execution_gates','venue_gates'], '{}'::jsonb)
    ) AS gates(key, value)
    WHERE e.topic = '${diagnostics_topic}'
      AND e.event_type = 'ControllerDiagnostics'
      AND e.symbol = ANY(ARRAY[${symbol:sqlstring}]::text[])
      AND e.actor_id = ANY(ARRAY[${actor_id:sqlstring}]::text[])
      AND $__timeFilter(e.ts)
      AND gates.value->>'role' IN ('local', 'global')
    UNION ALL
    SELECT 'local'::text,
           jf.payload #>> ARRAY['local_market','exchange'],
           jf.payload #>> ARRAY['local_market','symbol']
    FROM mm_events jf
    WHERE jf.event_type = 'JointStrategyFrame'
      AND (jf.payload #>> ARRAY['local_market','symbol']) = ANY(ARRAY[${symbol:sqlstring}]::text[])
      AND $__timeFilter(jf.ts)
    UNION ALL
    SELECT 'global'::text,
           jf.payload #>> ARRAY['global_market','exchange'],
           jf.payload #>> ARRAY['global_market','symbol']
    FROM mm_events jf
    WHERE jf.event_type = 'JointStrategyFrame'
      AND (jf.payload #>> ARRAY['global_market','exchange']) IS NOT NULL
      AND (jf.payload #>> ARRAY['global_market','symbol']) = ANY(ARRAY[${symbol:sqlstring}]::text[])
      AND $__timeFilter(jf.ts)
  ) r
  WHERE role IN ('local', 'global')
    AND exchange IS NOT NULL AND exchange <> ''
    AND symbol IS NOT NULL AND symbol <> ''
)"""


def _role_price_vwap_sql(role: str) -> str:
    role_key = "global" if role == "global" else "local"
    best_bid = _numeric_sql("m.payload #>> ARRAY['orderbook','bids','0','price']", default="NULL")
    best_ask = _numeric_sql("m.payload #>> ARRAY['orderbook','asks','0','price']", default="NULL")
    if role_key == "local":
        vwap_join = "r.role = 'local' AND r.exchange = e.exchange AND r.symbol = e.symbol"
        buy_field = "local_buy_vwap"
        sell_field = "local_sell_vwap"
    else:
        vwap_join = "r.role = 'global' AND r.symbol = e.symbol"
        buy_field = "global_buy_vwap"
        sell_field = "global_sell_vwap"
    buy_vwap = _numeric_sql(f"e.payload #>> ARRAY['state','{buy_field}']", default="NULL")
    sell_vwap = _numeric_sql(f"e.payload #>> ARRAY['state','{sell_field}']", default="NULL")
    return f"""WITH {_role_exchange_cte()}
SELECT m.ts AS "time", (m.exchange || ' best_bid') AS metric,
       {best_bid} AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND m.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND (m.payload->'orderbook'->'bids'->0->>'price') IS NOT NULL
  AND $__timeFilter(m.ts)
UNION ALL
SELECT m.ts AS "time", (m.exchange || ' best_ask') AS metric,
       {best_ask} AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND m.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND (m.payload->'orderbook'->'asks'->0->>'price') IS NOT NULL
  AND $__timeFilter(m.ts)
UNION ALL
SELECT e.ts AS "time", (r.exchange || ' buy_vwap') AS metric,
       {buy_vwap} AS value
FROM mm_events e
JOIN role_exchange r ON {vwap_join}
WHERE e.topic = '${{diagnostics_topic}}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
  AND r.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.payload #>> ARRAY['state','{buy_field}']) IS NOT NULL
  AND $__timeFilter(e.ts)
UNION ALL
SELECT e.ts AS "time", (r.exchange || ' sell_vwap') AS metric,
       {sell_vwap} AS value
FROM mm_events e
JOIN role_exchange r ON {vwap_join}
WHERE e.topic = '${{diagnostics_topic}}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
  AND r.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.payload #>> ARRAY['state','{sell_field}']) IS NOT NULL
  AND $__timeFilter(e.ts)
ORDER BY 1"""


def _role_volume_sql(role: str) -> str:
    role_key = "global" if role == "global" else "local"
    buy_volume = _numeric_sql("m.payload->>'buy_volume'", default="NULL")
    sell_volume = _numeric_sql("m.payload->>'sell_volume'", default="NULL")
    return f"""WITH {_role_exchange_cte()}
SELECT m.ts AS "time", (m.exchange || ' buy_volume') AS metric,
       {buy_volume} AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND m.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND (m.payload->>'buy_volume') IS NOT NULL
  AND $__timeFilter(m.ts)
UNION ALL
SELECT m.ts AS "time", (m.exchange || ' sell_volume') AS metric,
       {sell_volume} AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND m.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND (m.payload->>'sell_volume') IS NOT NULL
  AND $__timeFilter(m.ts)
ORDER BY 1"""


def _spread_sql() -> str:
    best_bid = _numeric_sql("payload #>> ARRAY['orderbook','bids','0','price']", default="NULL")
    best_ask = _numeric_sql("payload #>> ARRAY['orderbook','asks','0','price']", default="NULL")
    return f"""SELECT ts AS "time", (exchange || ' spread') AS metric,
       ({best_ask} - {best_bid}) AS value
FROM mm_events
WHERE event_type = 'MarketEvent'
  AND exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND $__timeFilter(ts)
ORDER BY 1"""


def _volume_sql() -> str:
    buy_volume = _numeric_sql("payload->>'buy_volume'", default="NULL")
    sell_volume = _numeric_sql("payload->>'sell_volume'", default="NULL")
    return f"""SELECT ts AS "time", (exchange || ' buy_volume') AS metric, {buy_volume} AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[]) AND symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) AND $__timeFilter(ts)
UNION ALL
SELECT ts AS "time", (exchange || ' sell_volume') AS metric, {sell_volume} AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[]) AND symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) AND $__timeFilter(ts)
ORDER BY 1"""


def _latest_model_scores_sql() -> str:
    nll = _numeric_sql("m.value->>'nll'", default="NULL")
    standardized_residual = _numeric_sql("m.value->>'standardized_residual'", default="NULL")
    return f"""SELECT e.ts, e.actor_id, e.exchange, e.symbol,
       m.key AS model,
       m.value->>'kind' AS kind,
       m.value->>'skip_reason' AS skip_reason,
       {nll} AS nll,
       {standardized_residual} AS standardized_residual,
       m.value->'params_after' AS params_after
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(COALESCE(e.payload->'models', '{{}}'::jsonb)) AS m(key, value)
WHERE {_diagnostics_filter()}
ORDER BY e.ts DESC
LIMIT 100"""


def _optimizer_objective_sql() -> str:
    value = _numeric_sql("e.payload #>> ARRAY['optimizer','objective',stage,'value']", default="NULL")
    return f"""SELECT e.ts AS "time", stage AS metric,
       {value} AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['initial','final','best']) AS stage
WHERE {_diagnostics_filter()}
  AND e.payload #>> ARRAY['optimizer','objective',stage,'value'] IS NOT NULL
ORDER BY 1"""


def _optimizer_improvement_sql() -> str:
    value = _numeric_sql("kv.value", default="NULL")
    return f"""SELECT e.ts AS "time", kv.key AS metric, {value} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['optimizer','objective','improvement'], '{{}}'::jsonb)) AS kv
WHERE {_diagnostics_filter()}
ORDER BY 1"""


_OBJECTIVE_CONTRIBUTION_KEYS_SQL = ", ".join(
    f"'{key}'"
    for key in (
        "risk_adjusted_return",
        "book_quality",
        "inventory_zone_penalty",
        "global_book_quality",
        "global_inventory_zone_penalty",
    )
)


def _optimizer_contrib_sql() -> str:
    value = _numeric_sql("kv.value", default="NULL")
    return f"""SELECT e.ts AS "time", kv.key AS metric, {value} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(
    e.payload #> ARRAY['optimizer','objective','contribution_fractions'],
    e.payload #> ARRAY['optimizer','objective','final','contribution_fractions'],
    '{{}}'::jsonb
  )
) AS kv
WHERE {_diagnostics_filter()}
  AND kv.key IN ({_OBJECTIVE_CONTRIBUTION_KEYS_SQL})
ORDER BY 1"""


def _optimizer_weighted_components_sql() -> str:
    value = _numeric_sql("kv.value", default="NULL")
    return f"""SELECT e.ts AS "time", kv.key AS metric, {value} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(e.payload #> ARRAY['optimizer','objective','final','components'], '{{}}'::jsonb)
) AS kv
WHERE {_diagnostics_filter()}
  AND kv.key IN ({_OBJECTIVE_CONTRIBUTION_KEYS_SQL})
ORDER BY 1"""


def _optimizer_book_breakdown_sql() -> str:
    value = _numeric_sql("kv.value", default="NULL")
    return f"""SELECT e.ts AS "time", kv.key AS metric, {value} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(e.payload #> ARRAY['optimizer','objective','final','components'], '{{}}'::jsonb)
) AS kv
WHERE {_diagnostics_filter()}
  AND (
    kv.key LIKE 'book_%'
    OR kv.key LIKE 'global_book_%'
  )
  AND kv.key NOT IN (
    'book_quality',
    'global_book_quality',
    'book_quality_score',
    'global_book_quality_score'
  )
ORDER BY 1"""


def _optimizer_inventory_zone_breakdown_sql() -> str:
    value = _numeric_sql("kv.value", default="NULL")
    return f"""SELECT e.ts AS "time", kv.key AS metric, {value} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(e.payload #> ARRAY['optimizer','objective','final','components'], '{{}}'::jsonb)
) AS kv
WHERE {_diagnostics_filter()}
  AND (
    kv.key IN (
      'inventory_zone_penalty_raw',
      'inventory_zone_imbalance',
      'inventory_zone_low_penalty',
      'inventory_zone_high_penalty',
      'global_inventory_zone_penalty_raw',
      'global_inventory_zone_imbalance',
      'global_inventory_zone_low_penalty',
      'global_inventory_zone_high_penalty'
    )
  )
ORDER BY 1"""


def _optimizer_inventory_zone_config_sql() -> str:
    value = _numeric_sql("kv.value", default="NULL")
    zone_keys = (
        "comfort_low",
        "comfort_high",
        "low_ramp_width",
        "high_ramp_width",
        "low_penalty_cap",
        "high_penalty_cap",
    )
    keys_sql = ", ".join(f"'{key}'" for key in zone_keys)
    return f"""SELECT e.ts AS "time",
       (scope || '.' || kv.key) AS metric,
       {value} AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['local_optimizer', 'global_optimizer']) AS scope
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(
    e.payload #> ARRAY['optimizer','config', scope, 'inventory_zone'],
    '{{}}'::jsonb
  )
) AS kv
WHERE {_diagnostics_filter()}
  AND kv.key IN ({keys_sql})
ORDER BY 1"""


def _optimizer_timing_sql() -> str:
    value = _numeric_sql("kv.value", default="NULL")
    return f"""SELECT e.ts AS "time", kv.key AS metric,
       CASE WHEN kv.key LIKE '%_sec' THEN ({value}) * 1000.0 ELSE ({value}) END AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['optimizer','timing'], '{{}}'::jsonb)) AS kv
WHERE {_diagnostics_filter()}
  AND trim(kv.value) ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$'
ORDER BY 1"""


def _optimizer_gradients_sql() -> str:
    value = _numeric_sql("kv.value", default="NULL")
    return f"""SELECT e.ts AS "time", kv.key AS metric, {value} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['optimizer','gradients'], '{{}}'::jsonb)) AS kv
WHERE {_diagnostics_filter()}
  AND trim(kv.value) ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$'
ORDER BY 1"""


def _action_contract_funnel_sql() -> str:
    order_count = _numeric_sql("e.payload #>> ARRAY['action_pipeline', stage, 'order_count']")
    return f"""SELECT e.ts AS "time", label AS metric,
       {order_count} AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES
    ('target_book', 'desired target'),
    ('risk_approved_target', 'risk approved'),
    ('router_accepted', 'router accepted')
) AS stages(stage, label)
WHERE {_diagnostics_filter()}
ORDER BY 1"""


def _action_rejection_reason_sql() -> str:
    count_value = _numeric_sql("reason.value", default="0.0")
    return f"""SELECT e.ts AS "time",
       (stage_label || ': ' || reason.key) AS metric,
       {count_value} AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES
    ('capped', 'strategy cap'),
    ('risk_approved_target', 'risk'),
    ('rules_valid', 'exchange rules'),
    ('router_accepted', 'router')
) AS stages(stage, stage_label)
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(e.payload #> ARRAY['action_pipeline', stage, 'rejection_counts'], '{{}}'::jsonb)
) AS reason(key, value)
WHERE {_diagnostics_filter()}
ORDER BY 1"""


def _role_execution_contract_sql(role: str) -> str:
    if role not in {"local", "global"}:
        raise ValueError(f"unknown execution contract role: {role}")
    target_path = "global" if role == "global" else "local"
    target_step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    gate_step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    return f"""WITH target_books AS (
  SELECT {_approx_step_time_alias_sql(target_step_source)},
         'target orders'::text AS metric,
         SUM(jsonb_array_length(COALESCE(e.payload->'{target_path}'->'orders', '[]'::jsonb)))::double precision AS value
  FROM mm_events e
  WHERE e.topic = '${{target_books_topic}}'
    AND e.event_type = 'JointTargetBook'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  GROUP BY 1, 2
),
venue_gates AS (
  SELECT {_approx_step_time_alias_sql(gate_step_source)},
         stage.metric,
         MAX(CASE WHEN stage.ok THEN 1.0 ELSE 0.0 END) AS value
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_each(
    COALESCE(e.payload #> ARRAY['execution_gates','venue_gates'], '{{}}'::jsonb)
  ) AS gate(key, value)
  CROSS JOIN LATERAL (
    VALUES
      ('gate has orders', COALESCE(NULLIF(gate.value->>'has_orders', ''), 'false')::boolean),
      ('gate can sync', COALESCE(NULLIF(gate.value->>'can_sync', ''), 'false')::boolean),
      ('gate live', COALESCE(NULLIF(gate.value->>'live', ''), 'false')::boolean)
  ) AS stage(metric, ok)
  WHERE {_diagnostics_filter()}
    AND COALESCE(gate.value->>'role', '') = '{role}'
  GROUP BY 1, 2
)
SELECT "time", metric, value FROM target_books
UNION ALL
SELECT "time", metric, value FROM venue_gates
ORDER BY 1"""


def _role_rejection_reason_sql(role: str) -> str:
    if role not in {"local", "global"}:
        raise ValueError(f"unknown rejection role: {role}")
    global_reason = """(reason.key LIKE 'global_%'
       OR reason.key IN ('missing_global_market_data', 'stale_global_market_data', 'missing_global_ledger'))"""
    risk_filter = global_reason if role == "global" else f"NOT {global_reason}"
    local_stage_rejections = ""
    if role == "local":
        local_step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
        local_stage_rejections = f"""
UNION ALL
SELECT {_approx_step_time_alias_sql(local_step_source)},
       (stage_label || ': ' || reason.key) AS metric,
       {_numeric_sql("reason.value", default="0.0")} AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES
    ('capped', 'strategy cap'),
    ('rules_valid', 'exchange rules'),
    ('router_accepted', 'router')
) AS stages(stage, stage_label)
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(e.payload #> ARRAY['action_pipeline', stage, 'rejection_counts'], '{{}}'::jsonb)
) AS reason(key, value)
WHERE {_diagnostics_filter()}"""
    risk_step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    return f"""WITH role_rejections AS (
SELECT {_approx_step_time_alias_sql(risk_step_source)},
       ('risk: ' || reason.key) AS metric,
       {_numeric_sql("reason.value", default="0.0")} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(
    e.payload #> ARRAY['action_pipeline','risk_approved_target','rejection_counts'],
    e.payload #> ARRAY['risk','rejection_counts'],
    '{{}}'::jsonb
  )
) AS reason(key, value)
WHERE {_diagnostics_filter()}
  AND {risk_filter}
{local_stage_rejections}
)
SELECT "time", metric, SUM(value) AS value
FROM role_rejections
GROUP BY 1, 2
ORDER BY 1"""


def _local_stage_drop_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    ladder_count = _numeric_sql("e.payload #>> ARRAY['action_pipeline','ladder','order_count']")
    capped_count = _numeric_sql("e.payload #>> ARRAY['action_pipeline','capped','order_count']")
    return f"""WITH drops AS (
  SELECT {_approx_step_time_alias_sql(step_source)},
         drop_stage.metric,
         GREATEST(drop_stage.before_count - drop_stage.after_count, 0.0) AS value
  FROM mm_events e
  CROSS JOIN LATERAL (
    VALUES
      ('strategy cap'::text, {ladder_count}, {capped_count})
  ) AS drop_stage(metric, before_count, after_count)
  WHERE {_diagnostics_filter()}
)
SELECT "time", metric, SUM(value) AS value
FROM drops
GROUP BY 1, 2
ORDER BY 1"""


def _role_gate_block_reason_sql(role: str) -> str:
    if role not in {"local", "global"}:
        raise ValueError(f"unknown gate block role: {role}")
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    return f"""SELECT {_approx_step_time_alias_sql(step_source)},
       blocked.reason AS metric,
       COUNT(*)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(
  COALESCE(e.payload #> ARRAY['execution_gates','venue_gates'], '{{}}'::jsonb)
) AS gate(key, value)
CROSS JOIN LATERAL jsonb_array_elements_text(
  COALESCE(gate.value->'blocked_reasons', '[]'::jsonb)
) AS blocked(reason)
WHERE {_diagnostics_filter()}
  AND COALESCE(gate.value->>'role', '') = '{role}'
GROUP BY 1, 2
ORDER BY 1"""


def _action_execution_summary_sql() -> str:
    desired = _numeric_sql("e.payload #>> ARRAY['action_pipeline','target_book','order_count']")
    approved = _numeric_sql("e.payload #>> ARRAY['action_pipeline','risk_approved_target','order_count']")
    accepted = _numeric_sql("e.payload #>> ARRAY['action_pipeline','router_accepted','order_count']")
    new_orders = _numeric_sql("e.payload #>> ARRAY['action_pipeline','planned_place','order_count']")
    cancel_orders = _numeric_sql("e.payload #>> ARRAY['action_pipeline','planned_cancel','order_count']")
    kept_orders = _numeric_sql("e.payload #>> ARRAY['action_pipeline','kept','order_count']")
    valid_new_orders = _numeric_sql("e.payload #>> ARRAY['action_pipeline','rules_valid','order_count']")
    return f"""SELECT e.ts,
       e.actor_id,
       e.exchange,
       e.symbol,
       {desired} AS target_orders,
       {approved} AS risk_passed_orders,
       {kept_orders} AS kept_live_orders,
       {cancel_orders} AS cancel_orders,
       {new_orders} AS new_order_candidates,
       {valid_new_orders} AS valid_new_orders,
       {accepted} AS exchange_accepted_new_orders,
       COALESCE(e.payload #> ARRAY['action_pipeline','risk_approved_target','rejection_counts'], '{{}}'::jsonb) AS risk_blocks,
       COALESCE(e.payload #> ARRAY['action_pipeline','rules_valid','rejection_counts'], '{{}}'::jsonb) AS rule_blocks,
       COALESCE(e.payload #> ARRAY['action_pipeline','router_accepted','rejection_counts'], '{{}}'::jsonb) AS router_blocks
FROM mm_events e
WHERE {_diagnostics_filter()}
ORDER BY e.ts DESC
LIMIT 50"""


def _latest_target_orders_sql(limit: int = 200) -> str:
    price = _numeric_sql("o.value->>'price'", default="NULL")
    qty = _numeric_sql("o.value->>'qty'", default="NULL")
    return f"""WITH recent_books AS (
  SELECT e.ts,
         e.actor_id,
         e.exchange AS frame_exchange,
         e.symbol AS frame_symbol,
         e.payload
  FROM mm_events e
  WHERE e.topic = '${{target_books_topic}}'
    AND e.event_type = 'JointTargetBook'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND e.actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
  ORDER BY e.ts DESC
  LIMIT 25
)
SELECT rb.ts,
       rb.actor_id,
       book.role,
       COALESCE(book.value->>'exchange', rb.frame_exchange, '') AS exchange,
       COALESCE(book.value->>'symbol', rb.frame_symbol, '') AS symbol,
       COALESCE(book.value->>'policy', '') AS policy,
       o.order_no,
       COALESCE(o.value->>'side', '') AS side,
       {price} AS price,
       {qty} AS qty,
       COALESCE(o.value->>'order_type', 'limit') AS order_type
FROM recent_books rb
CROSS JOIN LATERAL (
  VALUES ('local', rb.payload->'local'), ('global', rb.payload->'global')
) AS book(role, value)
CROSS JOIN LATERAL jsonb_array_elements(COALESCE(book.value->'orders', '[]'::jsonb)) WITH ORDINALITY AS o(value, order_no)
WHERE book.value IS NOT NULL
ORDER BY rb.ts DESC, book.role, o.order_no
LIMIT {limit}"""


def _recent_sync_summaries_sql(limit: int = 100) -> str:
    return f"""SELECT e.ts,
       e.event_type,
       e.exchange,
       e.symbol,
       e.actor_id,
       e.payload->>'target_count' AS target_count,
       e.payload->>'keep_count' AS keep_count,
       e.payload->>'cancel_count' AS cancel_count,
       e.payload->>'place_count' AS place_count,
       e.payload->>'accepted_count' AS accepted_count,
       e.payload->>'rejected_count' AS rejected_count,
       e.payload->>'placed' AS placed,
       e.payload->>'latency_ms' AS latency_ms,
       e.payload->>'error' AS error,
       e.payload->>'message' AS message
FROM mm_events e
WHERE e.event_type IN ('ExecutionSyncRequested', 'ExecutionSyncCompleted', 'ExecutionSyncFailed')
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
  AND $__timeFilter(e.ts)
ORDER BY e.ts DESC
LIMIT {limit}"""


def _recent_placed_orders_sql(limit: int = 100) -> str:
    price = _numeric_sql("o.value->>'price'", default="NULL")
    qty = _numeric_sql("o.value->>'qty'", default="NULL")
    return f"""SELECT e.ts,
       COALESCE(o.value->>'exchange', e.exchange, '') AS exchange,
       COALESCE(o.value->>'symbol', e.symbol, '') AS symbol,
       e.actor_id,
       COALESCE(o.value->>'exchange_order_id', '') AS exchange_order_id,
       o.order_no,
       COALESCE(o.value->>'side', '') AS side,
       {price} AS price,
       {qty} AS qty,
       COALESCE(o.value->>'order_type', 'limit') AS order_type,
       COALESCE(o.value->>'client_order_id', '') AS client_order_id
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) WITH ORDINALITY AS o(value, order_no)
WHERE e.event_type = 'PlaceOrdersAccepted'
  AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND $__timeFilter(e.ts)
ORDER BY e.ts DESC, o.order_no
LIMIT {limit}"""


def _recent_place_requests_sql(limit: int = 100) -> str:
    price = _numeric_sql("o.value->>'price'", default="NULL")
    qty = _numeric_sql("o.value->>'qty'", default="NULL")
    return f"""SELECT e.ts,
       COALESCE(o.value->>'exchange', e.exchange, '') AS exchange,
       COALESCE(o.value->>'symbol', e.symbol, '') AS symbol,
       e.actor_id,
       COALESCE(e.payload->>'policy', '') AS policy,
       o.order_no,
       COALESCE(o.value->>'side', '') AS side,
       {price} AS price,
       {qty} AS qty,
       COALESCE(o.value->>'order_type', 'limit') AS order_type,
       COALESCE(o.value->>'client_order_id', '') AS client_order_id
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) WITH ORDINALITY AS o(value, order_no)
WHERE e.event_type = 'PlaceOrdersRequested'
  AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND $__timeFilter(e.ts)
ORDER BY e.ts DESC, o.order_no
LIMIT {limit}"""


def _recent_canceled_orders_sql(limit: int = 100) -> str:
    return f"""SELECT e.ts,
       e.event_type,
       e.exchange,
       e.symbol,
       e.actor_id,
       order_id.value AS exchange_order_id,
       COALESCE(e.payload->>'policy', '') AS policy,
       COALESCE(e.payload->>'error', '') AS error,
       COALESCE(e.payload->>'message', '') AS message
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(e.payload->'order_ids', '[]'::jsonb)) AS order_id(value)
WHERE e.event_type IN ('CancelOrdersRequested', 'CancelOrdersAccepted', 'CancelOrdersFailed')
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
  AND $__timeFilter(e.ts)
ORDER BY e.ts DESC
LIMIT {limit}"""


def _recent_rejected_orders_sql(limit: int = 100) -> str:
    validation_price = _numeric_sql("e.payload->>'price'", default="NULL")
    validation_qty = _numeric_sql("e.payload->>'qty'", default="NULL")
    failed_price = _numeric_sql("o.value->>'price'", default="NULL")
    failed_qty = _numeric_sql("o.value->>'qty'", default="NULL")
    return f"""SELECT *
FROM (
  SELECT e.ts,
         'validation'::text AS source,
         e.exchange,
         e.symbol,
         e.actor_id,
         COALESCE(e.payload->>'stage', 'rules') AS stage,
         COALESCE(e.payload->>'reason', '') AS reason,
         COALESCE(e.payload->>'side', '') AS side,
         {validation_price} AS price,
         {validation_qty} AS qty,
         ''::text AS exchange_order_id
  FROM mm_events e
  WHERE e.event_type = 'OrderValidationRejected'
    AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND (e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR e.symbol IS NULL)
    AND $__timeFilter(e.ts)
  UNION ALL
  SELECT e.ts,
         'router'::text AS source,
         COALESCE(o.value->>'exchange', e.exchange, '') AS exchange,
         COALESCE(o.value->>'symbol', e.symbol, '') AS symbol,
         e.actor_id,
         'router'::text AS stage,
         COALESCE(e.payload->>'error', '') AS reason,
         COALESCE(o.value->>'side', '') AS side,
         {failed_price} AS price,
         {failed_qty} AS qty,
         ''::text AS exchange_order_id
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
  WHERE e.event_type = 'PlaceOrdersFailed'
    AND COALESCE(o.value->>'exchange', e.exchange, '') = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND COALESCE(o.value->>'symbol', e.symbol, '') = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    AND $__timeFilter(e.ts)
) rejected
ORDER BY ts DESC
LIMIT {limit}"""


def _recent_execution_events_sql(limit: int = 50) -> str:
    return f"""SELECT ts, topic, event_type, exchange, symbol, actor_id, payload
FROM mm_events
WHERE topic LIKE 'execution:%'
  AND exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND (symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[]) OR symbol IS NULL)
  AND $__timeFilter(ts)
ORDER BY ts DESC
LIMIT {limit}"""


def _portfolio_field_sql(*fields: str) -> str:
    selects = []
    for field in fields:
        value = _numeric_sql(f"e.payload #>> ARRAY['state','{field}']", default="NULL")
        selects.append(
            f"""SELECT e.ts AS "time", '{field}' AS metric,
       {value} AS value
FROM mm_events e
WHERE {_diagnostics_filter()}
  AND e.payload #>> ARRAY['state','{field}'] IS NOT NULL"""
        )
    return "\nUNION ALL\n".join(selects) + "\nORDER BY 1"


def _portfolio_wallet_or_state_total(role: str, leg: str) -> str:
    if role not in {"local", "global"}:
        raise ValueError(f"unsupported portfolio role: {role}")
    if leg == "cash":
        wallet_path = f"e.payload #>> ARRAY['state','{role}_quote_wallet','total']"
        state_path = f"e.payload #>> ARRAY['state','{role}_cash']"
    elif leg == "inventory":
        wallet_path = f"e.payload #>> ARRAY['state','{role}_base_wallet','total']"
        state_path = f"e.payload #>> ARRAY['state','{role}_inventory']"
    else:
        raise ValueError(f"unsupported portfolio leg: {leg}")
    return f"COALESCE({_numeric_sql(wallet_path, default='NULL')}, {_numeric_sql(state_path)})"


def _portfolio_total_value_sql() -> str:
    local_cash = _portfolio_wallet_or_state_total("local", "cash")
    global_cash = _portfolio_wallet_or_state_total("global", "cash")
    local_inventory = _portfolio_wallet_or_state_total("local", "inventory")
    global_inventory = _portfolio_wallet_or_state_total("global", "inventory")
    reference_price = _numeric_sql("e.payload #>> ARRAY['optimizer','config','reference_price']", default="NULL")
    global_sell_vwap = _numeric_sql("e.payload #>> ARRAY['state','global_sell_vwap']", default="NULL")
    local_sell_vwap = _numeric_sql("e.payload #>> ARRAY['state','local_sell_vwap']", default="NULL")
    mid = _numeric_sql("e.payload #>> ARRAY['state','mid']", default="NULL")
    return f"""WITH state_rows AS (
  SELECT e.ts AS "time",
         {local_cash} AS local_cash,
         {global_cash} AS global_cash,
         {local_inventory} AS local_inventory,
         {global_inventory} AS global_inventory,
         COALESCE(
           {reference_price},
           {global_sell_vwap},
           {local_sell_vwap},
           {mid},
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


def _portfolio_venue_value_sql(role: str) -> str:
    if role not in {"local", "global"}:
        raise ValueError(f"unsupported portfolio role: {role}")
    cash = _portfolio_wallet_or_state_total(role, "cash")
    inventory = _portfolio_wallet_or_state_total(role, "inventory")
    wealth = _numeric_sql(f"e.payload #>> ARRAY['state','{role}_wealth']", default="NULL")
    role_sell_vwap = _numeric_sql(f"e.payload #>> ARRAY['state','{role}_sell_vwap']", default="NULL")
    reference_price = _numeric_sql("e.payload #>> ARRAY['optimizer','config','reference_price']", default="NULL")
    mid = _numeric_sql("e.payload #>> ARRAY['state','mid']", default="NULL")
    return f"""WITH state_rows AS (
  SELECT e.ts AS "time",
         {cash} AS cash,
         {inventory} AS inventory,
         {wealth} AS reported_wealth,
         COALESCE(
           {role_sell_vwap},
           {reference_price},
           {mid},
           0.0
         ) AS mark_price
  FROM mm_events e
  WHERE {_diagnostics_filter()}
)
SELECT "time", '{role}_cash' AS metric,
       cash AS value
FROM state_rows
UNION ALL
SELECT "time", '{role}_inventory_value',
       (inventory * mark_price) AS value
FROM state_rows
UNION ALL
SELECT "time", '{role}_wealth',
       COALESCE(reported_wealth, cash + inventory * mark_price) AS value
FROM state_rows
ORDER BY 1"""


def _portfolio_interval_delta_sql(metric: str | None = None) -> str:
    if metric is not None and metric not in {"delta_cash", "delta_inventory", "delta_wealth", "delta_price"}:
        raise ValueError(f"unsupported interval delta metric: {metric}")
    local_cash = _portfolio_wallet_or_state_total("local", "cash")
    global_cash = _portfolio_wallet_or_state_total("global", "cash")
    local_inventory = _portfolio_wallet_or_state_total("local", "inventory")
    global_inventory = _portfolio_wallet_or_state_total("global", "inventory")
    local_sell_vwap = _numeric_sql("e.payload #>> ARRAY['state','local_sell_vwap']", default="NULL")
    global_sell_vwap = _numeric_sql("e.payload #>> ARRAY['state','global_sell_vwap']", default="NULL")
    reference_price = _numeric_sql("e.payload #>> ARRAY['optimizer','config','reference_price']", default="NULL")
    mid = _numeric_sql("e.payload #>> ARRAY['state','mid']", default="NULL")
    mark_price = f"COALESCE({reference_price}, {global_sell_vwap}, {local_sell_vwap}, {mid}, 0.0)"
    cash = f"({local_cash} + {global_cash})"
    inventory = f"({local_inventory} + {global_inventory})"
    return f"""WITH state_rows AS (
  SELECT e.ts AS "time",
         {cash} AS cash,
         {inventory} AS inventory,
         {mark_price} AS mark_price,
         ({cash}) + ({inventory}) * ({mark_price}) AS wealth
  FROM mm_events e
  WHERE {_diagnostics_filter()}
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (ORDER BY "time" ASC) AS first_rank,
         ROW_NUMBER() OVER (ORDER BY "time" DESC) AS last_rank
  FROM state_rows
),
bounds AS (
  SELECT
    MAX(cash) FILTER (WHERE first_rank = 1) AS first_cash,
    MAX(cash) FILTER (WHERE last_rank = 1) AS last_cash,
    MAX(inventory) FILTER (WHERE first_rank = 1) AS first_inventory,
    MAX(inventory) FILTER (WHERE last_rank = 1) AS last_inventory,
    MAX(wealth) FILTER (WHERE first_rank = 1) AS first_wealth,
    MAX(wealth) FILTER (WHERE last_rank = 1) AS last_wealth,
    MAX(mark_price) FILTER (WHERE first_rank = 1) AS first_price,
    MAX(mark_price) FILTER (WHERE last_rank = 1) AS last_price
  FROM ranked
)
SELECT ($__timeTo())::timestamptz AS "time", metric, value
FROM bounds
CROSS JOIN LATERAL (
  VALUES
    ('delta_cash', last_cash - first_cash),
    ('delta_inventory', last_inventory - first_inventory),
    ('delta_wealth', last_wealth - first_wealth),
    ('delta_price', last_price - first_price)
) AS deltas(metric, value)
WHERE value IS NOT NULL{f" AND metric = '{metric}'" if metric is not None else ""}
ORDER BY metric"""


def _portfolio_latest_total_sql(leg: str) -> str:
    if leg not in {"cash", "inventory"}:
        raise ValueError(f"unsupported latest total portfolio leg: {leg}")
    local_value = _portfolio_wallet_or_state_total("local", leg)
    global_value = _portfolio_wallet_or_state_total("global", leg)
    return f"""SELECT e.ts AS "time",
       ({local_value} + {global_value}) AS value
FROM mm_events e
WHERE {_diagnostics_filter()}
ORDER BY e.ts DESC
LIMIT 1"""


def _portfolio_cumulative_pnl_component_sql(component: str) -> str:
    component_expr = {
        "net_spread_capture": "(delta_cash + delta_inventory * mark_price_prev)",
        "inventory_drift": "(inventory_prev * delta_price)",
        "adverse_selection": "(delta_inventory * delta_price)",
    }.get(component)
    if component_expr is None:
        raise ValueError(f"unsupported portfolio PnL component: {component}")
    local_cash = _portfolio_wallet_or_state_total("local", "cash")
    global_cash = _portfolio_wallet_or_state_total("global", "cash")
    local_inventory = _portfolio_wallet_or_state_total("local", "inventory")
    global_inventory = _portfolio_wallet_or_state_total("global", "inventory")
    local_sell_vwap = _numeric_sql("e.payload #>> ARRAY['state','local_sell_vwap']", default="NULL")
    global_sell_vwap = _numeric_sql("e.payload #>> ARRAY['state','global_sell_vwap']", default="NULL")
    reference_price = _numeric_sql("e.payload #>> ARRAY['optimizer','config','reference_price']", default="NULL")
    mid = _numeric_sql("e.payload #>> ARRAY['state','mid']", default="NULL")
    return f"""WITH state_rows AS (
  SELECT e.ts AS "time",
         {local_cash} AS local_cash,
         {global_cash} AS global_cash,
         {local_inventory} AS local_inventory,
         {global_inventory} AS global_inventory,
         COALESCE({local_sell_vwap}, {reference_price}, {mid}, 0.0) AS local_mark_price,
         COALESCE({global_sell_vwap}, {reference_price}, {local_sell_vwap}, {mid}, 0.0) AS global_mark_price,
         COALESCE({reference_price}, {global_sell_vwap}, {local_sell_vwap}, {mid}, 0.0) AS total_mark_price
  FROM mm_events e
  WHERE {_diagnostics_filter()}
),
scope_rows AS (
  SELECT "time", 'local' AS metric,
         local_cash AS cash,
         local_inventory AS inventory,
         local_mark_price AS mark_price
  FROM state_rows
  UNION ALL
  SELECT "time", 'global',
         global_cash,
         global_inventory,
         global_mark_price
  FROM state_rows
  UNION ALL
  SELECT "time", 'total',
         (local_cash + global_cash),
         (local_inventory + global_inventory),
         total_mark_price
  FROM state_rows
),
deltas AS (
  SELECT "time",
         metric,
         cash,
         inventory,
         mark_price,
         LAG(cash) OVER (PARTITION BY metric ORDER BY "time") AS cash_prev,
         LAG(inventory) OVER (PARTITION BY metric ORDER BY "time") AS inventory_prev,
         LAG(mark_price) OVER (PARTITION BY metric ORDER BY "time") AS mark_price_prev
  FROM scope_rows
),
step_parts AS (
  SELECT "time",
         metric,
         (cash - cash_prev) AS delta_cash,
         (inventory - inventory_prev) AS delta_inventory,
         (mark_price - mark_price_prev) AS delta_price,
         inventory_prev,
         mark_price_prev
  FROM deltas
  WHERE cash_prev IS NOT NULL
    AND inventory_prev IS NOT NULL
    AND mark_price_prev IS NOT NULL
),
component_steps AS (
  SELECT "time", metric,
         {component_expr} AS step_value
  FROM step_parts
)
SELECT "time", metric,
       SUM(step_value) OVER (
         PARTITION BY metric
         ORDER BY "time"
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS value
FROM component_steps
ORDER BY 1, 2"""


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
WHERE exchange = ANY(ARRAY[${exchange:sqlstring}]::text[])
  AND symbol = ANY(ARRAY[${symbol:sqlstring}]::text[])
  AND $__timeFilter(ts)
ORDER BY 1"""


def _fill_interval_stat_sql(stat: str, *, role: str | None = None) -> str:
    if role is not None and role not in {"local", "global"}:
        raise ValueError(f"unsupported fill interval role: {role}")
    stat_exprs = {
        "bid_volume": "COALESCE(bid_qty, 0.0)",
        "ask_volume": "COALESCE(ask_qty, 0.0)",
        "bid_avg_price": "bid_avg_price",
        "ask_avg_price": "ask_avg_price",
        "spread_pnl": "LEAST(COALESCE(bid_qty, 0.0), COALESCE(ask_qty, 0.0)) * (COALESCE(ask_avg_price, 0.0) - COALESCE(bid_avg_price, 0.0))",
        "total_fee": "COALESCE(total_fee, 0.0)",
    }
    if stat not in stat_exprs:
        raise ValueError(f"unknown fill interval stat: {stat}")
    exchange = "COALESCE(e.exchange, e.payload->>'exchange', substring(e.topic from '^private:state:([^:]+):fills$'), 'unknown')"
    symbol = "COALESCE(e.symbol, e.payload->>'symbol', '')"
    side = _side_sql("e.payload->>'side'")
    qty = _numeric_sql("e.payload->>'qty'")
    price = _numeric_sql("e.payload->>'price'", default="NULL")
    fee = _numeric_sql("e.payload->>'fee'")
    role_filter = ""
    if role is not None:
        role_filter = f"""
    AND {exchange} IN (
      SELECT role_exchange
      FROM role_exchanges
      WHERE role = '{role}'
    )"""
    return f"""WITH latest_diagnostics AS (
  SELECT e.payload, e.exchange
  FROM mm_events e
  WHERE {_diagnostics_filter()}
  ORDER BY e.ts DESC
  LIMIT 1
),
role_exchanges AS (
  SELECT gates.key AS role_exchange,
         gates.value->>'role' AS role
  FROM latest_diagnostics d
  CROSS JOIN LATERAL jsonb_each(COALESCE(d.payload #> ARRAY['execution_gates','venue_gates'], '{{}}'::jsonb)) AS gates
  UNION
  SELECT d.exchange AS role_exchange,
         'local' AS role
  FROM latest_diagnostics d
  WHERE d.exchange IS NOT NULL
),
fills AS (
  SELECT
    {side} AS side,
    {qty} AS qty,
    {price} AS price,
    {fee} AS fee
  FROM mm_events e
  LEFT JOIN role_exchanges r ON r.role_exchange = {exchange}
  WHERE e.topic LIKE 'private:state:%:fills'
    AND {exchange} = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
    AND {symbol} = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
    {role_filter}
    AND $__timeFilter(e.ts)
),
agg AS (
  SELECT
    SUM(qty) FILTER (WHERE side = 'bid' AND qty > 0) AS bid_qty,
    SUM(qty) FILTER (WHERE side = 'ask' AND qty > 0) AS ask_qty,
    SUM(fee) AS total_fee,
    SUM(price * qty) FILTER (WHERE side = 'bid' AND price IS NOT NULL AND qty > 0)
      / NULLIF(SUM(qty) FILTER (WHERE side = 'bid' AND price IS NOT NULL AND qty > 0), 0.0) AS bid_avg_price,
    SUM(price * qty) FILTER (WHERE side = 'ask' AND price IS NOT NULL AND qty > 0)
      / NULLIF(SUM(qty) FILTER (WHERE side = 'ask' AND price IS NOT NULL AND qty > 0), 0.0) AS ask_avg_price
  FROM fills
)
SELECT ($__timeTo())::timestamptz AS "time",
       {stat_exprs[stat]} AS value
FROM agg"""


def _venue_gates_sql() -> str:
    return f"""SELECT e.ts, gates.key AS exchange, gates.value->>'role' AS role,
       gates.value->>'live' AS live,
       gates.value->>'has_orders' AS has_orders,
       gates.value->>'can_sync' AS can_sync,
       e.payload #> ARRAY['execution_gates','blocked_reasons'] AS frame_blocked_reasons,
       gates.value->'blocked_reasons' AS venue_blocked_reasons
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(COALESCE(e.payload #> ARRAY['execution_gates','venue_gates'], '{{}}'::jsonb)) AS gates
WHERE {_diagnostics_filter()}
ORDER BY e.ts DESC
LIMIT 100"""


def _target_book_values_sql(venue: str | None) -> str:
    if venue is None:
        return "VALUES ('local', payload->'local'), ('global', payload->'global')"
    if venue not in {"local", "global"}:
        raise ValueError(f"unknown target-book venue: {venue}")
    return f"VALUES ('{venue}', payload->'{venue}')"


def _target_book_count_sql(venue: str | None = None) -> str:
    values_sql = _target_book_values_sql(venue).replace("payload", "e.payload")
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT {_approx_step_time_alias_sql(step_source)},
       venue AS metric,
       SUM(jsonb_array_length(COALESCE(book->'orders', '[]'::jsonb)))::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL (
  {values_sql}
) AS v(venue, book)
WHERE e.topic = '${{target_books_topic}}'
  AND e.event_type = 'JointTargetBook'
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND book IS NOT NULL
  AND $__timeFilter(e.ts)
GROUP BY 1, 2
ORDER BY 1"""
    metrics_sql = (
        f"""  SELECT '{venue}'::text AS metric"""
        if venue is not None
        else """  SELECT metric
  FROM (
    VALUES ('local'), ('global')
  ) AS m(metric)"""
    )
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _target_book_qty_sql(venue: str | None = None) -> str:
    qty = _numeric_sql("o->>'qty'")
    side = _side_sql("o->>'side'")
    values_sql = _target_book_values_sql(venue).replace("payload", "e.payload")
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT {_approx_step_time_alias_sql(step_source)}, (venue || ' ' || {side}) AS metric,
       SUM({qty}) AS value
FROM mm_events e
CROSS JOIN LATERAL (
  {values_sql}
) AS v(venue, book)
CROSS JOIN LATERAL jsonb_array_elements(COALESCE(book->'orders', '[]'::jsonb)) AS o
WHERE e.topic = '${{target_books_topic}}'
  AND e.event_type = 'JointTargetBook'
  AND e.exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])
  AND e.symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])
  AND $__timeFilter(e.ts)
GROUP BY 1, venue, {side}
ORDER BY 1"""
    if venue is not None:
        metrics_sql = f"""  SELECT metric
  FROM (
    VALUES ('{venue} bid'), ('{venue} ask')
  ) AS m(metric)"""
    else:
        metrics_sql = """  SELECT metric
  FROM (
    VALUES ('local bid'), ('local ask'), ('global bid'), ('global ask')
  ) AS m(metric)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _latest_payload_sql(kind: str) -> str:
    if kind == "diagnostics":
        where = f"{_diagnostics_filter()}"
    elif kind == "decision":
        where = """event_type = 'StrategyDecision'
  AND exchange = ANY(ARRAY[${exchange:sqlstring}]::text[])
  AND symbol = ANY(ARRAY[${symbol:sqlstring}]::text[])
  AND actor_id = ANY(ARRAY[${actor_id:sqlstring}]::text[])
  AND $__timeFilter(ts)"""
    else:
        where = """topic = '${target_books_topic}'
  AND event_type = 'JointTargetBook'
  AND exchange = ANY(ARRAY[${exchange:sqlstring}]::text[])
  AND symbol = ANY(ARRAY[${symbol:sqlstring}]::text[])
  AND actor_id = ANY(ARRAY[${actor_id:sqlstring}]::text[])
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
        pipeline_time_profile(),
        model_quality(),
        model_parameters(),
        optimizer_diagnostics(),
        action_pipeline(),
        target_order_stems(),
        action_pipeline_side_counts(),
        risk_execution(),
        execution_errors(),
        portfolio_pnl(),
        fill_spread_pnl(),
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
