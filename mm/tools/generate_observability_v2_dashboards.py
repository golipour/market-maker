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
MM_SCOPE_EVENT_TYPES = (
    "'StrategyDecision', 'ControllerDiagnostics', 'JointTargetBook'"
)


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
        _query_var(
            "exchange",
            "Exchange",
            f"SELECT DISTINCT exchange FROM mm_events WHERE exchange IS NOT NULL AND event_type IN ({MM_SCOPE_EVENT_TYPES}) ORDER BY 1",
            multi=True,
        ),
        _query_var(
            "symbol",
            "Symbol",
            f"SELECT DISTINCT symbol FROM mm_events WHERE symbol IS NOT NULL AND event_type IN ({MM_SCOPE_EVENT_TYPES}) ORDER BY 1",
            multi=True,
        ),
        _query_var(
            "actor_id",
            "Actor",
            f"SELECT DISTINCT actor_id FROM mm_events WHERE actor_id IS NOT NULL AND event_type IN ({MM_SCOPE_EVENT_TYPES}) ORDER BY 1",
            multi=True,
        ),
        _query_var("run_id", "Run", "SELECT DISTINCT run_id FROM mm_events WHERE run_id <> '' ORDER BY 1 DESC", multi=True),
        _query_var("diagnostics_topic", "Diagnostics topic", "SELECT DISTINCT topic FROM mm_events WHERE topic LIKE 'strategy:diagnostics:%' ORDER BY 1"),
        _query_var("target_books_topic", "Target books topic", "SELECT DISTINCT topic FROM mm_events WHERE topic LIKE 'strategy:target_books:%' ORDER BY 1"),
        _query_var(
            "interval_ms",
            "Interval ms",
            _interval_ms_query(),
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
        "sort": 1,
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


def _interval_ms_query() -> str:
    interval_expr = "COALESCE(NULLIF(payload->>'interval_ms', ''), substring(topic from ':([0-9]+)$'))"
    return (
        f"SELECT DISTINCT {interval_expr} AS interval_ms "
        f"FROM mm_events WHERE {interval_expr} IS NOT NULL ORDER BY 1"
    )


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
    metrics_sql = _mm_scope_pair_metric_sql()
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


def _latest_action_pipeline_stage_orders_sql(
    stage: str,
    *,
    field: str = "orders",
    reason: str | None = None,
    include_reason_in_series: bool = False,
) -> str:
    side = _side_sql("o.value->>'side'")
    series = side
    if include_reason_in_series:
        series = f"(COALESCE(NULLIF(o.value->>'reason', ''), 'unknown') || ' ' || {side})"
    price = _numeric_sql("o.value->>'price'", default="NULL")
    qty = _numeric_sql("o.value->>'qty'")
    axis_price = _numeric_sql("axis_o.value->>'price'", default="NULL")
    axis_qty = _numeric_sql("axis_o.value->>'qty'")
    reason_filter = ""
    if reason is not None:
        reason_filter = f"\n    AND COALESCE(o.value->>'reason', '') = '{reason}'"
    return f"""WITH latest AS (
  SELECT e.payload
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
),
orders AS (
  SELECT {series} AS series,
         {price} AS price,
         {qty} AS value
  FROM latest l
  CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(l.payload #> ARRAY['action_pipeline', '{stage}', '{field}'], '[]'::jsonb)
  ) AS o(value)
  WHERE {side} IN ('bid', 'ask')
    AND {price} IS NOT NULL{reason_filter}
),
axis_points AS (
  SELECT {axis_price} AS price,
         {axis_qty} AS value
  FROM latest l
  CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(l.payload #> ARRAY['action_pipeline', 'ladder', 'orders'], '[]'::jsonb)
  ) AS axis_o(value)
  WHERE {axis_price} IS NOT NULL
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


def _target_to_execution_stage_counts_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  stage.metric,
  SUM(stage.value) AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES
    ('raw ladder', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','ladder','order_count']")}),
    ('strategy-capped target', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','capped','order_count']")}),
    ('target book', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','target_book','order_count']")}),
    ('risk-passed target', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','risk_approved_target','order_count']")}),
    ('kept live orders', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','kept','order_count']")}),
    ('new order candidates', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','planned_place','order_count']")}),
    ('exchange-rule valid', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','rules_valid','order_count']")}),
    ('exchange accepted', {_numeric_sql("e.payload #>> ARRAY['action_pipeline','router_accepted','order_count']")})
) AS stage(metric, value)
WHERE {_diagnostics_filter()}
GROUP BY 1, 2"""
    metrics_sql = """  SELECT metric
  FROM (
    VALUES
      ('raw ladder'),
      ('strategy-capped target'),
      ('target book'),
      ('risk-passed target'),
      ('kept live orders'),
      ('new order candidates'),
      ('exchange-rule valid'),
      ('exchange accepted')
  ) AS m(metric)"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


def _target_to_execution_filter_reason_sql() -> str:
    step_source = _event_step_source_sql("e.payload->>'t_end_ms'", "e.ts")
    reason_payload = """COALESCE(
    e.payload #> ARRAY['action_pipeline', stages.stage, 'rejection_counts'],
    CASE
      WHEN stages.stage = 'risk_approved_target'
      THEN e.payload #> ARRAY['risk','rejection_counts']
      ELSE NULL::jsonb
    END,
    '{}'::jsonb
  )"""
    stages_sql = """VALUES
    ('ladder', 'raw ladder'),
    ('capped', 'strategy cap'),
    ('risk_approved_target', 'risk'),
    ('rules_valid', 'exchange rules'),
    ('router_accepted', 'router')"""
    data_sql = f"""SELECT
  {_approx_step_time_alias_sql(step_source)},
  (stages.stage_label || ': ' || reason.key) AS metric,
  SUM({_numeric_sql("reason.value", default="0.0")}) AS value
FROM mm_events e
CROSS JOIN LATERAL (
  {stages_sql}
) AS stages(stage, stage_label)
CROSS JOIN LATERAL jsonb_each_text({reason_payload}) AS reason(key, value)
WHERE {_diagnostics_filter()}
GROUP BY 1, 2"""
    metrics_sql = f"""  SELECT DISTINCT
         (stages.stage_label || ': ' || reason.key) AS metric
  FROM mm_events e
  CROSS JOIN LATERAL (
    {stages_sql}
  ) AS stages(stage, stage_label)
  CROSS JOIN LATERAL jsonb_each_text({reason_payload}) AS reason(key, value)
  WHERE {_diagnostics_filter()}
  UNION
  SELECT 'no filter reasons'::text AS metric
  WHERE NOT EXISTS (
    SELECT 1
    FROM mm_events e
    CROSS JOIN LATERAL (
      {stages_sql}
    ) AS stages(stage, stage_label)
    CROSS JOIN LATERAL jsonb_each_text({reason_payload}) AS reason(key, value)
    WHERE {_diagnostics_filter()}
  )"""
    return _zero_filled_step_sql(data_sql, metrics_sql)


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


def market_data_quality() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Raw orderbook/trade feed health before interval aggregation, plus local/global price-volume comparison.", y=0),
        _panel(2, "Market events / step", "timeseries", PG_DS, _market_event_step_sql(), x=0, y=3, w=6, unit="ops"),
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
        _stem_order_qty_price_panel(
            5,
            "Marginal ladder",
            _latest_action_pipeline_stage_orders_sql("ladder"),
            x=0,
            y=27,
            w=24,
        ),
        _stem_order_qty_price_panel(
            6,
            "After venue rounding",
            _latest_action_pipeline_stage_orders_sql("rules_valid"),
            x=0,
            y=37,
            w=24,
        ),
        _stem_order_qty_price_panel(
            7,
            "Exchange-accepted placements",
            _latest_action_pipeline_stage_orders_sql("router_accepted"),
            x=0,
            y=47,
            w=24,
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


def model_quality() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Model NLL and latest model scores emitted from `ControllerDiagnostics.models`.", y=0),
        _panel(2, "Model NLL / step", "timeseries", PG_DS, _model_nll_step_sql(), x=0, y=3, w=24),
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


def target_order_stems() -> dict[str, Any]:
    stem_h = 7
    stem_gap = 7
    panels = [
        _text(
            1,
            "Scope",
            "Live or pinned strategy-to-execution ladder. Latest mode follows the newest ControllerDiagnostics event; pinned mode uses the selected target frame. Every chart uses the same raw-ladder price/qty frame; bid is green, ask is red, and hidden black axis anchors keep scales aligned.",
            y=0,
        ),
        _stem_order_qty_price_panel(2, "01 Strategy raw ladder", _latest_action_pipeline_stage_orders_sql("ladder"), x=0, y=3, w=24, h=stem_h),
        _stem_order_qty_price_panel(3, "02 Strategy cap -> target orderbook", _latest_action_pipeline_stage_orders_sql("capped"), x=0, y=3 + stem_gap, w=24, h=stem_h),
        _stem_order_qty_price_panel(4, "03 Risk-approved target", _latest_action_pipeline_stage_orders_sql("risk_approved_target"), x=0, y=3 + stem_gap * 2, w=24, h=stem_h),
        _stem_order_qty_price_panel(5, "04 Planner new place orders", _latest_action_pipeline_stage_orders_sql("planned_place"), x=0, y=3 + stem_gap * 3, w=24, h=stem_h),
        _stem_order_qty_price_panel(6, "05 Venue rules valid after rounding", _latest_action_pipeline_stage_orders_sql("rules_valid"), x=0, y=3 + stem_gap * 4, w=24, h=stem_h),
        _stem_order_qty_price_panel(7, "06 Exchange accepted placed orders", _latest_action_pipeline_stage_orders_sql("router_accepted"), x=0, y=3 + stem_gap * 5, w=24, h=stem_h),
        _stem_order_qty_price_panel(8, "07 Planner kept live orders", _latest_action_pipeline_stage_orders_sql("kept"), x=0, y=3 + stem_gap * 6, w=24, h=stem_h),
        _stem_order_qty_price_panel(9, "08 Planner cancel orders", _latest_action_pipeline_stage_orders_sql("planned_cancel"), x=0, y=3 + stem_gap * 7, w=24, h=stem_h),
        _stem_order_qty_price_panel(10, "09 Removed by local risk / collateral", _latest_action_pipeline_stage_orders_sql("risk_approved_target", field="rejections", include_reason_in_series=True), x=0, y=3 + stem_gap * 8, w=24, h=stem_h),
        _stem_order_qty_price_panel(11, "10 Removed by venue adapter: invalid price/qty", _latest_action_pipeline_stage_orders_sql("rules_valid", field="rejections", reason="rules_invalid_price_or_qty"), x=0, y=3 + stem_gap * 9, w=24, h=stem_h),
        _stem_order_qty_price_panel(12, "11 Removed by venue adapter: min qty", _latest_action_pipeline_stage_orders_sql("rules_valid", field="rejections", reason="rules_min_qty"), x=0, y=3 + stem_gap * 10, w=24, h=stem_h),
        _stem_order_qty_price_panel(13, "12 Removed by venue adapter: min notional", _latest_action_pipeline_stage_orders_sql("rules_valid", field="rejections", reason="rules_min_notional"), x=0, y=3 + stem_gap * 11, w=24, h=stem_h),
    ]
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


def _orderbook_depth_sql(kind: str) -> str:
    if kind == "raw":
        raw_bid_price = _numeric_sql("m.payload #>> ARRAY['orderbook','bids','0','price']", default="NULL")
        raw_ask_price = _numeric_sql("m.payload #>> ARRAY['orderbook','asks','0','price']", default="NULL")
        raw_t_end_ms = _numeric_sql("m.payload->>'t_end_ms'", default="NULL")
        selected = f"""selected AS (
  SELECT COALESCE(
           NULLIF(ctx.mid, 0.0),
           (
             {raw_bid_price} +
             {raw_ask_price}
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
        selected = """selected AS (
  SELECT ctx.mid, ctx.spread_max, ctx.diag_payload AS payload
  FROM ctx
)"""
        bid_levels = f"""bid_levels AS (
  SELECT {_numeric_sql("b->>'price'", default="NULL")} AS price,
         {_numeric_sql("b->>'qty'")} AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'cleared_orderbook'->'bids', '[]'::jsonb)) AS b
)"""
        ask_levels = f"""ask_levels AS (
  SELECT {_numeric_sql("a->>'price'", default="NULL")} AS price,
         {_numeric_sql("a->>'qty'")} AS qty
  FROM selected, LATERAL jsonb_array_elements(COALESCE(payload->'cleared_orderbook'->'asks', '[]'::jsonb)) AS a
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
    return f"""WITH {_orderbook_context_cte()},
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


def _orderbook_context_cte() -> str:
    t_end_ms = _numeric_sql("e.payload->>'t_end_ms'")
    state_mid = _numeric_sql("e.payload #>> ARRAY['state','mid']", default="NULL")
    clean_bid = _numeric_sql("e.payload #>> ARRAY['cleared_orderbook','bids','0','price']", default="NULL")
    clean_ask = _numeric_sql("e.payload #>> ARRAY['cleared_orderbook','asks','0','price']", default="NULL")
    spread_max = _numeric_sql("'${spread_max}'", default="100.0")
    return f"""ctx AS (
  SELECT e.ts AS diag_ts,
         e.payload AS diag_payload,
         e.exchange,
         e.symbol,
         e.actor_id,
         {t_end_ms} AS t_end_ms,
         COALESCE(
           {state_mid},
           (
             {clean_bid} +
             {clean_ask}
           ) / 2.0,
           0.0
         ) AS mid,
         GREATEST({spread_max}, 1.0) AS spread_max
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


def _optimizer_contrib_sql() -> str:
    value = _numeric_sql("kv.value", default="NULL")
    return f"""SELECT e.ts AS "time", kv.key AS metric, {value} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(COALESCE(e.payload #> ARRAY['optimizer','objective','contribution_fractions'], '{{}}'::jsonb)) AS kv
WHERE {_diagnostics_filter()}
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


def _portfolio_total_value_sql() -> str:
    local_cash = _numeric_sql("e.payload #>> ARRAY['state','local_cash']")
    global_cash = _numeric_sql("e.payload #>> ARRAY['state','global_cash']")
    local_inventory = _numeric_sql("e.payload #>> ARRAY['state','local_inventory']")
    global_inventory = _numeric_sql("e.payload #>> ARRAY['state','global_inventory']")
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
        model_quality(),
        optimizer_diagnostics(),
        action_pipeline(),
        target_order_stems(),
        action_pipeline_side_counts(),
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
