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
            _interval_ms_query(),
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


def _zero_series_sql(metric: str) -> str:
    return f"""SELECT {_time_bucket_sql("$__timeFrom()")} AS "time", '{metric}' AS metric, 0.0 AS value
UNION ALL
SELECT {_time_bucket_sql("$__timeTo()")} AS "time", '{metric}' AS metric, 0.0 AS value"""


def _numeric_sql(expr: str, default: str = "0.0") -> str:
    return f"""CASE
  WHEN trim(COALESCE({expr}, '')) ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$'
  THEN trim(COALESCE({expr}, ''))::double precision
  ELSE {default}
END"""


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
    return f"""SELECT
  {_time_bucket_alias_sql("e.ts")},
  ((e.payload #>> ARRAY['local_market','exchange']) || ' ' || (e.payload #>> ARRAY['local_market','symbol'])) AS metric,
  COUNT(*)::double precision / {_rate_denominator_sql()} AS value
FROM mm_events e
WHERE e.event_type = 'JointStrategyFrame'
  AND (e.payload #>> ARRAY['local_market','exchange']) IN (${{exchange:sqlstring}})
  AND (e.payload #>> ARRAY['local_market','symbol']) IN (${{symbol:sqlstring}})
  AND $__timeFilter(e.ts)
GROUP BY 1, 2
UNION ALL
{_zero_series_sql("no frames")}
ORDER BY 1"""


def _runtime_decision_rate_sql() -> str:
    return f"""SELECT
  {_time_bucket_alias_sql("e.ts")},
  (e.actor_id || ' ' || e.exchange || ' ' || e.symbol) AS metric,
  COUNT(*)::double precision / {_rate_denominator_sql()} AS value
FROM mm_events e
WHERE e.event_type = 'StrategyDecision'
  AND e.exchange IN (${{exchange:sqlstring}})
  AND e.symbol IN (${{symbol:sqlstring}})
  AND e.actor_id IN (${{actor_id:sqlstring}})
  AND $__timeFilter(e.ts)
GROUP BY 1, 2
UNION ALL
{_zero_series_sql("no decisions")}
ORDER BY 1"""


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
UNION ALL
{_zero_series_sql("no stage timings")}
ORDER BY 1"""


def _runtime_order_sync_p95_sql() -> str:
    latency = _numeric_sql("e.payload->>'latency_ms'", default="NULL")
    return f"""SELECT
  {_time_bucket_alias_sql("e.ts")},
  COALESCE(e.exchange, 'unknown') AS metric,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY {latency}) AS value
FROM mm_events e
WHERE e.event_type IN ('ExecutionSyncCompleted', 'ExecutionSyncFailed')
  AND e.exchange IN (${{exchange:sqlstring}})
  AND (e.symbol IN (${{symbol:sqlstring}}) OR e.symbol IS NULL)
  AND e.payload->>'latency_ms' IS NOT NULL
  AND trim(e.payload->>'latency_ms') ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$'
  AND $__timeFilter(e.ts)
GROUP BY 1, 2
UNION ALL
{_zero_series_sql("no sync latency")}
ORDER BY 1"""


def _runtime_execution_event_rate_sql(event_type: str, zero_metric: str) -> str:
    return f"""SELECT
  {_time_bucket_alias_sql("e.ts")},
  COALESCE(e.exchange, 'unknown') AS metric,
  COUNT(*)::double precision / {_rate_denominator_sql()} AS value
FROM mm_events e
WHERE e.event_type = '{event_type}'
  AND e.exchange IN (${{exchange:sqlstring}})
  AND (e.symbol IN (${{symbol:sqlstring}}) OR e.symbol IS NULL)
  AND $__timeFilter(e.ts)
GROUP BY 1, 2
UNION ALL
{_zero_series_sql(zero_metric)}
ORDER BY 1"""


def _runtime_order_payload_rate_sql(event_type: str, measure: str, *, include_reason: bool = False) -> str:
    side = _side_sql("o.value->>'side'")
    qty = _numeric_sql("COALESCE(o.value->>'qty', o.value->>'remaining_qty')")
    value = "1.0" if measure == "count" else qty
    reason = " || ' ' || COALESCE(NULLIF(e.payload->>'error', ''), 'unknown')" if include_reason else ""
    zero_metric = "no failed orders" if include_reason else "no placed orders"
    return f"""SELECT
  {_time_bucket_alias_sql("e.ts")},
  (COALESCE(o.value->>'exchange', e.exchange, 'unknown') || ' ' ||
   COALESCE(o.value->>'symbol', e.symbol, 'unknown') || ' ' ||
   {side}{reason}) AS metric,
  SUM({value}) / {_rate_denominator_sql()} AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements(COALESCE(e.payload->'orders', '[]'::jsonb)) AS o(value)
WHERE e.event_type = '{event_type}'
  AND COALESCE(o.value->>'exchange', e.exchange, '') IN (${{exchange:sqlstring}})
  AND COALESCE(o.value->>'symbol', e.symbol, '') IN (${{symbol:sqlstring}})
  AND $__timeFilter(e.ts)
GROUP BY 1, 2
UNION ALL
{_zero_series_sql(zero_metric)}
ORDER BY 1"""


def _runtime_fill_rate_sql(measure: str) -> str:
    side = _side_sql("e.payload->>'side'")
    qty = _numeric_sql("e.payload->>'qty'")
    value = "1.0" if measure == "count" else qty
    zero_metric = "no fills" if measure == "count" else "no filled qty"
    return f"""SELECT
  {_time_bucket_alias_sql("e.ts")},
  (COALESCE(e.exchange, 'unknown') || ' ' || COALESCE(e.symbol, 'unknown') || ' ' || {side}) AS metric,
  SUM({value}) / {_rate_denominator_sql()} AS value
FROM mm_events e
WHERE e.topic LIKE 'private:state:%:fills'
  AND e.exchange IN (${{exchange:sqlstring}})
  AND e.symbol IN (${{symbol:sqlstring}})
  AND $__timeFilter(e.ts)
GROUP BY 1, 2
UNION ALL
{_zero_series_sql(zero_metric)}
ORDER BY 1"""


def _runtime_live_orders_sql(measure: str) -> str:
    side = _side_sql("open_order.value #>> ARRAY['order','side']")
    qty = _numeric_sql("open_order.value->>'remaining_qty'")
    value = "COUNT(*)::double precision" if measure == "count" else "SUM(qty)"
    zero_metric = "no live orders" if measure == "count" else "no live order qty"
    return f"""WITH expanded AS (
  SELECT
    e.ts,
    COALESCE(e.exchange, 'unknown') AS exchange,
    COALESCE(open_order.value #>> ARRAY['order','symbol'], e.symbol, 'unknown') AS symbol,
    {side} AS side,
    {qty} AS qty
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_each(COALESCE(e.payload->'live_orders', '{{}}'::jsonb)) AS open_order(key, value)
  WHERE e.topic LIKE 'execution:state:%'
    AND e.exchange IN (${{exchange:sqlstring}})
    AND COALESCE(open_order.value #>> ARRAY['order','symbol'], e.symbol, '') IN (${{symbol:sqlstring}})
    AND $__timeFilter(e.ts)
),
per_snapshot AS (
  SELECT
    ts,
    exchange,
    symbol,
    side,
    {value} AS value
  FROM expanded
  GROUP BY ts, exchange, symbol, side
)
SELECT
  {_time_bucket_alias_sql("ts")},
  (exchange || ' ' || symbol || ' ' || side) AS metric,
  AVG(value) AS value
FROM per_snapshot
GROUP BY 1, 2
UNION ALL
{_zero_series_sql(zero_metric)}
ORDER BY 1"""


def runtime_overview() -> dict[str, Any]:
    panels = [
        _text(1, "Scope", "Runtime health from persisted events. Prometheus scrape status is shown separately so a failed `/metrics` socket does not make the whole overview blank.", y=0),
        _panel(2, "Prometheus scrape up", "stat", PROM_DS, 'max(up{job="market_maker"}) or vector(0)', x=0, y=3, w=6, h=4, legend="{{job}}"),
        _panel(3, "Frame rate", "timeseries", PG_DS, _runtime_frame_rate_sql(), x=6, y=3),
        _panel(4, "Decision rate", "timeseries", PG_DS, _runtime_decision_rate_sql(), x=18, y=3, w=6),
        _panel(5, "Controller stage p95 ms", "timeseries", PG_DS, _runtime_controller_stage_p95_sql(), x=0, y=11, unit="ms"),
        _panel(6, "Order sync p95 ms", "timeseries", PG_DS, _runtime_order_sync_p95_sql(), x=12, y=11, w=6, unit="ms"),
        _panel(15, "Sync requests / sec", "timeseries", PG_DS, _runtime_execution_event_rate_sql("ExecutionSyncRequested", "no sync requests"), x=18, y=11, w=6),
        _panel(7, "Placed orders / sec", "timeseries", PG_DS, _runtime_order_payload_rate_sql("PlaceOrdersAccepted", "count"), x=0, y=19, w=8),
        _panel(8, "Failed orders / sec", "timeseries", PG_DS, _runtime_order_payload_rate_sql("PlaceOrdersFailed", "count", include_reason=True), x=8, y=19, w=8),
        _panel(9, "Fill events / sec", "timeseries", PG_DS, _runtime_fill_rate_sql("count"), x=16, y=19, w=8),
        _panel(10, "Filled qty / sec", "timeseries", PG_DS, _runtime_fill_rate_sql("qty"), x=0, y=27, w=8),
        _panel(11, "Live orders by side", "timeseries", PG_DS, _runtime_live_orders_sql("count"), x=8, y=27, w=8),
        _panel(12, "Live order qty by side", "timeseries", PG_DS, _runtime_live_orders_sql("qty"), x=16, y=27, w=8),
        _panel(13, "Persistence errors", "timeseries", LOKI_DS, 'sum by (writer,error) (count_over_time({job="market-maker-runtime"} | json | event="persistence_batch_failed" [$__interval]))', x=0, y=35, legend="{{writer}} {{error}}"),
        _panel(14, "Latest controller warnings", "table", PG_DS, _latest_warnings_sql(), x=12, y=35, fmt="table"),
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
        _text(
            1,
            "Scope",
            "Joint target-book execution path. The main funnel is intentionally collapsed to the runtime contract: desired target book, risk-approved target, and router-accepted orders. Planner internals are summarized in the table.",
            y=0,
        ),
        _panel(2, "Desired target orders", "timeseries", PG_DS, _target_book_count_sql(), x=0, y=3, w=12),
        _panel(3, "Desired target qty", "timeseries", PG_DS, _target_book_qty_sql(), x=12, y=3, w=12),
        _panel(4, "Execution contract funnel", "timeseries", PG_DS, _action_contract_funnel_sql(), x=0, y=11, w=12),
        _panel(5, "Blocked/rejected reasons", "timeseries", PG_DS, _action_rejection_reason_sql(), x=12, y=11, w=12),
        _panel(6, "Latest execution summary", "table", PG_DS, _action_execution_summary_sql(), x=0, y=19, w=24, fmt="table"),
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
WHERE event_type = 'MarketEvent' AND exchange IN (${{exchange:sqlstring}}) AND symbol IN (${{symbol:sqlstring}}) AND $__timeFilter(ts)
UNION ALL
SELECT ts AS "time", (exchange || ' best_ask') AS metric,
       {best_ask} AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange IN (${{exchange:sqlstring}}) AND symbol IN (${{symbol:sqlstring}}) AND $__timeFilter(ts)
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
  AND m.exchange IN (${{exchange:sqlstring}})
  AND m.symbol IN (${{symbol:sqlstring}})
  AND (m.payload->'orderbook'->'bids'->0->>'price') IS NOT NULL
  AND $__timeFilter(m.ts)
UNION ALL
SELECT m.ts AS "time", (m.exchange || ' best_ask') AS metric,
       {best_ask} AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange IN (${{exchange:sqlstring}})
  AND m.symbol IN (${{symbol:sqlstring}})
  AND (m.payload->'orderbook'->'asks'->0->>'price') IS NOT NULL
  AND $__timeFilter(m.ts)
UNION ALL
SELECT e.ts AS "time", (r.exchange || ' buy_vwap') AS metric,
       {buy_vwap} AS value
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
       {sell_vwap} AS value
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
    buy_volume = _numeric_sql("m.payload->>'buy_volume'", default="NULL")
    sell_volume = _numeric_sql("m.payload->>'sell_volume'", default="NULL")
    return f"""WITH {_role_exchange_cte()}
SELECT m.ts AS "time", (m.exchange || ' buy_volume') AS metric,
       {buy_volume} AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange IN (${{exchange:sqlstring}})
  AND m.symbol IN (${{symbol:sqlstring}})
  AND (m.payload->>'buy_volume') IS NOT NULL
  AND $__timeFilter(m.ts)
UNION ALL
SELECT m.ts AS "time", (m.exchange || ' sell_volume') AS metric,
       {sell_volume} AS value
FROM mm_events m
JOIN role_exchange r ON r.role = '{role_key}' AND r.exchange = m.exchange AND r.symbol = m.symbol
WHERE m.event_type = 'MarketEvent'
  AND m.exchange IN (${{exchange:sqlstring}})
  AND m.symbol IN (${{symbol:sqlstring}})
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
  AND exchange IN (${{exchange:sqlstring}})
  AND symbol IN (${{symbol:sqlstring}})
  AND $__timeFilter(ts)
ORDER BY 1"""


def _volume_sql() -> str:
    buy_volume = _numeric_sql("payload->>'buy_volume'", default="NULL")
    sell_volume = _numeric_sql("payload->>'sell_volume'", default="NULL")
    return f"""SELECT ts AS "time", (exchange || ' buy_volume') AS metric, {buy_volume} AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange IN (${{exchange:sqlstring}}) AND symbol IN (${{symbol:sqlstring}}) AND $__timeFilter(ts)
UNION ALL
SELECT ts AS "time", (exchange || ' sell_volume') AS metric, {sell_volume} AS value
FROM mm_events
WHERE event_type = 'MarketEvent' AND exchange IN (${{exchange:sqlstring}}) AND symbol IN (${{symbol:sqlstring}}) AND $__timeFilter(ts)
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
    ('venue_quantized', 'venue limits'),
    ('capped', 'order shaping'),
    ('risk_approved_target', 'risk'),
    ('rules_valid', 'exchange rules'),
    ('router_accepted', 'router')
) AS stages(stage, stage_label)
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(e.payload #> ARRAY['action_pipeline', stage, 'rejection_counts'], '{{}}'::jsonb)
) AS reason(key, value)
WHERE {_diagnostics_filter()}
ORDER BY 1"""


def _action_execution_summary_sql() -> str:
    desired = _numeric_sql("e.payload #>> ARRAY['action_pipeline','target_book','order_count']")
    approved = _numeric_sql("e.payload #>> ARRAY['action_pipeline','risk_approved_target','order_count']")
    accepted = _numeric_sql("e.payload #>> ARRAY['action_pipeline','router_accepted','order_count']")
    new_orders = _numeric_sql("e.payload #>> ARRAY['action_pipeline','planned_place','order_count']")
    cancel_orders = _numeric_sql("e.payload #>> ARRAY['action_pipeline','planned_cancel','order_count']")
    kept_orders = _numeric_sql("e.payload #>> ARRAY['action_pipeline','kept','order_count']")
    return f"""SELECT e.ts,
       e.actor_id,
       e.exchange,
       e.symbol,
       {desired} AS desired_orders,
       {approved} AS risk_approved_orders,
       {accepted} AS router_accepted_orders,
       {new_orders} AS new_order_commands,
       {cancel_orders} AS cancel_commands,
       {kept_orders} AS unchanged_existing_orders,
       COALESCE(e.payload #> ARRAY['action_pipeline','risk_approved_target','rejection_counts'], '{{}}'::jsonb) AS risk_blocks,
       COALESCE(e.payload #> ARRAY['action_pipeline','rules_valid','rejection_counts'], '{{}}'::jsonb) AS rule_blocks,
       COALESCE(e.payload #> ARRAY['action_pipeline','router_accepted','rejection_counts'], '{{}}'::jsonb) AS router_blocks
FROM mm_events e
WHERE {_diagnostics_filter()}
ORDER BY e.ts DESC
LIMIT 50"""


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
    qty = _numeric_sql("o->>'qty'")
    return f"""SELECT e.ts AS "time", (venue || ' ' || (o->>'side')) AS metric,
       SUM({qty}) AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES ('local', e.payload->'local'), ('global', e.payload->'global')
) AS v(venue, book)
CROSS JOIN LATERAL jsonb_array_elements(COALESCE(book->'orders', '[]'::jsonb)) AS o
WHERE e.topic = '${{target_books_topic}}'
  AND e.event_type = 'JointTargetBook'
  AND e.exchange IN (${{exchange:sqlstring}})
  AND e.symbol IN (${{symbol:sqlstring}})
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
