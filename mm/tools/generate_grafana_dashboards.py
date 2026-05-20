#!/usr/bin/env python3
"""Generate Grafana dashboards for controller diagnostics, portfolio, and hedge execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "grafana" / "dashboards"
PG_DS = {"type": "postgres", "uid": "postgres-mm"}


def _panel_text(id_: int, y: int, content: str, title: str = "Notes") -> dict[str, Any]:
    return {
        "gridPos": {"h": 4, "w": 24, "x": 0, "y": y},
        "id": id_,
        "options": {"content": content, "mode": "markdown"},
        "title": title,
        "transparent": True,
        "type": "text",
    }


def _panel_sql_table(id_: int, y: int, title: str, sql: str, h: int = 8) -> dict[str, Any]:
    return {
        "datasource": PG_DS,
        "fieldConfig": {"defaults": {"custom": {"align": "auto"}}, "overrides": []},
        "gridPos": {"h": h, "w": 24, "x": 0, "y": y},
        "id": id_,
        "options": {"cellHeight": "sm", "footer": {"countRows": False}, "showHeader": True},
        "targets": [
            {
                "datasource": PG_DS,
                "editorMode": "code",
                "format": "table",
                "rawQuery": True,
                "rawSql": sql,
                "refId": "A",
            }
        ],
        "title": title,
        "type": "table",
    }


def _panel_sql_timeseries(id_: int, y: int, title: str, sql: str, h: int = 8, w: int = 24, x: int = 0) -> dict[str, Any]:
    return {
        "datasource": PG_DS,
        "fieldConfig": {"defaults": {"unit": "short", "decimals": 4}, "overrides": []},
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": id_,
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "none"},
        },
        "targets": [
            {
                "datasource": PG_DS,
                "editorMode": "code",
                "format": "time_series",
                "rawQuery": True,
                "rawSql": sql,
                "refId": "A",
            }
        ],
        "title": title,
        "type": "timeseries",
    }


def _panel_row(id_: int, y: int, title: str) -> dict[str, Any]:
    return {
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "id": id_,
        "panels": [],
        "title": title,
        "type": "row",
    }


def _dashboard(
    *,
    title: str,
    uid: str,
    description: str,
    tags: list[str],
    panels: list[dict[str, Any]],
    templating: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "annotations": {"list": []},
        "description": description,
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
        "tags": tags,
        "templating": {"list": templating},
        "time": {"from": "now-24h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": title,
        "uid": uid,
        "version": 1,
        "weekStart": "",
    }


def _panel_objective_contribution_fractions(
    id_: int,
    y: int,
    *,
    diag_filter: str,
    h: int = 8,
    w: int = 24,
    x: int = 0,
) -> dict[str, Any]:
    return {
        "datasource": PG_DS,
        "fieldConfig": {
            "defaults": {"unit": "percentunit", "decimals": 2, "min": 0, "max": 1},
            "overrides": [],
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": id_,
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "targets": [
            {
                "datasource": PG_DS,
                "editorMode": "code",
                "format": "time_series",
                "rawQuery": True,
                "rawSql": f"""SELECT e.ts AS "time",
  comp.key AS metric,
  (comp.value)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(
    e.payload #> ARRAY['optimizer','objective','contribution_fractions'],
    e.payload #> ARRAY['optimizer','objective','final','contribution_fractions'],
    '{{}}'::jsonb
  )
) AS comp(key, value)
WHERE {diag_filter}
ORDER BY 1""",
                "refId": "A",
            }
        ],
        "title": "Objective contribution fractions — final (|weighted term| share)",
        "type": "timeseries",
    }


def _panel_objective_component(
    id_: int,
    y: int,
    *,
    title: str,
    component_key: str,
    diag_filter: str,
    x: int = 0,
    w: int = 12,
    h: int = 6,
) -> dict[str, Any]:
    return _panel_sql_timeseries(
        id_,
        y,
        title,
        f"""SELECT e.ts AS "time", stage AS metric,
  (e.payload #>> ARRAY['optimizer','objective',stage,'components','{component_key}'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['initial','final','best']) AS stage
WHERE {diag_filter}
  AND e.payload #>> ARRAY['optimizer','objective',stage,'components','{component_key}'] IS NOT NULL
ORDER BY 1""",
        h=h,
        w=w,
        x=x,
    )


def controller_diagnostics_dashboard() -> dict[str, Any]:
    diag_filter = """
  e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.exchange = '${exchange}'
  AND e.symbol = '${symbol}'
  AND e.actor_id = '${actor_id}'
  AND $__timeFilter(e.ts)"""

    panels: list[dict[str, Any]] = [
        _panel_text(
            1,
            0,
            "**Source:** `mm_events` (`ControllerDiagnostics` on **Diagnostics topic**).\n\n"
            "Covers optimizer objectives, action pipeline funnel, execution gates, and risk. "
            "Set variables to match `controller` / persistence config.",
        ),
        _panel_sql_table(
            2,
            4,
            "Latest controller status",
            f"""SELECT
  e.ts,
  (e.payload->>'frame_step')::int AS frame_step,
  e.payload #>> '{{frame,hedge_status}}' AS hedge_status,
  (e.payload #>> '{{execution_gates,executed}}')::boolean AS executed,
  e.payload #>> '{{execution_gates,blocked_reasons}}' AS blocked_reasons,
  e.payload->'risk'->'rejection_counts' AS risk_rejection_counts,
  (e.payload->'optimizer'->'timing'->>'total_sec')::double precision AS optimizer_total_sec,
  (e.payload->'optimizer'->'timing'->>'best_epoch')::int AS best_epoch
FROM mm_events e
WHERE {diag_filter}
ORDER BY e.ts DESC
LIMIT 1""",
            h=6,
        ),
        _panel_sql_timeseries(
            10,
            10,
            "Objective value (initial / final / best)",
            f"""SELECT e.ts AS "time", stage AS metric,
  (e.payload #>> ARRAY['optimizer','objective',stage,'value'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['initial','final','best']) AS stage
WHERE {diag_filter}
  AND e.payload #>> ARRAY['optimizer','objective',stage,'value'] IS NOT NULL
ORDER BY 1""",
        ),
        _panel_row(11, 18, "Objective components"),
        _panel_objective_component(
            12,
            19,
            title="expected_pnl",
            component_key="expected_pnl",
            diag_filter=diag_filter,
            x=0,
        ),
        _panel_objective_component(
            13,
            19,
            title="inventory_target_penalty",
            component_key="inventory_target_penalty",
            diag_filter=diag_filter,
            x=12,
        ),
        _panel_objective_component(
            14,
            25,
            title="pnl_volatility_penalty",
            component_key="pnl_volatility_penalty",
            diag_filter=diag_filter,
            x=0,
        ),
        _panel_objective_component(
            15,
            25,
            title="quote_shape_concentration_penalty",
            component_key="quote_shape_concentration_penalty",
            diag_filter=diag_filter,
            x=12,
        ),
        _panel_objective_component(
            16,
            31,
            title="quote_size_target_penalty",
            component_key="quote_size_target_penalty",
            diag_filter=diag_filter,
            x=0,
        ),
        _panel_objective_component(
            17,
            31,
            title="quote_spread_penalty",
            component_key="quote_spread_penalty",
            diag_filter=diag_filter,
            x=12,
        ),
        _panel_objective_contribution_fractions(
            18,
            37,
            diag_filter=diag_filter,
        ),
        _panel_sql_timeseries(
            30,
            45,
            "Action pipeline — order count by stage",
            f"""SELECT e.ts AS "time", stage AS metric,
  (e.payload #> ARRAY['action_pipeline',stage,'order_count'])::int::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY[
  'ladder','venue_quantized','capped','risk_approved','target','kept','canceled','placed','venue_accepted'
]) AS stage
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            31,
            53,
            "Execution gate blocks (by reason)",
            f"""SELECT e.ts AS "time", reason.value AS metric, 1::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements_text(
  COALESCE(e.payload->'execution_gates'->'blocked_reasons', '[]'::jsonb)
) AS reason
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            32,
            61,
            "Risk rejection counts",
            f"""SELECT e.ts AS "time", kv.key AS metric, (kv.value)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(COALESCE(e.payload->'risk'->'rejection_counts', '{{}}'::jsonb)) kv
WHERE {diag_filter}
ORDER BY 1""",
        ),
    ]

    templating = [
        {
            "current": {"text": "mpc-controller", "value": "mpc-controller"},
            "label": "Actor ID",
            "name": "actor_id",
            "query": "mpc-controller",
            "type": "textbox",
        },
        {
            "current": {"text": "nobitex", "value": "nobitex"},
            "label": "Exchange",
            "name": "exchange",
            "options": [
                {"selected": True, "text": "nobitex", "value": "nobitex"},
                {"selected": False, "text": "wallex", "value": "wallex"},
            ],
            "query": "nobitex,wallex",
            "type": "custom",
        },
        {
            "current": {"text": "USDT_IRT", "value": "USDT_IRT"},
            "label": "Symbol",
            "name": "symbol",
            "query": "USDT_IRT",
            "type": "textbox",
        },
        {
            "current": {"text": "strategy:diagnostics:mpc-controller", "value": "strategy:diagnostics:mpc-controller"},
            "label": "Diagnostics topic",
            "name": "diagnostics_topic",
            "query": "strategy:diagnostics:mpc-controller",
            "type": "textbox",
        },
        {
            "current": {"text": "10000", "value": "10000"},
            "label": "Interval ms",
            "name": "interval_ms",
            "query": "10000",
            "type": "textbox",
        },
    ]
    return _dashboard(
        title="MM controller diagnostics (PostgreSQL)",
        uid="mm-controller-diagnostics-pg",
        description="ControllerDiagnostics from mm_events: optimizer, pipeline, risk, execution gates.",
        tags=["market-maker", "postgres", "controller", "diagnostics"],
        panels=panels,
        templating=templating,
    )


def portfolio_dashboard() -> dict[str, Any]:
    diag_filter = """
  e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.symbol = '${symbol}'
  AND e.actor_id = '${actor_id}'
  AND $__timeFilter(e.ts)"""

    panels: list[dict[str, Any]] = [
        _panel_text(
            1,
            0,
            "**Source:** `ControllerDiagnostics.state` and `risk.limits`.\n\n"
            "Inventory/cash from diagnostics frames; limits from configured hard risk.",
        ),
        _panel_sql_timeseries(
            10,
            4,
            "Inventory (primary vs hedge)",
            f"""SELECT e.ts AS "time", 'primary_inventory' AS metric,
  (e.payload #>> '{{state,primary_inventory}}')::double precision AS value
FROM mm_events e
WHERE {diag_filter}
UNION ALL
SELECT e.ts, 'hedge_inventory',
  (e.payload #>> '{{state,hedge_inventory}}')::double precision
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            11,
            12,
            "Cash (primary vs hedge)",
            f"""SELECT e.ts AS "time", 'primary_cash' AS metric,
  (e.payload #>> '{{state,primary_cash}}')::double precision AS value
FROM mm_events e
WHERE {diag_filter}
UNION ALL
SELECT e.ts, 'hedge_cash',
  (e.payload #>> '{{state,hedge_cash}}')::double precision
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            12,
            20,
            "Marked portfolio value (approx)",
            f"""SELECT e.ts AS "time", 'primary_marked' AS metric,
  ((e.payload #>> '{{state,primary_cash}}')::double precision
   + (e.payload #>> '{{state,primary_inventory}}')::double precision
     * COALESCE((e.payload #>> '{{state,primary_sell_vwap}}')::double precision, 0)) AS value
FROM mm_events e
WHERE {diag_filter}
UNION ALL
SELECT e.ts, 'hedge_marked',
  ((e.payload #>> '{{state,hedge_cash}}')::double precision
   + (e.payload #>> '{{state,hedge_inventory}}')::double precision
     * COALESCE((e.payload #>> '{{state,hedge_sell_vwap}}')::double precision, 0)) AS value
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            13,
            28,
            "Inventory limits (primary)",
            f"""SELECT e.ts AS "time", lim AS metric, v AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES
    ('min', (e.payload #>> ARRAY['risk','limits','primary','min_inventory'])::double precision),
    ('max', (e.payload #>> ARRAY['risk','limits','primary','max_inventory'])::double precision)
) AS t(lim, v)
WHERE {diag_filter} AND v IS NOT NULL
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            14,
            36,
            "Combined inventory exposure",
            f"""SELECT e.ts AS "time", 'primary_plus_hedge' AS metric,
  ((e.payload #>> '{{state,primary_inventory}}')::double precision
   + (e.payload #>> '{{state,hedge_inventory}}')::double precision) AS value
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        ),
    ]
    templating = [
        {
            "current": {"text": "nobitex", "value": "nobitex"},
            "label": "Primary exchange",
            "name": "primary_exchange",
            "query": "nobitex,wallex",
            "type": "custom",
        },
        {
            "current": {"text": "wallex", "value": "wallex"},
            "label": "Hedge exchange",
            "name": "hedge_exchange",
            "query": "nobitex,wallex",
            "type": "custom",
        },
        {
            "current": {"text": "USDT_IRT", "value": "USDT_IRT"},
            "label": "Symbol",
            "name": "symbol",
            "query": "USDT_IRT",
            "type": "textbox",
        },
        {
            "current": {"text": "mpc-controller", "value": "mpc-controller"},
            "label": "Actor ID",
            "name": "actor_id",
            "query": "mpc-controller",
            "type": "textbox",
        },
        {
            "current": {"text": "strategy:diagnostics:mpc-controller", "value": "strategy:diagnostics:mpc-controller"},
            "label": "Diagnostics topic",
            "name": "diagnostics_topic",
            "query": "strategy:diagnostics:mpc-controller",
            "type": "textbox",
        },
    ]
    return _dashboard(
        title="MM portfolio (PostgreSQL)",
        uid="mm-portfolio-pg",
        description="Portfolio state and limits from ControllerDiagnostics.",
        tags=["market-maker", "postgres", "portfolio"],
        panels=panels,
        templating=templating,
    )


def hedge_execution_dashboard() -> dict[str, Any]:
    hedge_diag = """
  e.topic = '${hedge_diagnostics_topic}'
  AND e.event_type = 'HedgeExecutionDiagnostic'
  AND e.symbol = '${symbol}'
  AND e.actor_id = '${actor_id}'
  AND $__timeFilter(e.ts)"""

    panels: list[dict[str, Any]] = [
        _panel_text(
            1,
            0,
            "**Source:** `HedgeIntent` and `HedgeExecutionDiagnostic` on hedge topics.\n\n"
            "Live fill-driven hedging (separate from MPC `hedge_rollout_enabled`).",
        ),
        _panel_sql_timeseries(
            10,
            4,
            "Hedge outcomes by status",
            f"""SELECT date_trunc('minute', e.ts) AS "time", e.payload->>'status' AS metric,
  COUNT(*)::double precision AS value
FROM mm_events e
WHERE {hedge_diag}
GROUP BY 1, 2
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            11,
            12,
            "Residual unhedged qty",
            f"""SELECT e.ts AS "time", 'unhedged_qty' AS metric,
  (e.payload #>> '{{residual,unhedged_qty}}')::double precision AS value
FROM mm_events e
WHERE {hedge_diag}
  AND e.payload #>> '{{residual,unhedged_qty}}' IS NOT NULL
ORDER BY 1""",
        ),
        _panel_sql_table(
            12,
            20,
            "Rejection reasons (count)",
            f"""SELECT COALESCE(e.payload->>'reason', 'unknown') AS reason,
  COUNT(*) AS count
FROM mm_events e
WHERE {hedge_diag}
  AND e.payload->>'status' IN ('rejected', 'failed', 'skipped')
GROUP BY 1
ORDER BY count DESC""",
            h=8,
        ),
        _panel_sql_table(
            13,
            28,
            "Latest hedge trace",
            f"""SELECT e.ts,
  e.payload->>'source_fill_id' AS source_fill_id,
  e.payload->>'status' AS status,
  e.payload->>'reason' AS reason,
  e.payload #>> '{{intent,primary_side}}' AS primary_side,
  e.payload #>> '{{intent,hedge_side}}' AS hedge_side,
  (e.payload #>> '{{intent,qty}}')::double precision AS qty,
  e.payload #>> '{{hedge_order,client_order_id}}' AS hedge_client_order_id
FROM mm_events e
WHERE {hedge_diag}
ORDER BY e.ts DESC
LIMIT 20""",
            h=10,
        ),
    ]
    templating = [
        {
            "current": {"text": "hedge-executor", "value": "hedge-executor"},
            "label": "Actor ID",
            "name": "actor_id",
            "query": "hedge-executor",
            "type": "textbox",
        },
        {
            "current": {"text": "nobitex", "value": "nobitex"},
            "label": "Primary exchange",
            "name": "primary_exchange",
            "query": "nobitex,wallex",
            "type": "custom",
        },
        {
            "current": {"text": "wallex", "value": "wallex"},
            "label": "Hedge exchange",
            "name": "hedge_exchange",
            "query": "nobitex,wallex",
            "type": "custom",
        },
        {
            "current": {"text": "USDT_IRT", "value": "USDT_IRT"},
            "label": "Symbol",
            "name": "symbol",
            "query": "USDT_IRT",
            "type": "textbox",
        },
        {
            "current": {"text": "hedge:diagnostics:hedge-executor", "value": "hedge:diagnostics:hedge-executor"},
            "label": "Hedge diagnostics topic",
            "name": "hedge_diagnostics_topic",
            "query": "hedge:diagnostics:hedge-executor",
            "type": "textbox",
        },
        {
            "current": {"text": "hedge:intents:hedge-executor", "value": "hedge:intents:hedge-executor"},
            "label": "Hedge intents topic",
            "name": "hedge_intents_topic",
            "query": "hedge:intents:hedge-executor",
            "type": "textbox",
        },
    ]
    return _dashboard(
        title="MM hedge execution (PostgreSQL)",
        uid="mm-hedge-execution-pg",
        description="Fill-driven hedge intents and execution diagnostics.",
        tags=["market-maker", "postgres", "hedge"],
        panels=panels,
        templating=templating,
    )


def patch_model_params_dashboard() -> None:
    path = OUT / "mm_strategy_model_params_pg.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    panels = data["panels"]
    y = max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in panels)

    diag_filter = """
  e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.exchange = '${exchange}'
  AND e.symbol = '${symbol}'
  AND $__timeFilter(e.ts)"""

    panels.append(_panel_row(100, y, "Model scores (ControllerDiagnostics)"))
    y += 1
    panels.append(
        _panel_sql_timeseries(
            101,
            y,
            "Model NLL",
            f"""SELECT e.ts AS "time", model AS metric,
  (e.payload #>> ARRAY['models', model, 'nll'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['price','market_flow','hedge_price']) AS model
WHERE {diag_filter}
  AND e.payload #>> ARRAY['models', model, 'nll'] IS NOT NULL
ORDER BY 1""",
        )
    )
    y += 8
    panels.append(
        _panel_sql_timeseries(
            102,
            y,
            "Rolling mean NLL (5 frames)",
            f"""SELECT "time", metric, value FROM (
  SELECT e.ts AS "time", model AS metric,
    AVG((e.payload #>> ARRAY['models', model, 'nll'])::double precision)
      OVER (PARTITION BY model ORDER BY e.ts ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)) AS value
  FROM mm_events e
  CROSS JOIN LATERAL unnest(ARRAY['price','market_flow','hedge_price']) AS model
  WHERE {diag_filter}
    AND e.payload #>> ARRAY['models', model, 'nll'] IS NOT NULL
) s
WHERE value IS NOT NULL
ORDER BY 1""",
        )
    )
    y += 8
    panels.append(
        _panel_sql_timeseries(
            103,
            y,
            "Standardized residuals",
            f"""SELECT e.ts AS "time", model || '.z' AS metric,
  (e.payload #>> ARRAY['models', model, 'standardized_residual'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['price','market_flow','hedge_price']) AS model
WHERE {diag_filter}
  AND e.payload #>> ARRAY['models', model, 'standardized_residual'] IS NOT NULL
ORDER BY 1""",
        )
    )
    y += 8
    panels.append(
        _panel_sql_table(
            104,
            y,
            "Score skip reasons",
            f"""SELECT model AS model,
  COALESCE(e.payload #>> ARRAY['models', model, 'skip_reason'], '') AS skip_reason,
  COUNT(*) AS count
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['price','market_flow','hedge_price']) AS model
WHERE {diag_filter}
  AND e.payload #>> ARRAY['models', model, 'skip_reason'] IS NOT NULL
  AND e.payload #>> ARRAY['models', model, 'skip_reason'] <> ''
GROUP BY 1, 2
ORDER BY count DESC""",
        )
    )

    templating = data.setdefault("templating", {}).setdefault("list", [])
    if not any(t.get("name") == "diagnostics_topic" for t in templating):
        templating.append(
            {
                "current": {
                    "text": "strategy:diagnostics:mpc-controller",
                    "value": "strategy:diagnostics:mpc-controller",
                },
                "label": "Diagnostics topic",
                "name": "diagnostics_topic",
                "query": "strategy:diagnostics:mpc-controller",
                "type": "textbox",
            }
        )
    data["panels"] = panels
    data["version"] = int(data.get("version", 1)) + 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mm_controller_diagnostics_pg.json").write_text(
        json.dumps(controller_diagnostics_dashboard(), indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "mm_portfolio_pg.json").write_text(
        json.dumps(portfolio_dashboard(), indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "mm_hedge_execution_pg.json").write_text(
        json.dumps(hedge_execution_dashboard(), indent=2) + "\n",
        encoding="utf-8",
    )
    patch_model_params_dashboard()
    print("Wrote grafana dashboards to", OUT)


if __name__ == "__main__":
    main()
