#!/usr/bin/env python3
"""Generate Grafana dashboards for controller diagnostics, local/global action, portfolio, and liquidity sourcing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "grafana" / "dashboards"
PG_DS = {"type": "postgres", "uid": "postgres-mm"}

# Grafana grid row height scale (1 unit ≈ 30px). Halving makes more panels visible without scrolling.
PANEL_HEIGHT_SCALE = 0.5
PANEL_MIN_HEIGHT = 2


def _scale_panel_height(height: int, *, scale: float | None = None) -> int:
    factor = PANEL_HEIGHT_SCALE if scale is None else scale
    if height <= 1:
        return height
    return max(PANEL_MIN_HEIGHT, round(height * factor))


def compact_dashboard_panels(data: dict[str, Any], *, height_scale: float | None = None) -> dict[str, Any]:
    """Scale panel heights and reflow y positions row-by-row."""

    panels = data.get("panels") or []
    if not panels:
        return data

    rows: dict[int, list[dict[str, Any]]] = {}
    for panel in panels:
        grid = panel.setdefault("gridPos", {})
        rows.setdefault(int(grid.get("y", 0)), []).append(panel)

    new_y = 0
    for old_y in sorted(rows):
        row_panels = rows[old_y]
        for panel in row_panels:
            grid = panel["gridPos"]
            grid["h"] = _scale_panel_height(int(grid.get("h", 8)), scale=height_scale)
            grid["y"] = new_y
        new_y += max(int(panel["gridPos"]["h"]) for panel in row_panels)

    data["panels"] = panels
    return data


def _apply_standard_panel_heights(panels: list[dict[str, Any]]) -> None:
    """Set readable Grafana heights before reflow (1 grid unit ≈ 30px)."""

    for panel in panels:
        grid = panel.setdefault("gridPos", {})
        ptype = panel.get("type")
        if ptype == "row":
            grid["h"] = 1
        elif ptype == "text":
            grid["h"] = 4
        elif ptype == "table":
            grid["h"] = 6
        elif ptype == "timeseries":
            grid["h"] = 8


def compact_dashboard_file(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    compact_dashboard_panels(data)
    data["version"] = int(data.get("version", 1)) + 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compact_all_dashboards() -> None:
    for path in sorted(OUT.glob("*.json")):
        compact_dashboard_file(path)
        print("Compacted", path.name)


def _panel_text(id_: int, y: int, content: str, title: str = "Notes", *, h: int = 4) -> dict[str, Any]:
    return {
        "gridPos": {"h": h, "w": 24, "x": 0, "y": y},
        "id": id_,
        "options": {"content": content, "mode": "markdown"},
        "title": title,
        "transparent": True,
        "type": "text",
    }


def _panel_sql_table(id_: int, y: int, title: str, sql: str, h: int = 6) -> dict[str, Any]:
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


def _pg_query_var(name: str, label: str, query: str, *, multi: bool = False) -> dict[str, Any]:
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


def _textbox_var(name: str, label: str, value: str) -> dict[str, Any]:
    return {
        "current": {"text": value, "value": value},
        "label": label,
        "name": name,
        "query": value,
        "type": "textbox",
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
        "title": "Objective contribution fractions",
        "type": "timeseries",
    }


_OPTIMIZER_OBJECTIVE_STAGES = "ARRAY['initial','final']"

_DISCRETIZATION_STAGES = "ARRAY['ladder','venue_quantized','capped']"
_EXECUTION_PLAN_STAGES = "ARRAY['kept','planned_cancel','planned_place','rules_valid','router_accepted']"


def _panel_pipeline_order_counts(
    id_: int,
    y: int,
    *,
    title: str,
    stages_sql: str,
    diag_filter: str,
    h: int = 6,
    w: int = 24,
    x: int = 0,
) -> dict[str, Any]:
    return _panel_sql_timeseries(
        id_,
        y,
        title,
        f"""SELECT e.ts AS "time", stage AS metric,
  COALESCE((e.payload #>> ARRAY['action_pipeline', stage, 'order_count'])::double precision, 0) AS value
FROM mm_events e
CROSS JOIN LATERAL unnest({stages_sql}) AS stage
WHERE {diag_filter}
ORDER BY 1""",
        h=h,
        w=w,
        x=x,
    )


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
CROSS JOIN LATERAL unnest({_OPTIMIZER_OBJECTIVE_STAGES}) AS stage
WHERE {diag_filter}
  AND e.payload #>> ARRAY['optimizer','objective',stage,'components','{component_key}'] IS NOT NULL
ORDER BY 1""",
        h=h,
        w=w,
        x=x,
    )


def _controller_diagnostics_filter() -> str:
    return """
  e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.exchange = '${exchange}'
  AND e.symbol = '${symbol}'
  AND e.actor_id = '${actor_id}'
  AND $__timeFilter(e.ts)"""


def _ledger_templating() -> list[dict[str, Any]]:
    return [
        _pg_query_var(
            "ledger_topic",
            "Ledger topic",
            "SELECT DISTINCT topic FROM mm_events WHERE event_type = 'JointLedgerSnapshot' ORDER BY 1",
        ),
        _pg_query_var(
            "symbol",
            "Symbol",
            "SELECT DISTINCT symbol FROM mm_events WHERE symbol IS NOT NULL ORDER BY 1",
        ),
        _pg_query_var(
            "local_exchange",
            "Local exchange",
            "SELECT DISTINCT payload #>> '{local,exchange}' FROM mm_events WHERE event_type = 'JointLedgerSnapshot' ORDER BY 1",
        ),
        _pg_query_var(
            "global_exchange",
            "Global exchange",
            "SELECT DISTINCT payload #>> '{global,exchange}' FROM mm_events WHERE event_type = 'JointLedgerSnapshot' AND payload #>> '{global,exchange}' IS NOT NULL ORDER BY 1",
        ),
    ]


def _ledger_filter() -> str:
    return """
  e.topic = '${ledger_topic}'
  AND e.event_type = 'JointLedgerSnapshot'
  AND e.payload #>> '{local,symbol}' = '${symbol}'
  AND e.payload #>> '{local,exchange}' = '${local_exchange}'
  AND $__timeFilter(e.ts)"""


def _joint_target_book_filter() -> str:
    return """
  e.event_type = 'JointTargetBook'
  AND e.topic = '${target_books_topic}'
  AND e.symbol = '${symbol}'
  AND COALESCE(e.actor_id, e.payload->>'actor_id') = '${actor_id}'
  AND $__timeFilter(e.ts)"""


def _controller_diagnostics_templating() -> list[dict[str, Any]]:
    return [
        _pg_query_var("actor_id", "Actor ID", "SELECT DISTINCT actor_id FROM mm_events WHERE actor_id IS NOT NULL ORDER BY 1"),
        _pg_query_var("exchange", "Exchange", "SELECT DISTINCT exchange FROM mm_events WHERE exchange IS NOT NULL ORDER BY 1"),
        _pg_query_var("symbol", "Symbol", "SELECT DISTINCT symbol FROM mm_events WHERE symbol IS NOT NULL ORDER BY 1"),
        _pg_query_var("diagnostics_topic", "Diagnostics topic", "SELECT DISTINCT topic FROM mm_events WHERE topic LIKE 'strategy:diagnostics:%' ORDER BY 1"),
        _pg_query_var("target_books_topic", "Target books topic", "SELECT DISTINCT topic FROM mm_events WHERE topic LIKE 'strategy:target_books:%' ORDER BY 1"),
        _pg_query_var(
            "interval_ms",
            "Interval ms",
            "SELECT DISTINCT COALESCE(payload->>'interval_ms', split_part(topic, ':', 5)) AS interval_ms FROM mm_events WHERE COALESCE(payload->>'interval_ms', split_part(topic, ':', 5)) <> '' ORDER BY 1",
        ),
    ]


def controller_diagnostics_dashboard() -> dict[str, Any]:
    diag_filter = _controller_diagnostics_filter()

    panels: list[dict[str, Any]] = [
        _panel_sql_timeseries(
            10,
            0,
            "Objective value (initial / final)",
            f"""SELECT e.ts AS "time", stage AS metric,
  (e.payload #>> ARRAY['optimizer','objective',stage,'value'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest({_OPTIMIZER_OBJECTIVE_STAGES}) AS stage
WHERE {diag_filter}
  AND e.payload #>> ARRAY['optimizer','objective',stage,'value'] IS NOT NULL
ORDER BY 1""",
        ),
        _panel_objective_contribution_fractions(
            11,
            8,
            diag_filter=diag_filter,
        ),
        _panel_objective_component(
            15,
            16,
            title="risk_adjusted_return",
            component_key="risk_adjusted_return",
            diag_filter=diag_filter,
            w=24,
            h=8,
        ),
        _panel_objective_component(
            12,
            24,
            title="inventory_capacity_penalty",
            component_key="inventory_capacity_penalty",
            diag_filter=diag_filter,
            w=24,
            h=8,
        ),
        _panel_objective_component(
            16,
            32,
            title="local_inventory_penalty",
            component_key="local_inventory_penalty",
            diag_filter=diag_filter,
            w=12,
            h=6,
        ),
        _panel_objective_component(
            17,
            32,
            title="global_inventory_penalty",
            component_key="global_inventory_penalty",
            diag_filter=diag_filter,
            w=12,
            h=6,
            x=12,
        ),
        _panel_sql_timeseries(
            18,
            38,
            "Consolidated wealth means (main / global reward)",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> ARRAY['optimizer','objective',stage,field])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest({_OPTIMIZER_OBJECTIVE_STAGES}) AS stage
CROSS JOIN (
  VALUES ('main_reward_mean', 'main_reward_mean'), ('global_reward_mean', 'global_reward_mean')
) AS fields(metric, field)
WHERE {diag_filter}
  AND e.payload #>> ARRAY['optimizer','objective',stage,field] IS NOT NULL
ORDER BY 1""",
            h=6,
        ),
        _panel_sql_timeseries(
            19,
            44,
            "Joint MPC expected fills (rollout)",
            f"""SELECT e.ts AS "time", kv.key AS metric,
  (kv.value)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(e.payload #> ARRAY['optimizer','rollout','joint_diagnostics'], '{{}}'::jsonb)
) AS kv
WHERE {diag_filter}
ORDER BY 1""",
            h=6,
        ),
        _panel_sql_timeseries(
            13,
            50,
            "Expected total return (mean MC path return)",
            f"""SELECT e.ts AS "time", stage AS metric,
  (e.payload #>> ARRAY['optimizer','objective',stage,'expected_total_return'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest({_OPTIMIZER_OBJECTIVE_STAGES}) AS stage
WHERE {diag_filter}
  AND e.payload #>> ARRAY['optimizer','objective',stage,'expected_total_return'] IS NOT NULL
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            14,
            58,
            "Expected total reward (mean MC delta wealth)",
            f"""SELECT e.ts AS "time", stage AS metric,
  (e.payload #>> ARRAY['optimizer','objective',stage,'expected_total_reward'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest({_OPTIMIZER_OBJECTIVE_STAGES}) AS stage
WHERE {diag_filter}
  AND e.payload #>> ARRAY['optimizer','objective',stage,'expected_total_reward'] IS NOT NULL
ORDER BY 1""",
        ),
    ]

    return _dashboard(
        title="MM controller diagnostics (PostgreSQL)",
        uid="mm-controller-diagnostics-pg",
        description="ControllerDiagnostics optimizer objective metrics from mm_events.",
        tags=["market-maker", "postgres", "controller", "diagnostics"],
        panels=panels,
        templating=_controller_diagnostics_templating(),
    )


def _panel_action_pipeline_rejections(
    id_: int,
    y: int,
    *,
    stage: str,
    title: str,
    diag_filter: str,
    rejection_counts_json: str | None = None,
    h: int = 6,
) -> dict[str, Any]:
    counts_expr = (
        rejection_counts_json
        if rejection_counts_json is not None
        else f"e.payload #> ARRAY['action_pipeline','{stage}','rejection_counts']"
    )
    return _panel_sql_timeseries(
        id_,
        y,
        title,
        f"""SELECT e.ts AS "time",
  kv.key AS metric,
  (kv.value)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(
  COALESCE({counts_expr}, '{{}}'::jsonb)
) AS kv
WHERE {diag_filter}
ORDER BY 1""",
        h=h,
    )


def local_action_dashboard() -> dict[str, Any]:
    """Execution pipeline: joint target books → risk → ledger diff → rules → router."""

    diag_filter = _controller_diagnostics_filter()
    target_filter = _joint_target_book_filter()

    panels: list[dict[str, Any]] = [
        _panel_text(
            1,
            0,
            "**Sources:** `JointTargetBook` on **Target books topic**, plus `ControllerDiagnostics.action_pipeline`.\n\n"
            "If target-book charts are empty, verify **Target books topic** is in `persistence.topics` "
            "and check **Controller warnings** for strategy errors (empty pipeline stages mean `decide()` failed).",
        ),
        _panel_row(2, 4, "Joint target books (strategy output)"),
        _panel_sql_timeseries(
            10,
            5,
            "Target book order count by venue",
            f"""SELECT e.ts AS "time", 'local' AS metric,
  jsonb_array_length(COALESCE(e.payload->'local'->'orders', '[]'::jsonb))::double precision AS value
FROM mm_events e
WHERE {target_filter}
UNION ALL
SELECT e.ts, 'global',
  jsonb_array_length(COALESCE(e.payload->'global'->'orders', '[]'::jsonb))::double precision
FROM mm_events e
WHERE {target_filter}
ORDER BY 1""",
            w=12,
        ),
        _panel_sql_timeseries(
            11,
            5,
            "Target book buy/sell qty by venue",
            f"""SELECT e.ts AS "time",
  books.venue || '_' || COALESCE(books.ord->>'side', '') AS metric,
  SUM((books.ord->>'qty')::double precision) AS value
FROM mm_events e
CROSS JOIN LATERAL (
  SELECT 'local' AS venue, ord
  FROM jsonb_array_elements(COALESCE(e.payload->'local'->'orders', '[]'::jsonb)) AS ord
  UNION ALL
  SELECT 'global', ord
  FROM jsonb_array_elements(COALESCE(e.payload->'global'->'orders', '[]'::jsonb)) AS ord
) AS books(venue, ord)
WHERE {target_filter}
GROUP BY 1, 2
ORDER BY 1""",
            w=12,
            x=12,
        ),
        _panel_sql_timeseries(
            20,
            13,
            "Target book order count (diagnostics fallback)",
            f"""SELECT e.ts AS "time", 'target_book_total' AS metric,
  COALESCE((e.payload #>> ARRAY['action_pipeline','target_book','order_count'])::double precision, 0) AS value
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_table(
            21,
            21,
            "Controller warnings (latest)",
            f"""SELECT e.ts AS ts,
  COALESCE(e.payload->'warnings'->>0, '') AS warning
FROM mm_events e
WHERE {diag_filter}
  AND jsonb_array_length(COALESCE(e.payload->'warnings', '[]'::jsonb)) > 0
ORDER BY e.ts DESC
LIMIT 20""",
            h=6,
        ),
        _panel_row(3, 27, "Local curve discretization (MPC → orders)"),
        _panel_pipeline_order_counts(
            12,
            28,
            title="Discretization funnel (local venue only)",
            stages_sql=_DISCRETIZATION_STAGES,
            diag_filter=diag_filter,
        ),
        _panel_row(4, 36, "Net-delta risk"),
        _panel_sql_timeseries(
            13,
            37,
            "Risk-approved order count",
            f"""SELECT e.ts AS "time", 'risk_approved' AS metric,
  COALESCE((e.payload #>> ARRAY['action_pipeline','risk_approved_target','order_count'])::double precision, 0) AS value
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
            w=12,
        ),
        _panel_sql_timeseries(
            14,
            37,
            "Risk approved flag",
            f"""SELECT e.ts AS "time", 'approved' AS metric,
  CASE WHEN COALESCE((e.payload->'risk'->>'approved')::boolean, true) THEN 1 ELSE 0 END::double precision AS value
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
            w=12,
            x=12,
        ),
        _panel_row(5, 45, "Execution plan (ledger diff, both venues merged)"),
        _panel_pipeline_order_counts(
            15,
            46,
            title="Planner output: keep / cancel / place / rules / router",
            stages_sql=_EXECUTION_PLAN_STAGES,
            diag_filter=diag_filter,
            h=8,
        ),
        _panel_sql_timeseries(
            16,
            40,
            "Execution plan notional (planned place buy/sell)",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> ARRAY['action_pipeline','planned_place', field])::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES ('buy_notional', 'buy_notional'), ('sell_notional', 'sell_notional')
) AS legs(metric, field)
WHERE {diag_filter}
ORDER BY 1""",
            w=12,
        ),
        _panel_sql_timeseries(
            17,
            40,
            "Orders kept vs canceled",
            f"""SELECT e.ts AS "time", stage AS metric,
  COALESCE((e.payload #>> ARRAY['action_pipeline', stage, 'order_count'])::double precision, 0) AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['kept', 'planned_cancel']) AS stage
WHERE {diag_filter}
ORDER BY 1""",
            w=12,
            x=12,
        ),
        _panel_row(6, 48, "Live execution gates"),
        _panel_sql_timeseries(
            18,
            49,
            "Live sync executed",
            f"""SELECT e.ts AS "time", 'executed' AS metric,
  CASE WHEN COALESCE((e.payload->'execution_gates'->>'executed')::boolean, false) THEN 1 ELSE 0 END::double precision AS value
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
            w=12,
        ),
        _panel_sql_timeseries(
            19,
            49,
            "Venue gate live flags",
            f"""SELECT e.ts AS "time", gate.key AS metric,
  CASE WHEN (gate.value->>'live')::boolean THEN 1 ELSE 0 END::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(
  COALESCE(e.payload->'execution_gates'->'venue_gates', '{{}}'::jsonb)
) AS gate(key, value)
WHERE {diag_filter}
ORDER BY 1""",
            w=12,
            x=12,
        ),
    ]

    return _dashboard(
        title="MM execution pipeline (PostgreSQL)",
        uid="mm-local-action-pg",
        description="Joint target book → net-delta risk → ledger execution plan from ControllerDiagnostics.",
        tags=["market-maker", "postgres", "controller", "execution", "target-book"],
        panels=panels,
        templating=_controller_diagnostics_templating(),
    )


def local_action_funnel_rejections_dashboard() -> dict[str, Any]:
    diag_filter = _controller_diagnostics_filter()

    risk_rejection_expr = (
        "COALESCE("
        "e.payload #> ARRAY['action_pipeline','risk_approved_target','rejection_counts'], "
        "e.payload->'risk'->'rejection_counts'"
        ")"
    )

    panels: list[dict[str, Any]] = [
        _panel_text(
            1,
            0,
            "**Source:** `ControllerDiagnostics` rejection counts and execution gate reasons.\n\n"
            "**Discretization:** local curve → ladder → venue rules → capped.\n"
            "**Target-book risk:** frame gates (stale data, warmup) and net-delta collateral limits.\n"
            "**Rules gateway:** tick size, min qty, min notional.\n"
            "**Live gates:** observe-only, venue paused, stale global market, etc.",
        ),
        _panel_row(2, 4, "Discretization rejections (local venue)"),
        _panel_action_pipeline_rejections(
            20,
            5,
            stage="venue_quantized",
            title="Venue quantization rejections",
            diag_filter=diag_filter,
        ),
        _panel_action_pipeline_rejections(
            21,
            11,
            stage="capped",
            title="Per-side cap rejections",
            diag_filter=diag_filter,
        ),
        _panel_row(3, 17, "Target-book / net-delta risk rejections"),
        _panel_action_pipeline_rejections(
            22,
            18,
            stage="risk_approved_target",
            title="Risk rejection counts",
            diag_filter=diag_filter,
            rejection_counts_json=risk_rejection_expr,
            h=8,
        ),
        _panel_row(4, 26, "Exchange rules rejections"),
        _panel_action_pipeline_rejections(
            23,
            27,
            stage="rules_valid",
            title="Rules gateway rejections",
            diag_filter=diag_filter,
            h=8,
        ),
        _panel_row(5, 35, "Live execution blocked reasons"),
        _panel_sql_timeseries(
            24,
            36,
            "Top-level blocked reasons (count per frame)",
            f"""SELECT e.ts AS "time", reason AS metric, 1::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_array_elements_text(
  COALESCE(e.payload->'execution_gates'->'blocked_reasons', '[]'::jsonb)
) AS reason
WHERE {diag_filter}
ORDER BY 1""",
            h=6,
        ),
        _panel_sql_table(
            25,
            42,
            "Venue gate blocked reasons (latest frame)",
            f"""SELECT gate.key AS exchange,
  COALESCE(gate.value->>'role', '') AS role,
  COALESCE(gate.value->>'live', '') AS live,
  COALESCE(gate.value->'blocked_reasons', '[]'::jsonb)::text AS blocked_reasons
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(
  COALESCE(e.payload->'execution_gates'->'venue_gates', '{{}}'::jsonb)
) AS gate(key, value)
WHERE {diag_filter}
ORDER BY e.ts DESC
LIMIT 20""",
            h=6,
        ),
    ]

    return _dashboard(
        title="MM execution pipeline rejections (PostgreSQL)",
        uid="mm-local-action-rejections-pg",
        description="Discretization, net-delta risk, rules, and live-gate rejection reasons.",
        tags=["market-maker", "postgres", "controller", "execution", "rejections"],
        panels=panels,
        templating=_controller_diagnostics_templating(),
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
            "**Source:** `ControllerDiagnostics.state`, `risk.exposure_*`, and `risk.limits`.\n\n"
            "Inventory/cash and wallet legs use **effective totals** (exchange total, wallet locked, "
            "and open-order collateral when locked under-reports). Net-delta risk exposure shows "
            "ledger locked collateral vs target-book locked collateral per venue.",
        ),
        _panel_sql_timeseries(
            10,
            4,
            "Inventory (local vs global)",
            f"""SELECT e.ts AS "time", 'local_inventory' AS metric,
  (e.payload #>> '{{state,local_inventory}}')::double precision AS value
FROM mm_events e
WHERE {diag_filter}
UNION ALL
SELECT e.ts, 'global_inventory',
  (e.payload #>> '{{state,global_inventory}}')::double precision
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            11,
            12,
            "Cash (local vs global)",
            f"""SELECT e.ts AS "time", 'local_cash' AS metric,
  (e.payload #>> '{{state,local_cash}}')::double precision AS value
FROM mm_events e
WHERE {diag_filter}
UNION ALL
SELECT e.ts, 'global_cash',
  (e.payload #>> '{{state,global_cash}}')::double precision
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            12,
            20,
            "Marked portfolio value (approx)",
            f"""SELECT e.ts AS "time", 'local_marked' AS metric,
  COALESCE((e.payload #>> '{{state,local_wealth}}')::double precision,
    (e.payload #>> '{{state,local_cash}}')::double precision
     + (e.payload #>> '{{state,local_inventory}}')::double precision
       * COALESCE((e.payload #>> '{{state,local_sell_vwap}}')::double precision, 0)) AS value
FROM mm_events e
WHERE {diag_filter}
UNION ALL
SELECT e.ts, 'global_marked',
  COALESCE((e.payload #>> '{{state,global_wealth}}')::double precision,
    (e.payload #>> '{{state,global_cash}}')::double precision
     + (e.payload #>> '{{state,global_inventory}}')::double precision
       * COALESCE((e.payload #>> '{{state,global_sell_vwap}}')::double precision, 0)) AS value
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            15,
            28,
            "Local quote wallet (free / locked / total / effective)",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> ARRAY['state','local_quote_wallet', leg])::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES ('free', 'free'), ('locked', 'locked'), ('total', 'total'), ('effective_total', 'effective_total')
) AS legs(metric, leg)
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            16,
            36,
            "Local base wallet (free / locked / total / effective)",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> ARRAY['state','local_base_wallet', leg])::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES ('free', 'free'), ('locked', 'locked'), ('total', 'total'), ('effective_total', 'effective_total')
) AS legs(metric, leg)
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            13,
            44,
            "Inventory limits (local)",
            f"""SELECT e.ts AS "time", lim AS metric, v AS value
FROM mm_events e
CROSS JOIN LATERAL (
  VALUES
    ('min', (e.payload #>> ARRAY['risk','limits','local','min_inventory'])::double precision),
    ('max', (e.payload #>> ARRAY['risk','limits','local','max_inventory'])::double precision)
) AS t(lim, v)
WHERE {diag_filter} AND v IS NOT NULL
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            14,
            52,
            "Combined inventory exposure",
            f"""SELECT e.ts AS "time", 'local_plus_global' AS metric,
  ((e.payload #>> '{{state,local_inventory}}')::double precision
   + (e.payload #>> '{{state,global_inventory}}')::double precision) AS value
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            17,
            60,
            "Global quote wallet (free / locked / total / effective)",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> ARRAY['state','global_quote_wallet', leg])::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES ('free', 'free'), ('locked', 'locked'), ('total', 'total'), ('effective_total', 'effective_total')
) AS legs(metric, leg)
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            18,
            68,
            "Global base wallet (free / locked / total / effective)",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> ARRAY['state','global_base_wallet', leg])::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES ('free', 'free'), ('locked', 'locked'), ('total', 'total'), ('effective_total', 'effective_total')
) AS legs(metric, leg)
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            20,
            76,
            "Locked collateral — before risk (ledger)",
            f"""SELECT e.ts AS "time", kv.key AS metric, (kv.value)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(e.payload->'risk'->'exposure_before', '{{}}'::jsonb)
) AS kv
WHERE {diag_filter}
  AND kv.key LIKE '%locked%'
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            21,
            84,
            "Locked collateral — after risk (target book)",
            f"""SELECT e.ts AS "time", kv.key AS metric, (kv.value)::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each_text(
  COALESCE(e.payload->'risk'->'exposure_after', '{{}}'::jsonb)
) AS kv
WHERE {diag_filter}
  AND (kv.key LIKE '%locked%' OR kv.key LIKE 'consolidated_%')
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            22,
            92,
            "Consolidated ledger exposure",
            f"""SELECT e.ts AS "time", stage || '_' || key AS metric,
  (e.payload #>> exposure_path)::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES
    ('before', 'consolidated_base_inventory'),
    ('before', 'consolidated_quote_wealth'),
    ('after', 'consolidated_base_inventory'),
    ('after', 'consolidated_quote_wealth')
) AS legs(stage, key)
CROSS JOIN LATERAL (
  SELECT ARRAY['risk', 'exposure_' || stage, key] AS exposure_path
) AS paths
WHERE {diag_filter}
  AND e.payload #>> exposure_path IS NOT NULL
ORDER BY 1""",
        ),
    ]
    templating = [
        _pg_query_var(
            "local_exchange",
            "Local exchange",
            "SELECT DISTINCT payload #>> '{local,exchange}' FROM mm_events WHERE event_type = 'JointLedgerSnapshot' ORDER BY 1",
        ),
        _pg_query_var(
            "global_exchange",
            "Global exchange",
            "SELECT DISTINCT payload #>> '{global,exchange}' FROM mm_events WHERE event_type = 'JointLedgerSnapshot' AND payload #>> '{global,exchange}' IS NOT NULL ORDER BY 1",
        ),
        _pg_query_var("symbol", "Symbol", "SELECT DISTINCT symbol FROM mm_events WHERE symbol IS NOT NULL ORDER BY 1"),
        _pg_query_var("actor_id", "Actor ID", "SELECT DISTINCT actor_id FROM mm_events WHERE actor_id IS NOT NULL ORDER BY 1"),
        _pg_query_var("diagnostics_topic", "Diagnostics topic", "SELECT DISTINCT topic FROM mm_events WHERE topic LIKE 'strategy:diagnostics:%' ORDER BY 1"),
    ]
    return _dashboard(
        title="MM portfolio (PostgreSQL)",
        uid="mm-portfolio-pg",
        description="Portfolio state and limits from ControllerDiagnostics.",
        tags=["market-maker", "postgres", "portfolio"],
        panels=panels,
        templating=templating,
    )


def liquidity_sourcing_dashboard() -> dict[str, Any]:
    diag_filter = _controller_diagnostics_filter()
    target_filter = _joint_target_book_filter()
    panels: list[dict[str, Any]] = [
        _panel_text(
            1,
            0,
            "**Sources:** `ControllerDiagnostics.frame`, `execution_gates`, `JointTargetBook`.\n\n"
            "Frame sync readiness, liquidity-sourcing mode, per-venue live gates, and joint target books.",
        ),
        _panel_row(2, 4, "Frame sync & mode"),
        _panel_sql_timeseries(
            10,
            5,
            "Frame readiness flags",
            f"""SELECT e.ts AS "time", flag AS metric,
  CASE WHEN (e.payload #>> ARRAY['frame','sync_status', flag])::boolean THEN 1 ELSE 0 END::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES
    ('local_market_ready'),
    ('global_market_ready'),
    ('local_private_ready'),
    ('global_private_ready'),
    ('ledger_fresh')
) AS flags(flag)
WHERE {diag_filter}
ORDER BY 1""",
            w=12,
        ),
        _panel_sql_timeseries(
            11,
            5,
            "Frame age & latency skew (ms)",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> path)::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES
    ('frame_age_ms', ARRAY['frame','frame_age_ms']),
    ('global_lag_ms', ARRAY['frame','global_lag_ms']),
    ('local_private_age_ms', ARRAY['frame','local_private_age_ms']),
    ('global_private_age_ms', ARRAY['frame','global_private_age_ms']),
    ('latency_skew_ms', ARRAY['frame','sync_status','latency_skew_ms'])
) AS ages(metric, path)
WHERE {diag_filter}
  AND e.payload #>> path IS NOT NULL
ORDER BY 1""",
            w=12,
            x=12,
        ),
        _panel_row(3, 13, "Venue execution gates"),
        _panel_sql_timeseries(
            12,
            14,
            "Venue gate live flags",
            f"""SELECT e.ts AS "time", gate.key AS metric,
  CASE WHEN (gate.value->>'live')::boolean THEN 1 ELSE 0 END::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(
  COALESCE(e.payload->'execution_gates'->'venue_gates', '{{}}'::jsonb)
) AS gate(key, value)
WHERE {diag_filter}
ORDER BY 1""",
            w=12,
        ),
        _panel_sql_timeseries(
            13,
            14,
            "Venue gate has_orders flags",
            f"""SELECT e.ts AS "time", gate.key AS metric,
  CASE WHEN (gate.value->>'has_orders')::boolean THEN 1 ELSE 0 END::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(
  COALESCE(e.payload->'execution_gates'->'venue_gates', '{{}}'::jsonb)
) AS gate(key, value)
WHERE {diag_filter}
ORDER BY 1""",
            w=12,
            x=12,
        ),
        _panel_sql_table(
            14,
            22,
            "Venue gate blocked reasons (latest frame)",
            f"""SELECT gate.key AS exchange,
  COALESCE(gate.value->>'role', '') AS role,
  COALESCE(gate.value->>'live', '') AS live,
  COALESCE(gate.value->>'has_orders', '') AS has_orders,
  COALESCE(gate.value->'blocked_reasons', '[]'::jsonb)::text AS blocked_reasons
FROM mm_events e
CROSS JOIN LATERAL jsonb_each(
  COALESCE(e.payload->'execution_gates'->'venue_gates', '{{}}'::jsonb)
) AS gate(key, value)
WHERE {diag_filter}
ORDER BY e.ts DESC
LIMIT 20""",
            h=6,
        ),
        _panel_row(4, 28, "Joint target books"),
        _panel_sql_timeseries(
            15,
            29,
            "Target book order count (local / global)",
            f"""SELECT e.ts AS "time", 'local_orders' AS metric,
  jsonb_array_length(COALESCE(e.payload->'local'->'orders', '[]'::jsonb))::double precision AS value
FROM mm_events e
WHERE {target_filter}
UNION ALL
SELECT e.ts, 'global_orders',
  jsonb_array_length(COALESCE(e.payload->'global'->'orders', '[]'::jsonb))::double precision
FROM mm_events e
WHERE {target_filter}
ORDER BY 1""",
            w=12,
        ),
        _panel_sql_timeseries(
            16,
            29,
            "Liquidity sourcing mode (1 = liquidity_sourcing)",
            f"""SELECT e.ts AS "time", COALESCE(e.payload->>'mode', 'unknown') AS metric,
  CASE WHEN e.payload->>'mode' = 'liquidity_sourcing' THEN 1 ELSE 0 END::double precision AS value
FROM mm_events e
WHERE {target_filter}
ORDER BY 1""",
            w=12,
            x=12,
        ),
        _panel_sql_timeseries(
            17,
            37,
            "Target book buy/sell qty by venue",
            f"""SELECT e.ts AS "time",
  books.venue || '_' || COALESCE(books.ord->>'side', '') AS metric,
  SUM((books.ord->>'qty')::double precision) AS value
FROM mm_events e
CROSS JOIN LATERAL (
  SELECT 'local' AS venue, ord
  FROM jsonb_array_elements(COALESCE(e.payload->'local'->'orders', '[]'::jsonb)) AS ord
  UNION ALL
  SELECT 'global', ord
  FROM jsonb_array_elements(COALESCE(e.payload->'global'->'orders', '[]'::jsonb)) AS ord
) AS books(venue, ord)
WHERE {target_filter}
GROUP BY 1, 2
ORDER BY 1""",
        ),
    ]
    return _dashboard(
        title="MM liquidity sourcing (PostgreSQL)",
        uid="mm-liquidity-sourcing-pg",
        description="Frame sync, venue gates, and joint target books for liquidity sourcing.",
        tags=["market-maker", "postgres", "liquidity-sourcing"],
        panels=panels,
        templating=_controller_diagnostics_templating(),
    )


def ledger_dashboard() -> dict[str, Any]:
    ledger_filter = _ledger_filter()
    panels: list[dict[str, Any]] = [
        _panel_text(
            1,
            0,
            "**Source:** `mm_events` (`JointLedgerSnapshot` on **Ledger topic**).\n\n"
            "Portfolio ledger: balances, locked/available collateral, live order counts, "
            "consolidated inventory, reference price, and freshness ages.",
        ),
        _panel_row(2, 4, "Freshness"),
        _panel_sql_timeseries(
            10,
            5,
            "Private & market age (ms)",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> path)::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES
    ('local_private_age_ms', ARRAY['freshness','local_private_age_ms']),
    ('global_private_age_ms', ARRAY['freshness','global_private_age_ms']),
    ('local_market_age_ms', ARRAY['freshness','local_market_age_ms']),
    ('global_market_age_ms', ARRAY['freshness','global_market_age_ms'])
) AS ages(metric, path)
WHERE {ledger_filter}
  AND e.payload #>> path IS NOT NULL
ORDER BY 1""",
            w=12,
        ),
        _panel_sql_timeseries(
            11,
            5,
            "Ledger fresh flag",
            f"""SELECT e.ts AS "time", 'ledger_fresh' AS metric,
  CASE WHEN (e.payload #>> '{{freshness,ledger_fresh}}')::boolean THEN 1 ELSE 0 END::double precision AS value
FROM mm_events e
WHERE {ledger_filter}
ORDER BY 1""",
            w=12,
            x=12,
        ),
        _panel_row(3, 13, "Local venue ledger"),
        _panel_sql_timeseries(
            12,
            14,
            "Local available / locked collateral",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> path)::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES
    ('available_base', ARRAY['local','available_base']),
    ('available_quote', ARRAY['local','available_quote']),
    ('locked_base', ARRAY['local','locked_base']),
    ('locked_quote', ARRAY['local','locked_quote'])
) AS legs(metric, path)
WHERE {ledger_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            13,
            22,
            "Local live order count",
            f"""SELECT e.ts AS "time", 'local_live_orders' AS metric,
  COALESCE(jsonb_object_length(e.payload->'local'->'live_orders'), 0)::double precision AS value
FROM mm_events e
WHERE {ledger_filter}
ORDER BY 1""",
            w=12,
        ),
        _panel_sql_timeseries(
            14,
            22,
            "Global live order count",
            f"""SELECT e.ts AS "time", 'global_live_orders' AS metric,
  COALESCE(jsonb_object_length(e.payload->'global'->'live_orders'), 0)::double precision AS value
FROM mm_events e
WHERE {ledger_filter}
  AND e.payload->'global' IS NOT NULL
ORDER BY 1""",
            w=12,
            x=12,
        ),
        _panel_row(4, 30, "Global venue ledger"),
        _panel_sql_timeseries(
            15,
            31,
            "Global available / locked collateral",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> path)::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES
    ('available_base', ARRAY['global','available_base']),
    ('available_quote', ARRAY['global','available_quote']),
    ('locked_base', ARRAY['global','locked_base']),
    ('locked_quote', ARRAY['global','locked_quote'])
) AS legs(metric, path)
WHERE {ledger_filter}
  AND e.payload->'global' IS NOT NULL
ORDER BY 1""",
        ),
        _panel_row(5, 39, "Consolidated"),
        _panel_sql_timeseries(
            16,
            40,
            "Consolidated inventory & quote wealth",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> path)::double precision AS value
FROM mm_events e
CROSS JOIN (
  VALUES
    ('consolidated_base_inventory', ARRAY['consolidated_base_inventory']),
    ('consolidated_quote_wealth', ARRAY['consolidated_quote_wealth']),
    ('reference_price', ARRAY['reference_price'])
) AS legs(metric, path)
WHERE {ledger_filter}
ORDER BY 1""",
        ),
    ]
    return _dashboard(
        title="MM ledger (PostgreSQL)",
        uid="mm-ledger-pg",
        description="Joint ledger snapshots: balances, collateral, live orders, and freshness.",
        tags=["market-maker", "postgres", "ledger"],
        panels=panels,
        templating=_ledger_templating(),
    )

_MODEL_SCORE_ROW_TITLE = "Model scores (ControllerDiagnostics)"
_MODEL_SCORE_PANEL_TITLES = frozenset(
    {
        "Model NLL",
        "Rolling mean NLL (5 frames)",
        "Standardized residuals",
        "Score skip reasons",
    }
)


def _is_model_score_panel(panel: dict[str, Any]) -> bool:
    title = panel.get("title")
    if panel.get("type") == "row" and title == _MODEL_SCORE_ROW_TITLE:
        return True
    return title in _MODEL_SCORE_PANEL_TITLES


def _replace_template_var(templating: list[dict[str, Any]], name: str, value: dict[str, Any]) -> None:
    for idx, item in enumerate(templating):
        if item.get("name") == name:
            templating[idx] = value
            return
    templating.append(value)


def patch_model_params_dashboard() -> None:
    path = OUT / "mm_strategy_model_params_pg.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    panels = [p for p in data["panels"] if not _is_model_score_panel(p)]
    y = max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in panels)

    diag_filter = """
  e.topic = '${diagnostics_topic}'
  AND e.event_type = 'ControllerDiagnostics'
  AND e.exchange = '${exchange}'
  AND e.symbol = '${symbol}'
  AND $__timeFilter(e.ts)"""

    panels.append(_panel_row(100, y, _MODEL_SCORE_ROW_TITLE))
    y += 1
    panels.append(
        _panel_sql_timeseries(
            101,
            y,
            "Model NLL",
            f"""SELECT e.ts AS "time", model AS metric,
  (e.payload #>> ARRAY['models', model, 'nll'])::double precision AS value
FROM mm_events e
CROSS JOIN LATERAL unnest(ARRAY['local_price','local_market_flow','global_market_flow','global_price']) AS model
WHERE {diag_filter}
  AND e.payload #>> ARRAY['models', model, 'nll'] IS NOT NULL
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
CROSS JOIN LATERAL unnest(ARRAY['local_price','local_market_flow','global_market_flow','global_price']) AS model
WHERE {diag_filter}
  AND e.payload #>> ARRAY['models', model, 'skip_reason'] IS NOT NULL
  AND e.payload #>> ARRAY['models', model, 'skip_reason'] <> ''
GROUP BY 1, 2
ORDER BY count DESC""",
            h=6,
        )
    )

    templating = data.setdefault("templating", {}).setdefault("list", [])
    _replace_template_var(
        templating,
        "exchange",
        _pg_query_var("exchange", "Exchange", "SELECT DISTINCT exchange FROM mm_events WHERE exchange IS NOT NULL ORDER BY 1"),
    )
    _replace_template_var(
        templating,
        "symbol",
        _pg_query_var("symbol", "Symbol", "SELECT DISTINCT symbol FROM mm_events WHERE symbol IS NOT NULL ORDER BY 1"),
    )
    _replace_template_var(
        templating,
        "diagnostics_topic",
        _pg_query_var(
            "diagnostics_topic",
            "Diagnostics topic",
            "SELECT DISTINCT topic FROM mm_events WHERE topic LIKE 'strategy:diagnostics:%' ORDER BY 1",
        ),
    )
    data["panels"] = panels
    _apply_standard_panel_heights(panels)
    compact_dashboard_panels(data, height_scale=1.0)
    data["version"] = int(data.get("version", 1)) + 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "compact":
        compact_all_dashboards()
        return

    (OUT / "mm_controller_diagnostics_pg.json").write_text(
        json.dumps(
            compact_dashboard_panels(controller_diagnostics_dashboard(), height_scale=1.0),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "mm_local_action_pg.json").write_text(
        json.dumps(
            compact_dashboard_panels(local_action_dashboard(), height_scale=1.0),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "mm_local_action_rejections_pg.json").write_text(
        json.dumps(
            compact_dashboard_panels(local_action_funnel_rejections_dashboard(), height_scale=1.0),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "mm_portfolio_pg.json").write_text(
        json.dumps(compact_dashboard_panels(portfolio_dashboard()), indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "mm_liquidity_sourcing_pg.json").write_text(
        json.dumps(compact_dashboard_panels(liquidity_sourcing_dashboard()), indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "mm_ledger_pg.json").write_text(
        json.dumps(compact_dashboard_panels(ledger_dashboard()), indent=2) + "\n",
        encoding="utf-8",
    )
    patch_model_params_dashboard()
    print("Wrote grafana dashboards to", OUT)


if __name__ == "__main__":
    main()
