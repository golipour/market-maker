#!/usr/bin/env python3
"""Generate Grafana dashboards for controller diagnostics, primary action, portfolio, and hedge execution."""

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

_ACTION_PIPELINE_STAGES: tuple[tuple[str, str], ...] = (
    ("ladder", "ladder"),
    ("venue_quantized", "venue_quantized"),
    ("capped", "capped"),
    ("target_book", "target_book"),
    ("risk_approved_target", "risk_approved_target"),
    ("planned_cancel", "planned_cancel"),
    ("planned_place", "planned_place"),
    ("rules_valid", "rules_valid"),
    ("router_accepted", "router_accepted"),
    ("kept", "kept"),
    # Deprecated aliases retained for existing dashboards.
    ("risk_approved", "risk_approved"),
    ("target", "target"),
    ("canceled", "canceled"),
    ("placed", "placed"),
    ("venue_accepted", "venue_accepted"),
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


def _controller_diagnostics_templating() -> list[dict[str, Any]]:
    return [
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
        _panel_sql_timeseries(
            13,
            32,
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
            40,
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


def _panel_action_pipeline_stage(
    id_: int,
    y: int,
    *,
    stage: str,
    title: str,
    diag_filter: str,
    h: int = 6,
) -> dict[str, Any]:
    return _panel_sql_timeseries(
        id_,
        y,
        title,
        f"""SELECT e.ts AS "time",
  (e.payload #> ARRAY['action_pipeline','{stage}','order_count'])::int::double precision AS value
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        h=h,
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


def primary_action_dashboard() -> dict[str, Any]:
    diag_filter = _controller_diagnostics_filter()

    panels: list[dict[str, Any]] = [
        _panel_text(
            1,
            0,
            "**Source:** `mm_events` (`ControllerDiagnostics.action_pipeline` on **Diagnostics topic**).\n\n"
            "One chart per execution funnel stage, top to bottom in pipeline order.",
        ),
    ]

    y = 4
    for index, (stage, title) in enumerate(_ACTION_PIPELINE_STAGES):
        panels.append(
            _panel_action_pipeline_stage(
                10 + index,
                y,
                stage=stage,
                title=f"Order count — {title}",
                diag_filter=diag_filter,
            )
        )
        y += 6

    return _dashboard(
        title="MM primary action (PostgreSQL)",
        uid="mm-primary-action-pg",
        description="Primary action pipeline funnel from ControllerDiagnostics in mm_events.",
        tags=["market-maker", "postgres", "controller", "action"],
        panels=panels,
        templating=_controller_diagnostics_templating(),
    )


def primary_action_funnel_rejections_dashboard() -> dict[str, Any]:
    diag_filter = _controller_diagnostics_filter()

    panels: list[dict[str, Any]] = [
        _panel_text(
            1,
            0,
            "**Source:** `mm_events` (`ControllerDiagnostics.action_pipeline.*.rejection_counts`).\n\n"
            "One chart per funnel stage; each line is a rejection reason count at that stage. "
            "Risk uses `action_pipeline.risk_approved` with fallback to top-level `risk.rejection_counts`.",
        ),
    ]

    y = 4
    for index, (stage, title) in enumerate(_ACTION_PIPELINE_STAGES):
        rejection_expr: str | None = None
        if stage == "risk_approved":
            rejection_expr = (
                "COALESCE("
                "e.payload #> ARRAY['action_pipeline','risk_approved','rejection_counts'], "
                "e.payload->'risk'->'rejection_counts'"
                ")"
            )
        panels.append(
            _panel_action_pipeline_rejections(
                20 + index,
                y,
                stage=stage,
                title=f"Rejections — {title}",
                diag_filter=diag_filter,
                rejection_counts_json=rejection_expr,
            )
        )
        y += 6

    return _dashboard(
        title="MM primary action funnel rejections (PostgreSQL)",
        uid="mm-primary-action-rejections-pg",
        description="Per-stage rejection reason counts from ControllerDiagnostics action_pipeline.",
        tags=["market-maker", "postgres", "controller", "action", "rejections"],
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
            "**Source:** `ControllerDiagnostics.state` and `risk.limits`.\n\n"
            "Inventory/cash and wallet legs use **effective totals** (exchange total, wallet locked, "
            "and open-order collateral when locked under-reports). Limits from configured hard risk.",
        ),
        _panel_sql_timeseries(
            10,
            4,
            "Inventory (primary vs hedge)",
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
  COALESCE((e.payload #>> '{{state,primary_wealth}}')::double precision,
    (e.payload #>> '{{state,primary_cash}}')::double precision
     + (e.payload #>> '{{state,local_inventory}}')::double precision
       * COALESCE((e.payload #>> '{{state,local_sell_vwap}}')::double precision, 0)) AS value
FROM mm_events e
WHERE {diag_filter}
UNION ALL
SELECT e.ts, 'hedge_marked',
  COALESCE((e.payload #>> '{{state,hedge_wealth}}')::double precision,
    (e.payload #>> '{{state,hedge_cash}}')::double precision
     + (e.payload #>> '{{state,global_inventory}}')::double precision
       * COALESCE((e.payload #>> '{{state,global_sell_vwap}}')::double precision, 0)) AS value
FROM mm_events e
WHERE {diag_filter}
ORDER BY 1""",
        ),
        _panel_sql_timeseries(
            15,
            28,
            "Primary quote wallet (free / locked / total / effective)",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> ARRAY['state','primary_quote_wallet', leg])::double precision AS value
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
            "Primary base wallet (free / locked / total / effective)",
            f"""SELECT e.ts AS "time", metric,
  (e.payload #>> ARRAY['state','primary_base_wallet', leg])::double precision AS value
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
            52,
            "Combined inventory exposure",
            f"""SELECT e.ts AS "time", 'primary_plus_hedge' AS metric,
  ((e.payload #>> '{{state,local_inventory}}')::double precision
   + (e.payload #>> '{{state,global_inventory}}')::double precision) AS value
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


def hedge_exec_dashboard() -> dict[str, Any]:
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
            "Live fill-driven hedging (separate from MPC `hedge_rollout`).",
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
            h=4,
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
            h=5,
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
CROSS JOIN LATERAL unnest(ARRAY['price','market_flow','global_price']) AS model
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
CROSS JOIN LATERAL unnest(ARRAY['price','market_flow','global_price']) AS model
WHERE {diag_filter}
  AND e.payload #>> ARRAY['models', model, 'skip_reason'] IS NOT NULL
  AND e.payload #>> ARRAY['models', model, 'skip_reason'] <> ''
GROUP BY 1, 2
ORDER BY count DESC""",
            h=6,
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
    (OUT / "mm_primary_action_pg.json").write_text(
        json.dumps(
            compact_dashboard_panels(primary_action_dashboard(), height_scale=1.0),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "mm_primary_action_rejections_pg.json").write_text(
        json.dumps(
            compact_dashboard_panels(primary_action_funnel_rejections_dashboard(), height_scale=1.0),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "mm_portfolio_pg.json").write_text(
        json.dumps(compact_dashboard_panels(portfolio_dashboard()), indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "mm_hedge_exec_pg.json").write_text(
        json.dumps(compact_dashboard_panels(hedge_exec_dashboard()), indent=2) + "\n",
        encoding="utf-8",
    )
    patch_model_params_dashboard()
    print("Wrote grafana dashboards to", OUT)


if __name__ == "__main__":
    main()
