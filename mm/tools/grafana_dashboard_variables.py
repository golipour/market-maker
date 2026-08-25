"""Shared Grafana dashboard variable queries.

Variables are ordered by most recent activity so a symbol/exchange change in
runtime config does not leave dashboards stuck on stale alphabetical defaults.
"""

from __future__ import annotations

MM_SCOPE_EVENT_TYPES = (
    "'StrategyDecision', 'ControllerDiagnostics', 'JointTargetBook'"
)

# Full-table mm_events scans time out Grafana variable queries (dropdowns collapse to All).
VARIABLE_LOOKBACK = "NOW() - INTERVAL '7 days'"

INTERVAL_MS_EXPR = (
    "COALESCE(NULLIF(payload->>'interval_ms', ''), substring(topic from ':([0-9]+)$'))"
)
_INTERVAL_MS_EXPR = INTERVAL_MS_EXPR


def _ranked_values_sql(column: str, sources: list[str]) -> str:
    union = "\n  UNION ALL\n  ".join(sources)
    return f"""SELECT {column}
FROM (
  SELECT {column}, max(latest_ts) AS latest_ts
  FROM (
  {union}
  ) sources
  WHERE {column} IS NOT NULL AND CAST({column} AS text) <> ''
  GROUP BY {column}
) ranked
ORDER BY latest_ts DESC"""


def interval_ms_events_filter_sql(*, table_ref: str = "") -> str:
    """Match Grafana ``interval_ms`` variable against MarketEvent rows (optional filter)."""

    prefix = f"{table_ref}." if table_ref else ""
    expr = (
        f"COALESCE(NULLIF({prefix}payload->>'interval_ms', ''), "
        f"substring({prefix}topic from ':([0-9]+)$'))"
    )
    return (
        f"(NULLIF('${{interval_ms}}', '') IS NULL OR "
        f"{expr} = NULLIF('${{interval_ms}}', ''))"
    )


def interval_ms_facts_filter_sql() -> str:
    """Match Grafana ``interval_ms`` variable against ``mm_market_facts`` rows."""

    return (
        "(NULLIF('${interval_ms}', '') IS NULL OR "
        "interval_ms::text = NULLIF('${interval_ms}', ''))"
    )


def symbol_var_query(*, include_market_facts: bool = False) -> str:
    sources = [
        """SELECT symbol, updated_at AS latest_ts
  FROM registry_market_sources
  WHERE status = 'active'
    AND symbol IS NOT NULL
    AND symbol <> ''""",
    ]
    if include_market_facts:
        sources.append(
            f"""SELECT symbol, max(ts) AS latest_ts
  FROM mm_market_facts
  WHERE symbol IS NOT NULL AND symbol <> ''
    AND ts > {VARIABLE_LOOKBACK}
  GROUP BY symbol"""
        )
    return _ranked_values_sql("symbol", sources)


def exchange_var_query(*, include_market_facts: bool = False) -> str:
    sources = [
        """SELECT exchange, updated_at AS latest_ts
  FROM registry_exchanges
  WHERE status = 'active'
    AND exchange IS NOT NULL
    AND exchange <> ''""",
    ]
    if include_market_facts:
        sources.append(
            f"""SELECT exchange, max(ts) AS latest_ts
  FROM mm_market_facts
  WHERE exchange IS NOT NULL AND exchange <> ''
    AND ts > {VARIABLE_LOOKBACK}
  GROUP BY exchange"""
        )
    return _ranked_values_sql("exchange", sources)


def actor_id_var_query() -> str:
    return _ranked_values_sql(
        "actor_id",
        [
            """SELECT actor_id, updated_at AS latest_ts
  FROM registry_strategies
  WHERE status = 'active'
    AND actor_id IS NOT NULL
    AND actor_id <> ''""",
        ],
    )


def interval_ms_var_query() -> str:
    return _ranked_values_sql(
        "interval_ms",
        [
            """SELECT interval_ms::text AS interval_ms, updated_at AS latest_ts
  FROM registry_market_feeds
  WHERE status = 'active'
    AND interval_ms IS NOT NULL""",
        ],
    )


def diagnostics_topic_var_query() -> str:
    return f"""SELECT topic
FROM (
  SELECT topic, max(ts) AS latest_ts
  FROM mm_events
  WHERE topic LIKE 'strategy:diagnostics:%'
    AND ts > {VARIABLE_LOOKBACK}
  GROUP BY topic
) ranked
ORDER BY latest_ts DESC"""


def target_books_topic_var_query() -> str:
    return f"""SELECT topic
FROM (
  SELECT topic, max(ts) AS latest_ts
  FROM mm_events
  WHERE topic LIKE 'strategy:target_books:%'
    AND ts > {VARIABLE_LOOKBACK}
  GROUP BY topic
) ranked
ORDER BY latest_ts DESC"""


def diagnostics_scope_filter_sql(*, table_ref: str = "e", include_time: bool = True) -> str:
    """Scope ControllerDiagnostics rows by exchange/symbol/actor (not a single topic)."""

    prefix = f"{table_ref}." if table_ref else ""
    clauses = [
        f"{prefix}topic LIKE 'strategy:diagnostics:%'",
        f"{prefix}event_type = 'ControllerDiagnostics'",
        f"{prefix}exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])",
        f"{prefix}symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])",
        f"{prefix}actor_id = ANY(ARRAY[${{actor_id:sqlstring}}]::text[])",
    ]
    if include_time:
        clauses.append(f"$__timeFilter({prefix}ts)")
    return "\n  AND ".join(clauses)


def target_books_scope_filter_sql(*, table_ref: str = "e", include_time: bool = True) -> str:
    """Scope JointTargetBook rows by exchange/symbol/actor (not a single topic)."""

    prefix = f"{table_ref}." if table_ref else ""
    clauses = [
        f"{prefix}topic LIKE 'strategy:target_books:%'",
        f"{prefix}event_type = 'JointTargetBook'",
        f"{prefix}exchange = ANY(ARRAY[${{exchange:sqlstring}}]::text[])",
        f"{prefix}symbol = ANY(ARRAY[${{symbol:sqlstring}}]::text[])",
        (
            f"COALESCE(NULLIF({prefix}actor_id, ''), "
            f"substring({prefix}topic from '^strategy:target_books:(.+)$')) "
            f"= ANY(ARRAY[${{actor_id:sqlstring}}]::text[])"
        ),
    ]
    if include_time:
        clauses.append(f"$__timeFilter({prefix}ts)")
    return "\n  AND ".join(clauses)


def run_id_var_query() -> str:
    return _ranked_values_sql(
        "run_id",
        [
            f"""SELECT run_id, max(ts) AS latest_ts
  FROM mm_market_facts
  WHERE run_id IS NOT NULL AND run_id <> ''
    AND ts > {VARIABLE_LOOKBACK}
  GROUP BY run_id""",
        ],
    )


def market_event_exchange_var_query() -> str:
    return exchange_var_query()
