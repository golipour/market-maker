"""Shared Grafana dashboard variable queries.

Variables are ordered by most recent activity so a symbol/exchange change in
runtime config does not leave dashboards stuck on stale alphabetical defaults.
"""

from __future__ import annotations

MM_SCOPE_EVENT_TYPES = (
    "'StrategyDecision', 'ControllerDiagnostics', 'JointTargetBook'"
)

INTERVAL_MS_EXPR = (
    "COALESCE(NULLIF(payload->>'interval_ms', ''), substring(topic from ':([0-9]+)$'))"
)
_INTERVAL_MS_EXPR = INTERVAL_MS_EXPR


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


def symbol_var_query(*, include_market_facts: bool = True) -> str:
    sources = [
        f"""SELECT symbol, ts
  FROM mm_events
  WHERE symbol IS NOT NULL
    AND symbol <> ''
    AND event_type IN ({MM_SCOPE_EVENT_TYPES})""",
        """SELECT payload->>'symbol' AS symbol, ts
  FROM mm_events
  WHERE event_type = 'MarketEvent'
    AND payload->>'symbol' IS NOT NULL
    AND payload->>'symbol' <> ''""",
        """SELECT payload #>> ARRAY['local_market','symbol'] AS symbol, ts
  FROM mm_events
  WHERE event_type = 'JointStrategyFrame'""",
        """SELECT payload #>> ARRAY['global_market','symbol'] AS symbol, ts
  FROM mm_events
  WHERE event_type = 'JointStrategyFrame'""",
        """SELECT symbol, ts
  FROM mm_events
  WHERE event_type = 'VenueLedgerSnapshot'
    AND symbol IS NOT NULL
    AND symbol <> ''""",
        """SELECT payload->>'symbol' AS symbol, ts
  FROM mm_events
  WHERE event_type = 'VenueLedgerSnapshot'
    AND payload->>'symbol' IS NOT NULL
    AND payload->>'symbol' <> ''""",
    ]
    if include_market_facts:
        sources.append(
            """SELECT symbol, ts
  FROM mm_market_facts
  WHERE symbol IS NOT NULL AND symbol <> ''"""
        )
    union = "\n  UNION ALL\n  ".join(sources)
    return f"""SELECT symbol
FROM (
  SELECT symbol, max(ts) AS latest_ts
  FROM (
  {union}
  ) sources
  WHERE symbol IS NOT NULL AND symbol <> ''
  GROUP BY symbol
) ranked
ORDER BY latest_ts DESC"""


def exchange_var_query(*, include_market_facts: bool = True) -> str:
    sources = [
        f"""SELECT exchange, ts
  FROM mm_events
  WHERE exchange IS NOT NULL
    AND exchange <> ''
    AND event_type IN ({MM_SCOPE_EVENT_TYPES})""",
        """SELECT gates.key AS exchange, e.ts
  FROM mm_events e
  CROSS JOIN LATERAL jsonb_each(
    COALESCE(e.payload #> ARRAY['execution_gates','venue_gates'], '{}'::jsonb)
  ) AS gates(key, value)
  WHERE e.event_type = 'ControllerDiagnostics'
    AND gates.key IS NOT NULL
    AND gates.key <> ''""",
        """SELECT payload #>> ARRAY['local_market','exchange'] AS exchange, ts
  FROM mm_events
  WHERE event_type = 'JointStrategyFrame'""",
        """SELECT payload #>> ARRAY['global_market','exchange'] AS exchange, ts
  FROM mm_events
  WHERE event_type = 'JointStrategyFrame'""",
        """SELECT exchange, ts
  FROM mm_events
  WHERE event_type = 'MarketEvent'
    AND exchange IS NOT NULL
    AND exchange <> ''""",
        """SELECT exchange, ts
  FROM mm_events
  WHERE event_type IN ('PrivateStateSnapshot', 'VenueLedgerSnapshot')
    AND exchange IS NOT NULL
    AND exchange <> ''""",
        """SELECT COALESCE(
      exchange,
      payload->>'exchange',
      substring(topic from '^private:state:([^:]+):snapshot$'),
      substring(topic from '^ledger:snapshot:([^:]+)$')
    ) AS exchange,
    ts
  FROM mm_events
  WHERE event_type IN ('PrivateStateSnapshot', 'VenueLedgerSnapshot')""",
    ]
    if include_market_facts:
        sources.append(
            """SELECT exchange, ts
  FROM mm_market_facts
  WHERE exchange IS NOT NULL AND exchange <> ''"""
        )
    union = "\n  UNION ALL\n  ".join(sources)
    return f"""SELECT exchange
FROM (
  SELECT exchange, max(ts) AS latest_ts
  FROM (
  {union}
  ) sources
  WHERE exchange IS NOT NULL AND exchange <> ''
  GROUP BY exchange
) ranked
ORDER BY latest_ts DESC"""


def actor_id_var_query() -> str:
    return f"""SELECT actor_id
FROM (
  SELECT actor_id, max(ts) AS latest_ts
  FROM mm_events
  WHERE actor_id IS NOT NULL
    AND actor_id <> ''
    AND event_type IN ({MM_SCOPE_EVENT_TYPES})
  GROUP BY actor_id
) ranked
ORDER BY latest_ts DESC"""


def interval_ms_var_query() -> str:
    return f"""SELECT interval_ms
FROM (
  SELECT interval_ms, max(latest_ts) AS latest_ts
  FROM (
    SELECT {_INTERVAL_MS_EXPR} AS interval_ms, ts AS latest_ts
    FROM mm_events
    WHERE {_INTERVAL_MS_EXPR} IS NOT NULL
      AND {_INTERVAL_MS_EXPR} <> ''
    UNION ALL
    SELECT interval_ms::text AS interval_ms, ts AS latest_ts
    FROM mm_market_facts
    WHERE interval_ms IS NOT NULL
  ) sources
  GROUP BY interval_ms
) ranked
ORDER BY latest_ts DESC"""


def diagnostics_topic_var_query() -> str:
    return """SELECT topic
FROM (
  SELECT topic, max(ts) AS latest_ts
  FROM mm_events
  WHERE topic LIKE 'strategy:diagnostics:%'
  GROUP BY topic
) ranked
ORDER BY latest_ts DESC"""


def target_books_topic_var_query() -> str:
    return """SELECT topic
FROM (
  SELECT topic, max(ts) AS latest_ts
  FROM mm_events
  WHERE topic LIKE 'strategy:target_books:%'
  GROUP BY topic
) ranked
ORDER BY latest_ts DESC"""


def run_id_var_query() -> str:
    return """SELECT run_id
FROM (
  SELECT run_id, max(ts) AS latest_ts
  FROM mm_events
  WHERE run_id IS NOT NULL AND run_id <> ''
  GROUP BY run_id
) ranked
ORDER BY latest_ts DESC"""


def market_event_exchange_var_query() -> str:
    return """SELECT exchange
FROM (
  SELECT exchange, max(ts) AS latest_ts
  FROM (
    SELECT exchange, ts
    FROM mm_events
    WHERE event_type = 'MarketEvent'
      AND exchange IS NOT NULL
      AND exchange <> ''
    UNION ALL
    SELECT exchange, ts
    FROM mm_market_facts
    WHERE exchange IS NOT NULL AND exchange <> ''
  ) sources
  GROUP BY exchange
) ranked
ORDER BY latest_ts DESC"""
