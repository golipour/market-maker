from mm.tools.grafana_dashboard_variables import (
    actor_id_var_query,
    exchange_var_query,
    interval_ms_facts_filter_sql,
    interval_ms_var_query,
    run_id_var_query,
    symbol_var_query,
)


def test_symbol_var_query_uses_registry_not_event_scans() -> None:
    sql = symbol_var_query()

    assert "registry_market_sources" in sql
    assert "ORDER BY latest_ts DESC" in sql
    assert "FROM mm_events" not in sql
    assert "FROM mm_market_facts" not in sql
    assert "private:state" not in sql


def test_exchange_var_query_uses_registry_not_event_scans() -> None:
    sql = exchange_var_query()

    assert "registry_exchanges" in sql
    assert "ORDER BY latest_ts DESC" in sql
    assert "FROM mm_events" not in sql
    assert "FROM mm_market_facts" not in sql
    assert "execution_gates" not in sql


def test_interval_ms_var_query_uses_registry_feeds() -> None:
    sql = interval_ms_var_query()

    assert "registry_market_feeds" in sql
    assert "ORDER BY latest_ts DESC" in sql
    assert "FROM mm_events" not in sql
    assert "FROM mm_market_facts" not in sql


def test_actor_and_run_queries_avoid_full_event_scans() -> None:
    actor_sql = actor_id_var_query()
    run_sql = run_id_var_query()

    assert "registry_strategies" in actor_sql
    assert "FROM mm_events" not in actor_sql
    assert "mm_market_facts" in run_sql
    assert "INTERVAL '7 days'" in run_sql
    assert "FROM mm_events" not in run_sql


def test_diagnostics_and_target_book_topic_vars_use_registry() -> None:
    from mm.tools.grafana_dashboard_variables import (
        diagnostics_topic_var_query,
        target_books_topic_var_query,
    )

    diagnostics_sql = diagnostics_topic_var_query()
    target_sql = target_books_topic_var_query()

    assert "registry_strategies" in diagnostics_sql
    assert "strategy:diagnostics:" in diagnostics_sql
    assert "FROM mm_events" not in diagnostics_sql
    assert "registry_strategies" in target_sql
    assert "strategy:target_books:" in target_sql
    assert "FROM mm_events" not in target_sql


def test_interval_ms_filter_accepts_grafana_all_join() -> None:
    sql = interval_ms_facts_filter_sql()

    assert "'${interval_ms}' IN ('$__all', 'All')" in sql
    assert "string_to_array(replace('${interval_ms}', ' ', ''), ',')" in sql
