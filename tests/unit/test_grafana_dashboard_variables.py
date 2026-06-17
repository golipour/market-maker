from mm.tools.grafana_dashboard_variables import (
    exchange_var_query,
    interval_ms_var_query,
    symbol_var_query,
)


def test_symbol_var_query_includes_market_events_and_orders_by_recency() -> None:
    sql = symbol_var_query()

    assert "event_type = 'MarketEvent'" in sql
    assert "mm_market_facts" in sql
    assert "global_market','symbol" in sql
    assert "ORDER BY latest_ts DESC" in sql
    assert "private:state" not in sql


def test_exchange_var_query_includes_market_events_and_orders_by_recency() -> None:
    sql = exchange_var_query()

    assert "event_type = 'MarketEvent'" in sql
    assert "mm_market_facts" in sql
    assert "execution_gates','venue_gates" in sql
    assert "ORDER BY latest_ts DESC" in sql


def test_interval_ms_var_query_orders_by_recency() -> None:
    sql = interval_ms_var_query()

    assert "interval_ms" in sql
    assert "mm_market_facts" in sql
    assert "ORDER BY latest_ts DESC" in sql
