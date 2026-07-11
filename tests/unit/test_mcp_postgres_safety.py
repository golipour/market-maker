from __future__ import annotations

import pytest

from mcp_servers.postgres.safety import validate_readonly_sql


def test_validate_readonly_sql_allows_select() -> None:
    assert validate_readonly_sql("SELECT 1") == "SELECT 1"


def test_validate_readonly_sql_blocks_insert() -> None:
    with pytest.raises(ValueError, match="read-only"):
        validate_readonly_sql("INSERT INTO mm_events DEFAULT VALUES")


def test_validate_readonly_sql_blocks_multi_statement() -> None:
    with pytest.raises(ValueError, match="single"):
        validate_readonly_sql("SELECT 1; SELECT 2")


def test_validate_readonly_sql_blocks_delete_in_subquery_name() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_readonly_sql("SELECT * FROM delete_log")
