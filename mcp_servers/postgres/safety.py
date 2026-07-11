"""Read-only SQL guards for the Postgres MCP server."""

from __future__ import annotations

import re

_FORBIDDEN = re.compile(
    r"\b("
    r"insert|update|delete|drop|alter|truncate|create|grant|revoke|"
    r"copy|call|do|merge|replace|vacuum|reindex|cluster|"
    r"refresh|security|comment|attach|detach|discard|"
    r"lock|notify|listen|unlisten|reset|set\s+role|set\s+session\s+authorization"
    r")\b",
    re.IGNORECASE,
)

_ALLOWED_START = re.compile(
    r"^\s*(with|select|show|explain|describe|table|values)\b",
    re.IGNORECASE,
)


def validate_readonly_sql(sql: str) -> str:
    """Return normalized SQL or raise ValueError when the statement is not read-only."""

    text = sql.strip()
    if not text:
        raise ValueError("SQL must not be empty")
    if ";" in text.rstrip(";"):
        raise ValueError("Only a single SQL statement is allowed")
    text = text.rstrip(";").strip()
    if not _ALLOWED_START.match(text):
        raise ValueError(
            "Only read-only statements are allowed (SELECT, WITH, EXPLAIN, SHOW, DESCRIBE, TABLE, VALUES)"
        )
    if _FORBIDDEN.search(text):
        raise ValueError("Statement contains a forbidden keyword for read-only mode")
    return text
