"""Read-only Postgres MCP server for the market-maker database."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Protocol

import psycopg
from pydantic import AnyHttpUrl
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.fastmcp import FastMCP

from mcp_servers.postgres.db import DatabaseConfig, PostgresClient
from mcp_servers.postgres.oauth_config import OAuthSettings
from mcp_servers.postgres.oauth_provider import StaticClientOAuthProvider
from mcp_servers.postgres.safety import validate_readonly_sql
from mcp_servers.postgres.schema_reference import SCHEMA_OVERVIEW

LOGGER = logging.getLogger(__name__)


class _ToolHost(Protocol):
    def tool(self, *args: Any, **kwargs: Any) -> Any: ...

    def resource(self, *args: Any, **kwargs: Any) -> Any: ...


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)


def _host() -> str:
    return os.getenv("MCP_HOST", "127.0.0.1")


def _port() -> int:
    return int(os.getenv("MCP_PORT", "8787"))


def _default_transport() -> str:
    return os.getenv("MCP_TRANSPORT", "streamable-http")


def register_postgres_tools(app: _ToolHost, client: PostgresClient) -> None:
    @app.resource("mm://schema-overview")
    def schema_overview() -> str:
        """Market-maker Postgres table reference and example queries."""
        return SCHEMA_OVERVIEW

    @app.tool(
        name="run_sql",
        description=(
            "Run a single read-only SQL statement against the market-maker Postgres database. "
            "Results are capped at 500 rows."
        ),
    )
    def run_sql_tool(query: str) -> str:
        sql = validate_readonly_sql(query)
        try:
            rows = client.run_read_query(sql)
        except psycopg.Error as exc:
            raise RuntimeError(f"PostgreSQL rejected the query: {exc}") from exc
        return json.dumps(rows, default=str, indent=2)

    @app.tool(
        name="list_tables",
        description="List user tables in Postgres with approximate on-disk size.",
    )
    def list_tables_tool() -> str:
        return json.dumps(client.list_tables(), default=str, indent=2)

    @app.tool(
        name="describe_table",
        description="Describe columns for a table in the public schema (or pass schema=).",
    )
    def describe_table_tool(table: str, schema: str = "public") -> str:
        rows = client.describe_table(table, schema=schema)
        if not rows:
            raise ValueError(f"table not found: {schema}.{table}")
        return json.dumps(rows, default=str, indent=2)


def build_fastmcp_server(client: PostgresClient) -> FastMCP:
    server = FastMCP(
        name="market-maker-postgres",
        host=_host(),
        port=_port(),
        streamable_http_path="/mcp",
    )
    register_postgres_tools(server, client)
    return server


def build_oauth_server(client: PostgresClient, oauth_settings: OAuthSettings) -> FastMCP:
    oauth_provider = StaticClientOAuthProvider(oauth_settings)
    auth_settings = AuthSettings(
        issuer_url=AnyHttpUrl(oauth_settings.issuer_url),
        client_registration_options=ClientRegistrationOptions(
            enabled=False,
            valid_scopes=[oauth_settings.scope],
            default_scopes=[oauth_settings.scope],
        ),
        required_scopes=[oauth_settings.scope],
        resource_server_url=None,
    )

    server = FastMCP(
        name="market-maker-postgres",
        instructions="Read-only Postgres MCP for the market-maker runtime",
        host=_host(),
        port=_port(),
        streamable_http_path="/mcp",
        auth_server_provider=oauth_provider,
        auth=auth_settings,
    )
    register_postgres_tools(server, client)

    @server.custom_route("/login", methods=["GET"])
    async def login_page_handler(request: Request) -> Response:
        state = request.query_params.get("state")
        if not state:
            raise HTTPException(400, "Missing state parameter")
        return await oauth_provider.get_login_page(state)

    @server.custom_route("/login/callback", methods=["POST"])
    async def login_callback_handler(request: Request) -> Response:
        return await oauth_provider.handle_login_callback(request)

    return server


class _APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.url.path.startswith("/mcp"):
            auth = request.headers.get("authorization", "")
            expected = f"Bearer {self._api_key}"
            if auth != expected:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def _run_streamable_http(server: FastMCP, *, oauth_enabled: bool) -> None:
    if oauth_enabled:
        LOGGER.info("OAuth issuer: %s", os.getenv("MCP_PUBLIC_URL", "").rstrip("/"))
        LOGGER.info("OAuth client id: %s", os.getenv("MCP_OAUTH_CLIENT_ID", ""))
        server.run(transport="streamable-http")
        return

    api_key = os.getenv("MCP_API_KEY", "").strip()
    if not api_key:
        server.run(transport="streamable-http")
        return

    import uvicorn

    app = server.streamable_http_app()
    app.add_middleware(_APIKeyMiddleware, api_key=api_key)
    uvicorn.run(app, host=_host(), port=_port(), log_level="info")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Market-maker read-only Postgres MCP server (streamable HTTP or stdio)."
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default=_default_transport(),
        help="MCP transport (default: streamable-http or MCP_TRANSPORT).",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=int(os.getenv("PGCONNECT_TIMEOUT", "10")),
        help="Postgres connect timeout in seconds.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        config = DatabaseConfig.from_env()
    except ValueError as exc:
        parser.error(str(exc))

    try:
        oauth_settings = OAuthSettings.from_env()
    except ValueError as exc:
        parser.error(str(exc))

    client = PostgresClient(config, connect_timeout_s=args.connect_timeout)
    oauth_enabled = oauth_settings is not None
    server = (
        build_oauth_server(client, oauth_settings)
        if oauth_settings is not None
        else build_fastmcp_server(client)
    )

    LOGGER.info(
        "Postgres MCP target %s:%s/%s (user=%s, read-only)",
        config.host,
        config.port,
        config.database,
        config.user or "<none>",
    )
    LOGGER.info("Transport: %s", args.transport)
    if args.transport == "streamable-http":
        LOGGER.info("Endpoint: http://%s:%s/mcp", _host(), _port())
        if oauth_enabled:
            LOGGER.info("Auth mode: OAuth (static client id/secret)")
        elif os.getenv("MCP_API_KEY"):
            LOGGER.info("Auth mode: API key")
        else:
            LOGGER.info("Auth mode: none")

    if args.transport == "streamable-http":
        _run_streamable_http(server, oauth_enabled=oauth_enabled)
    else:
        if oauth_enabled:
            parser.error("OAuth mode requires --transport streamable-http")
        server.run(transport="stdio")


if __name__ == "__main__":
    main(sys.argv[1:])
