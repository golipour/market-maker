"""OAuth settings for the Postgres MCP server."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

CURSOR_DESKTOP_REDIRECT_URI = "cursor://anysphere.cursor-mcp/oauth/callback"
CURSOR_CLOUD_REDIRECT_URI = "https://www.cursor.com/agents/mcp/oauth/callback"
CLAUDE_WEB_REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"
CLAUDE_WEB_ALT_REDIRECT_URI = "https://claude.com/api/mcp/auth_callback"

DEFAULT_OAUTH_REDIRECT_URIS = (
    CURSOR_DESKTOP_REDIRECT_URI,
    CURSOR_CLOUD_REDIRECT_URI,
    CLAUDE_WEB_REDIRECT_URI,
    CLAUDE_WEB_ALT_REDIRECT_URI,
)


def _default_public_url() -> str:
    host = os.getenv("MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = os.getenv("MCP_PORT", "8787").strip() or "8787"
    return f"http://{host}:{port}"


@dataclass(frozen=True, slots=True)
class OAuthSettings:
    public_url: str
    client_id: str
    client_secret: str
    username: str
    password: str
    scope: str = "mcp:postgres"

    @classmethod
    def from_env(cls) -> OAuthSettings | None:
        client_id = os.getenv("MCP_OAUTH_CLIENT_ID", "").strip()
        client_secret = os.getenv("MCP_OAUTH_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            return None

        public_url = (
            os.getenv("MCP_PUBLIC_URL", "").strip()
            or os.getenv("MCP_OAUTH_ISSUER_URL", "").strip()
        )
        if not public_url:
            public_url = _default_public_url()
            LOGGER.warning(
                "MCP_PUBLIC_URL not set; using %s for OAuth discovery (fine for local Cursor; "
                "set MCP_PUBLIC_URL to your Cloudflare tunnel URL before exposing publicly)",
                public_url,
            )

        username = os.getenv("MCP_OAUTH_USERNAME", "").strip()
        password = os.getenv("MCP_OAUTH_PASSWORD", "").strip()
        if not username or not password:
            raise ValueError(
                "MCP_OAUTH_USERNAME and MCP_OAUTH_PASSWORD are required for OAuth login"
            )

        return cls(
            public_url=public_url.rstrip("/"),
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            scope=os.getenv("MCP_OAUTH_SCOPE", "mcp:postgres").strip() or "mcp:postgres",
        )

    @property
    def issuer_url(self) -> str:
        return self.public_url

    @property
    def login_callback_path(self) -> str:
        return f"{self.public_url}/login"

    @property
    def resource_server_url(self) -> str:
        return f"{self.public_url}/mcp"
