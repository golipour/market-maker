"""Generate OAuth client credentials for the Postgres MCP server."""

from __future__ import annotations

import secrets


def main() -> None:
    client_id = f"mm-cursor-{secrets.token_hex(4)}"
    client_secret = secrets.token_urlsafe(32)
    login_password = secrets.token_urlsafe(16)
    print("Add these to your repo-root .env:\n")
    print(f"MCP_OAUTH_CLIENT_ID={client_id}")
    print(f"MCP_OAUTH_CLIENT_SECRET={client_secret}")
    print("MCP_OAUTH_USERNAME=admin")
    print(f"MCP_OAUTH_PASSWORD={login_password}")
    print("MCP_PUBLIC_URL=https://your-tunnel-host.example.com")
    print("\nUse the same CLIENT_ID / CLIENT_SECRET in Cursor mcp.json auth block.")


if __name__ == "__main__":
    main()
