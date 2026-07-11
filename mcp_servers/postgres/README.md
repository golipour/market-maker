# Postgres MCP server

Read-only [Model Context Protocol](https://modelcontextprotocol.io/) server for the market-maker Postgres database. Exposes SQL inspection tools over **streamable HTTP** for Cursor and can be exposed through a **Cloudflare Tunnel**.

## Tools

| Tool | Description |
|------|-------------|
| `run_sql` | Read-only SQL (`SELECT`, `WITH`, `EXPLAIN`, `SHOW`, …), max 500 rows |
| `list_tables` | User tables with approximate size |
| `describe_table` | Column metadata for a table |

## Resource

- `mm://schema-overview` — table reference and starter queries for this repo

## Authentication modes

| Mode | Env vars | Cursor config |
|------|----------|---------------|
| **OAuth** (recommended for tunnel) | `MCP_PUBLIC_URL`, `MCP_OAUTH_CLIENT_ID`, `MCP_OAUTH_CLIENT_SECRET`, `MCP_OAUTH_USERNAME`, `MCP_OAUTH_PASSWORD` | `auth.CLIENT_ID` + `auth.CLIENT_SECRET` |
| **API key** | `MCP_API_KEY` | `headers.Authorization: Bearer …` |
| **None** | (unset) | local dev only |

OAuth uses a browser login page on the MCP server. Cursor redirect URIs are pre-registered:

- `cursor://anysphere.cursor-mcp/oauth/callback` (desktop)
- `https://www.cursor.com/agents/mcp/oauth/callback` (cloud agents)

## Quick start (OAuth + local)

Requires **Python 3.10+**.

```bash
# From repo root
pip install -r mcp_servers/postgres/requirements.txt

# Generate credentials
PYTHONPATH=. python -m mcp_servers.postgres.generate_oauth_credentials
# Add printed values to .env, set MCP_PUBLIC_URL=http://127.0.0.1:8787 for local testing

export PYTHONPATH=.
python -m mcp_servers.postgres
```

Endpoint: `http://127.0.0.1:8787/mcp`

### Cursor config (OAuth)

Copy `cursor.mcp.json.example` into Cursor MCP settings:

```json
{
  "mcpServers": {
    "market-maker-postgres": {
      "url": "https://mcp.your-tunnel-host.example.com/mcp",
      "auth": {
        "CLIENT_ID": "${env:MCP_OAUTH_CLIENT_ID}",
        "CLIENT_SECRET": "${env:MCP_OAUTH_CLIENT_SECRET}",
        "scopes": ["mcp:postgres"]
      }
    }
  }
}
```

Export the same `MCP_OAUTH_CLIENT_ID` and `MCP_OAUTH_CLIENT_SECRET` in your shell (or use literal values). Click **Connect** in Cursor → browser login → enter `MCP_OAUTH_USERNAME` / `MCP_OAUTH_PASSWORD`.

## Docker Compose

```bash
# Set OAuth vars in .env first (see .env.example)
docker compose up -d db mcp-postgres
```

## Cloudflare Tunnel

Expose the **MCP HTTP server**, not Postgres directly.

1. Start MCP with OAuth env vars; set `MCP_PUBLIC_URL` to your public hostname (e.g. `https://mcp.example.com`).
2. Point the tunnel at `http://127.0.0.1:8787` (or the `mcp-postgres` container).
3. Use the public `/mcp` URL in Cursor with the same `CLIENT_ID` / `CLIENT_SECRET`.

**Quick tunnel (testing):**

```bash
# Terminal 1
docker compose up -d mcp-postgres

# Terminal 2 — after tunnel starts, set MCP_PUBLIC_URL to the printed https URL and restart mcp-postgres
cloudflared tunnel --url http://127.0.0.1:8787
```

**Named tunnel:** see `cloudflared.config.example.yml`.

## Environment

| Variable | Required for OAuth | Purpose |
|----------|-------------------|---------|
| `MM_DATABASE_URL` | yes | Postgres URL |
| `MCP_PUBLIC_URL` | yes | Public base URL (tunnel hostname, no trailing slash) |
| `MCP_OAUTH_CLIENT_ID` | yes | Static OAuth client id (shared with Cursor) |
| `MCP_OAUTH_CLIENT_SECRET` | yes | Static OAuth client secret (shared with Cursor) |
| `MCP_OAUTH_USERNAME` | yes | Login page username |
| `MCP_OAUTH_PASSWORD` | yes | Login page password |
| `MCP_OAUTH_SCOPE` | no | Default `mcp:postgres` |
| `MCP_HOST` | no | Bind address (`0.0.0.0` in Docker) |
| `MCP_PORT` | no | Default `8787` |
| `MCP_API_KEY` | no | Bearer token auth (alternative to OAuth) |

See `.env.example` in this directory.
