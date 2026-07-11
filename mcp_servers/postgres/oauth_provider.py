"""OAuth authorization server provider with static Cursor client credentials."""

from __future__ import annotations

import secrets
import time
from typing import Any

from pydantic import AnyHttpUrl, AnyUrl
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from mcp_servers.postgres.oauth_config import DEFAULT_OAUTH_REDIRECT_URIS, OAuthSettings


class StaticClientOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """Minimal OAuth AS with a pre-registered static client (Cursor + Claude)."""

    def __init__(self, settings: OAuthSettings) -> None:
        self.settings = settings
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.auth_codes: dict[str, AuthorizationCode] = {}
        self.tokens: dict[str, AccessToken] = {}
        self.state_mapping: dict[str, dict[str, str | None]] = {}
        self._register_static_client()

    def _register_static_client(self) -> None:
        self.clients[self.settings.client_id] = OAuthClientInformationFull(
            client_id=self.settings.client_id,
            client_secret=self.settings.client_secret,
            redirect_uris=[AnyUrl(uri) for uri in DEFAULT_OAUTH_REDIRECT_URIS],
            grant_types=["authorization_code"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_post",
        )

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        raise HTTPException(403, "Dynamic client registration is disabled; use MCP_OAUTH_CLIENT_ID")

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        state = params.state or secrets.token_hex(16)
        self.state_mapping[state] = {
            "redirect_uri": str(params.redirect_uri),
            "code_challenge": params.code_challenge,
            "redirect_uri_provided_explicitly": str(params.redirect_uri_provided_explicitly),
            "client_id": client.client_id,
            "resource": params.resource,
        }
        return f"{self.settings.login_callback_path}?state={state}&client_id={client.client_id}"

    async def get_login_page(self, state: str) -> HTMLResponse:
        if not state:
            raise HTTPException(400, "Missing state parameter")
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Market Maker Postgres MCP</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 420px; margin: 3rem auto; padding: 0 1rem; }}
    label {{ display: block; margin-top: 1rem; font-weight: 600; }}
    input {{ width: 100%; padding: 0.5rem; margin-top: 0.25rem; box-sizing: border-box; }}
    button {{ margin-top: 1.25rem; padding: 0.6rem 1rem; background: #111827; color: #fff; border: 0; cursor: pointer; }}
  </style>
</head>
<body>
  <h1>Market Maker Postgres MCP</h1>
  <p>Sign in to authorize Cursor.</p>
  <form method="post" action="/login/callback">
    <input type="hidden" name="state" value="{state}" />
    <label>Username<input name="username" autocomplete="username" required /></label>
    <label>Password<input name="password" type="password" autocomplete="current-password" required /></label>
    <button type="submit">Authorize</button>
  </form>
</body>
</html>
"""
        return HTMLResponse(content=html_content)

    async def handle_login_callback(self, request: Request) -> Response:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        state = form.get("state")
        if not isinstance(username, str) or not isinstance(password, str) or not isinstance(state, str):
            raise HTTPException(400, "Invalid login form")
        redirect_uri = await self._complete_login(username, password, state)
        return RedirectResponse(url=redirect_uri, status_code=302)

    async def _complete_login(self, username: str, password: str, state: str) -> str:
        state_data = self.state_mapping.get(state)
        if not state_data:
            raise HTTPException(400, "Invalid state parameter")

        redirect_uri = state_data["redirect_uri"]
        code_challenge = state_data["code_challenge"]
        redirect_uri_provided_explicitly = state_data["redirect_uri_provided_explicitly"] == "True"
        client_id = state_data["client_id"]
        resource = state_data.get("resource")

        if redirect_uri is None or code_challenge is None or client_id is None:
            raise HTTPException(400, "Incomplete OAuth state")

        if username != self.settings.username or password != self.settings.password:
            raise HTTPException(401, "Invalid credentials")

        new_code = f"mm_{secrets.token_hex(16)}"
        auth_code = AuthorizationCode(
            code=new_code,
            client_id=client_id,
            redirect_uri=AnyHttpUrl(redirect_uri),
            redirect_uri_provided_explicitly=redirect_uri_provided_explicitly,
            expires_at=time.time() + 300,
            scopes=[self.settings.scope],
            code_challenge=code_challenge,
            resource=resource,
            subject=username,
        )
        self.auth_codes[new_code] = auth_code
        del self.state_mapping[state]
        return construct_redirect_uri(redirect_uri, code=new_code, state=state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self.auth_codes.get(authorization_code)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        if authorization_code.code not in self.auth_codes:
            raise ValueError("Invalid authorization code")
        if not client.client_id:
            raise ValueError("No client_id provided")

        access_token = f"mm_{secrets.token_hex(32)}"
        self.tokens[access_token] = AccessToken(
            token=access_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time()) + 3600,
            resource=authorization_code.resource,
            subject=authorization_code.subject,
        )
        del self.auth_codes[authorization_code.code]
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=3600,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        access_token = self.tokens.get(token)
        if not access_token:
            return None
        if access_token.expires_at and access_token.expires_at < time.time():
            del self.tokens[token]
            return None
        return access_token

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        return None

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        raise NotImplementedError("Refresh tokens are not supported")

    async def revoke_token(self, token: str, token_type_hint: str | None = None) -> None:
        self.tokens.pop(token, None)
