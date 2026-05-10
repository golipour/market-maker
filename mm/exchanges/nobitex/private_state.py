from __future__ import annotations

import asyncio

from mm.exchanges.nobitex import private_rest
from mm.exchanges.shared.private_state import RestPollingPrivateStateAdapter


class NobitexRestPrivateStateAdapter(RestPollingPrivateStateAdapter):
    def __init__(
        self,
        symbol: str,
        *,
        api_token: str | None,
        poll_interval_ms: int = 5000,
    ):
        super().__init__(
            "nobitex",
            symbol,
            api_token=api_token,
            poll_interval_ms=poll_interval_ms,
        )

    async def fetch_balances(self) -> dict:
        return await asyncio.to_thread(private_rest.get_balances, self.api_token)

    async def fetch_open_orders(self) -> list[dict]:
        return await asyncio.to_thread(private_rest.list_open_orders, self.symbol, self.api_token)
