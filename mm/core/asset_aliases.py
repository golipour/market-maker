"""Canonical asset codes shared across private state, balances, and strategy."""

from __future__ import annotations

# Venue-specific tickers mapped to one domain code (strategy / snapshots / DB).
# ``RLS`` / ``TMN`` → ``IRT`` is a *naming* alias only. Nobitex reports ``RLS`` / ``IRT``
# wallet and price notionals in **Rial**; convert amounts at the Nobitex adapter
# (``mm.exchanges.nobitex``), not here, so Wallex TMN and other venues stay unchanged.
CANONICAL_ASSET_ALIASES: dict[str, str] = {
    "RLS": "IRT",
    "TMN": "IRT",
}


def canonical_asset_code(raw: object) -> str:
    """Uppercase asset with exchange-specific aliases folded into canonical names."""
    asset = str(raw or "").upper().strip()
    if not asset:
        return ""
    return CANONICAL_ASSET_ALIASES.get(asset, asset)
