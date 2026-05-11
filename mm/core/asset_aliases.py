"""Canonical asset codes shared across private state, balances, and strategy."""

from __future__ import annotations

# Venue-specific tickers mapped to one domain code (strategy / snapshots / DB).
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
