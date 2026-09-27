"""Canonical optimizer config key names."""

from __future__ import annotations

from typing import Any

CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX = "cumulative_qty_floor_at_spread_max"
CUMULATIVE_QTY_CAP_AT_SPREAD_MAX = "cumulative_qty_cap_at_spread_max"
RESERVATION_SKEW = "reservation_skew"

SOFT_SCORING = "soft_scoring"
PRESENCE_REFERENCE_QTY = "presence_reference_qty"
TIGHTNESS_SPREAD_SCALE = "tightness_spread_scale"

MAX_ABS_SKEW = "max_abs_skew"

# Performance / observability knobs (Phase 0 of the optimizer performance plan).
# How often full per-epoch objective diagnostics are materialized. 1 keeps the
# historical every-epoch behavior; higher values record only the first, last,
# sampled, and best epochs, which is sufficient for the published diagnostics.
OPTIMIZER_DIAGNOSTICS_EVERY = "optimizer_diagnostics_every"

REMOVED_VENUE_OPTIMIZER_ALIASES: dict[str, str] = {
    "quote_size_min_per_side": CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX,
    "quote_size_max_per_side": CUMULATIVE_QTY_CAP_AT_SPREAD_MAX,
    "reservation_offset": RESERVATION_SKEW,
}

REMOVED_BOOK_QUALITY_ALIASES: dict[str, str] = {
    "soft_target": SOFT_SCORING,
    "depth_target": PRESENCE_REFERENCE_QTY,
    "spread_target": TIGHTNESS_SPREAD_SCALE,
}

REMOVED_RESERVATION_SKEW_ALIASES: dict[str, str] = {
    "r_max": MAX_ABS_SKEW,
}


def reject_removed_keys(
    payload: dict[str, Any],
    *,
    removed: dict[str, str],
    scope: str,
) -> None:
    found = sorted(key for key in removed if key in payload)
    if not found:
        return
    replacements = ", ".join(f"{key}->{removed[key]}" for key in found)
    raise ValueError(f"removed keys under {scope}: {replacements}")


def reject_removed_venue_optimizer_keys(payload: dict[str, Any], *, venue: str) -> None:
    reject_removed_keys(
        payload,
        removed=REMOVED_VENUE_OPTIMIZER_ALIASES,
        scope=f"controller.optimizer.{venue}",
    )


def reject_removed_book_quality_keys(payload: dict[str, Any]) -> None:
    reject_removed_keys(payload, removed=REMOVED_BOOK_QUALITY_ALIASES, scope="book_quality")
    soft = payload.get(SOFT_SCORING)
    if isinstance(soft, dict):
        reject_removed_keys(soft, removed=REMOVED_BOOK_QUALITY_ALIASES, scope=f"book_quality.{SOFT_SCORING}")


def reject_removed_reservation_skew_keys(payload: dict[str, Any]) -> None:
    reject_removed_keys(
        payload,
        removed=REMOVED_RESERVATION_SKEW_ALIASES,
        scope=RESERVATION_SKEW,
    )
