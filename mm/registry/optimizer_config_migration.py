"""One-shot Postgres migrations for registry JSON config key renames."""

from __future__ import annotations

import json
from typing import Any

from mm.mpc.optimizer_config_keys import (
    CUMULATIVE_QTY_CAP_AT_SPREAD_MAX,
    CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX,
    MAX_ABS_SKEW,
    PRESENCE_REFERENCE_QTY,
    RESERVATION_SKEW,
    SOFT_SCORING,
    TIGHTNESS_SPREAD_SCALE,
)

_LEGACY_VENUE_KEY_MAP = {
    "quote_size_min_per_side": CUMULATIVE_QTY_FLOOR_AT_SPREAD_MAX,
    "quote_size_max_per_side": CUMULATIVE_QTY_CAP_AT_SPREAD_MAX,
    "reservation_offset": RESERVATION_SKEW,
}

_LEGACY_SOFT_KEY_MAP = {
    "depth_target": PRESENCE_REFERENCE_QTY,
    "spread_target": TIGHTNESS_SPREAD_SCALE,
}

_LEGACY_SKEW_KEY_MAP = {
    "r_max": MAX_ABS_SKEW,
}


def _rename_keys(payload: dict[str, Any], key_map: dict[str, str]) -> bool:
    changed = False
    for legacy_key, canonical_key in key_map.items():
        if legacy_key not in payload:
            continue
        if canonical_key in payload:
            del payload[legacy_key]
        else:
            payload[canonical_key] = payload.pop(legacy_key)
        changed = True
    return changed


def _migrate_reservation_skew(payload: Any) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(payload, dict):
        return payload, False
    out = dict(payload)
    return out, _rename_keys(out, _LEGACY_SKEW_KEY_MAP)


def _migrate_soft_scoring(payload: Any) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(payload, dict):
        return payload, False
    out = dict(payload)
    return out, _rename_keys(out, _LEGACY_SOFT_KEY_MAP)


def _migrate_book_quality(payload: Any) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(payload, dict):
        return payload, False
    out = dict(payload)
    changed = False
    if "soft_target" in out:
        soft = out.pop("soft_target")
        if SOFT_SCORING not in out:
            out[SOFT_SCORING] = soft
        changed = True
    if SOFT_SCORING in out and isinstance(out[SOFT_SCORING], dict):
        migrated, soft_changed = _migrate_soft_scoring(out[SOFT_SCORING])
        out[SOFT_SCORING] = migrated
        changed = changed or soft_changed
    return out, changed


def _migrate_venue_optimizer(payload: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(payload, dict):
        return {}, False
    out = dict(payload)
    changed = _rename_keys(out, _LEGACY_VENUE_KEY_MAP)
    if RESERVATION_SKEW in out:
        migrated, skew_changed = _migrate_reservation_skew(out[RESERVATION_SKEW])
        if migrated is not None:
            out[RESERVATION_SKEW] = migrated
        changed = changed or skew_changed
    if "book_quality" in out:
        migrated, bq_changed = _migrate_book_quality(out["book_quality"])
        if migrated is not None:
            out["book_quality"] = migrated
        changed = changed or bq_changed
    return out, changed


def _migrate_optimizer(payload: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(payload, dict):
        return {}, False
    out = dict(payload)
    changed = False
    for venue in ("local", "global"):
        if venue in out and isinstance(out[venue], dict):
            migrated, venue_changed = _migrate_venue_optimizer(out[venue])
            out[venue] = migrated
            changed = changed or venue_changed
    return out, changed


def migrate_runtime_config_mapping(payload: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(payload, dict):
        return {}, False
    out = dict(payload)
    changed = False
    controller = out.get("controller")
    if isinstance(controller, dict):
        controller_out = dict(controller)
        optimizer = controller_out.get("optimizer")
        if isinstance(optimizer, dict):
            migrated_opt, opt_changed = _migrate_optimizer(optimizer)
            controller_out["optimizer"] = migrated_opt
            changed = changed or opt_changed
        out["controller"] = controller_out
    return out, changed


def _migrate_jsonb_config(raw: Any) -> tuple[dict[str, Any], bool]:
    if raw is None:
        return {}, False
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}, False
    elif isinstance(raw, dict):
        parsed = dict(raw)
    else:
        return {}, False
    return migrate_runtime_config_mapping(parsed)


def migrate_registry_strategy_preset_configs(conn: Any) -> int:
    updated = 0
    with conn.cursor() as cur:
        cur.execute("SELECT preset_id, config FROM registry_strategy_presets")
        rows = cur.fetchall()
        for preset_id, config in rows:
            migrated, changed = _migrate_jsonb_config(config)
            if not changed:
                continue
            cur.execute(
                """
                UPDATE registry_strategy_presets
                SET config = %s::jsonb, updated_at = NOW()
                WHERE preset_id = %s
                """,
                (json.dumps(migrated), preset_id),
            )
            updated += 1
    return updated


def migrate_registry_strategy_config_overrides(conn: Any) -> int:
    updated = 0
    with conn.cursor() as cur:
        cur.execute("SELECT strategy_id, config_overrides FROM registry_strategies")
        rows = cur.fetchall()
        for strategy_id, overrides in rows:
            migrated, changed = _migrate_jsonb_config(overrides)
            if not changed:
                continue
            cur.execute(
                """
                UPDATE registry_strategies
                SET config_overrides = %s::jsonb, updated_at = NOW()
                WHERE strategy_id = %s
                """,
                (json.dumps(migrated), strategy_id),
            )
            updated += 1
    return updated


def migrate_registry_optimizer_config_names(conn: Any) -> int:
    total = 0
    total += migrate_registry_strategy_preset_configs(conn)
    total += migrate_registry_strategy_config_overrides(conn)
    return total
