"""One-shot Postgres migrations for registry JSON config key renames."""

from __future__ import annotations

import copy
import json
from pathlib import Path
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

_CANONICAL_PRESET_ID = "thin_local_liquidity_weighted_mpc"
_RUNTIME_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "market_maker.runtime.example.json"
)


def _canonical_optimizer() -> dict[str, Any]:
    payload = json.loads(_RUNTIME_EXAMPLE_PATH.read_text())
    presets = payload.get("strategy_presets")
    if not isinstance(presets, dict):
        raise ValueError("runtime example missing strategy_presets")
    preset = presets.get(_CANONICAL_PRESET_ID)
    if not isinstance(preset, dict):
        raise ValueError(f"runtime example missing strategy_presets.{_CANONICAL_PRESET_ID}")
    controller = preset.get("controller")
    if not isinstance(controller, dict):
        raise ValueError(f"strategy_presets.{_CANONICAL_PRESET_ID}.controller is required")
    optimizer = controller.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError(
            f"strategy_presets.{_CANONICAL_PRESET_ID}.controller.optimizer is required"
        )
    return copy.deepcopy(optimizer)


def _deep_merge_defaults(base: Any, defaults: Any) -> tuple[Any, bool]:
    if not isinstance(defaults, dict):
        return base, False
    if not isinstance(base, dict):
        return copy.deepcopy(defaults), True
    out = dict(base)
    changed = False
    for key, default_val in defaults.items():
        if str(key).startswith("_"):
            continue
        if key not in out:
            out[key] = copy.deepcopy(default_val)
            changed = True
            continue
        if isinstance(out[key], dict) and isinstance(default_val, dict):
            merged, child_changed = _deep_merge_defaults(out[key], default_val)
            if child_changed:
                out[key] = merged
                changed = True
    return out, changed


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


_LEGACY_VIABILITY_KEY_MAP = {
    "min_quote_qty": "min_viable_quote",
    "min_free_quote": "min_viable_quote",
    "margin_sharpness": "margin_buffer",
}


def _canonical_viability_defaults() -> dict[str, float]:
    viability = _canonical_optimizer().get("local", {}).get("viability")
    if not isinstance(viability, dict):
        raise ValueError("runtime example missing optimizer.local.viability")
    return {
        "min_viable_quote": float(viability.get("min_viable_quote", 50_000.0)),
        "margin_buffer": float(viability.get("margin_buffer", 5_000.0)),
        "availability_target": float(viability.get("availability_target", 0.99)),
        "shortfall_tolerance": float(viability.get("shortfall_tolerance", 0.05)),
        "lambda_init": float(viability.get("lambda_init", 0.0)),
        "lambda_learning_rate": float(viability.get("lambda_learning_rate", 0.02)),
        "lambda_max": float(viability.get("lambda_max", 100.0)),
        "dual_ewma_decay": float(viability.get("dual_ewma_decay", 0.9)),
    }


_LEGACY_VIABILITY_REMOVED_KEYS = (
    "lambda_leakage",
    "lambda_reset_on_target",
    "lambda_log_update",
)


def _strip_removed_viability_keys(viability: dict[str, Any]) -> bool:
    changed = False
    for key in _LEGACY_VIABILITY_REMOVED_KEYS:
        if key in viability:
            del viability[key]
            changed = True
    return changed


def _migrate_viability_keys(payload: dict[str, Any]) -> bool:
    viability = payload.get("viability")
    if not isinstance(viability, dict):
        return False
    changed = _rename_keys(viability, _LEGACY_VIABILITY_KEY_MAP)
    changed = _strip_removed_viability_keys(viability) or changed
    return changed


def _migrate_viability_min_viable_quote_semantics(payload: dict[str, Any]) -> bool:
    """Re-sync legacy base-unit thresholds to quote-currency viable-volume floors."""

    viability = payload.get("viability")
    if not isinstance(viability, dict):
        return False
    defaults = _canonical_viability_defaults()
    changed = False
    if "min_viable_quote" not in viability:
        viability["min_viable_quote"] = defaults["min_viable_quote"]
        changed = True
    cur = float(viability.get("min_viable_quote", defaults["min_viable_quote"]) or 0.0)
    canonical = defaults["min_viable_quote"]
    if cur <= 1_000.0 or cur >= canonical * 100.0:
        if cur != canonical:
            viability["min_viable_quote"] = canonical
            changed = True
    if "margin_buffer" not in viability:
        viability["margin_buffer"] = defaults["margin_buffer"]
        changed = True
    for key in ("lambda_max", "dual_ewma_decay"):
        if key not in viability:
            viability[key] = defaults[key]
            changed = True
    changed = _strip_removed_viability_keys(viability) or changed
    if float(viability.get("availability_target", defaults["availability_target"])) < 0.95:
        viability["availability_target"] = defaults["availability_target"]
        changed = True
    return changed


def _migrate_inventory_zone_to_viability(payload: dict[str, Any]) -> bool:
    changed = False
    if "inventory_zone" in payload or "w_inv" in payload:
        w_inv = float(payload.pop("w_inv", 0.0) or 0.0)
        payload.pop("inventory_zone", None)
        if "viability" not in payload:
            defaults = _canonical_viability_defaults()
            payload["viability"] = {
                **defaults,
                "lambda_init": max(0.1, w_inv * 100.0) if w_inv > 0.0 else defaults["lambda_init"],
            }
        changed = True
    return changed


def _migrate_w_book_to_validity(payload: dict[str, Any]) -> bool:
    changed = False
    if "w_book" in payload:
        payload.pop("w_book")
        changed = True
    if "validity" not in payload:
        payload["validity"] = {
            "min_score": 0.3,
            "penalty_scale": 100.0,
        }
        changed = True
    return changed


def _migrate_venue_optimizer(payload: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(payload, dict):
        return {}, False
    out = dict(payload)
    changed = _rename_keys(out, _LEGACY_VENUE_KEY_MAP)
    changed = _migrate_inventory_zone_to_viability(out) or changed
    changed = _migrate_viability_keys(out) or changed
    changed = _migrate_viability_min_viable_quote_semantics(out) or changed
    changed = _migrate_w_book_to_validity(out) or changed
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


def _migrate_warm_start(payload: dict[str, Any]) -> bool:
    ws = payload.get("warm_start")
    if isinstance(ws, bool):
        enabled = bool(ws)
        payload["warm_start"] = {
            "enabled": enabled,
            "preserve_optimizer_state": enabled,
            "alpha": 1.0 if enabled else 0.0,
            "alpha_mode": "fixed",
            "alpha_max": 0.95,
            "grad_norm_eps": 1e-12,
        }
        return True
    if ws is None:
        canonical = _canonical_optimizer().get("warm_start")
        if isinstance(canonical, dict):
            payload["warm_start"] = copy.deepcopy(canonical)
            return True
    return False


def _migrate_optimizer(payload: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(payload, dict):
        return {}, False
    out = dict(payload)
    changed = _migrate_warm_start(out)
    for venue in ("local", "global"):
        if venue in out and isinstance(out[venue], dict):
            migrated, venue_changed = _migrate_venue_optimizer(out[venue])
            out[venue] = migrated
            changed = changed or venue_changed
    merged, merge_changed = _deep_merge_defaults(out, _canonical_optimizer())
    out = merged
    changed = changed or merge_changed
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
