"""Render one-bot runtime JSON from registry records."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from mm.registry.credentials import merge_private_config
from mm.registry.models import (
    AccountRecord,
    ExchangeRecord,
    PlatformSettingsRecord,
    StrategyPresetRecord,
    StrategyRecord,
)
from mm.runtime.config import MarketMakerRuntimeConfig, runtime_config_from_mapping


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def runtime_config_mapping_from_registry(
    strategy: StrategyRecord,
    *,
    account: AccountRecord,
    preset: StrategyPresetRecord,
    exchanges: dict[str, ExchangeRecord],
    platform: PlatformSettingsRecord,
) -> dict[str, Any]:
    """Build a one-bot runtime config mapping from registry rows."""

    quoting = strategy.quoting_exchange.lower().strip()
    liquidity = strategy.liquidity_exchange.lower().strip()
    if account.exchange.lower().strip() != quoting:
        raise ValueError(
            f"strategy {strategy.strategy_id} quoting exchange {quoting} "
            f"does not match account exchange {account.exchange}"
        )

    exchanges_payload: dict[str, Any] = {}
    for exchange_name in dict.fromkeys((quoting, liquidity)):
        record = exchanges.get(exchange_name)
        if record is None:
            raise ValueError(f"registry exchange {exchange_name!r} is required")
        exchange_config = copy.deepcopy(record.config)
        if exchange_name == quoting:
            private = exchange_config.get("private")
            exchange_config["private"] = merge_private_config(
                private if isinstance(private, dict) else None,
                credential_ref=account.credential_ref,
            )
        else:
            exchange_config.pop("private", None)
        exchanges_payload[exchange_name] = exchange_config

    preset_config = deep_merge(copy.deepcopy(preset.config), strategy.config_overrides)
    controller = preset_config.setdefault("controller", {})
    if not isinstance(controller, dict):
        raise ValueError(f"preset {preset.preset_id} controller must be an object")
    controller["actor_id"] = strategy.actor_id

    payload: dict[str, Any] = {
        "exchanges": exchanges_payload,
        "strategy_presets": {preset.preset_id: preset_config},
        "bots": {
            strategy.strategy_id: {
                "preset": preset.preset_id,
                "symbol": strategy.symbol,
                "quoting_exchange": quoting,
                "liquidity_exchange": liquidity,
            }
        },
    }
    if platform.bus:
        payload["bus"] = copy.deepcopy(platform.bus)
    if platform.persistence:
        payload["persistence"] = copy.deepcopy(platform.persistence)
    if platform.observability:
        payload["observability"] = copy.deepcopy(platform.observability)
    if platform.scheduler:
        payload["scheduler"] = copy.deepcopy(platform.scheduler)
    return payload


def runtime_config_from_registry(
    strategy: StrategyRecord,
    *,
    account: AccountRecord,
    preset: StrategyPresetRecord,
    exchanges: dict[str, ExchangeRecord],
    platform: PlatformSettingsRecord,
) -> MarketMakerRuntimeConfig:
    mapping = runtime_config_mapping_from_registry(
        strategy,
        account=account,
        preset=preset,
        exchanges=exchanges,
        platform=platform,
    )
    return runtime_config_from_mapping(mapping)


def load_platform_template(path: str | Path) -> PlatformSettingsRecord:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("platform template must be a JSON object")
    return PlatformSettingsRecord(
        settings_id="template",
        bus=dict(payload.get("bus") or {}),
        persistence=dict(payload.get("persistence") or {}),
        observability=dict(payload.get("observability") or {}),
        scheduler=dict(payload.get("scheduler") or {}),
    )


def default_platform_settings_from_env() -> PlatformSettingsRecord:
    template_path = os.environ.get("MM_PLATFORM_RUNTIME_TEMPLATE", "").strip()
    if template_path:
        return load_platform_template(template_path)
    return PlatformSettingsRecord(settings_id="default")
