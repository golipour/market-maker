"""Worker runtime configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mm.runtime.config import BusRuntimeConfig, ObservabilityRuntimeConfig, _bus_from_mapping, _observability_from_mapping


WorkerMode = Literal["multi", "single"]


@dataclass(frozen=True, slots=True)
class StrategyBackoffConfig:
    initial_s: float = 1.0
    max_s: float = 60.0


@dataclass(frozen=True, slots=True)
class WorkerRuntimeConfig:
    worker_id: str
    assignment_id: str | None = None
    strategy_ids: tuple[str, ...] = ()
    mode: WorkerMode = "multi"
    backoff: StrategyBackoffConfig = field(default_factory=StrategyBackoffConfig)
    heartbeat_interval_s: int = 30
    lease_ttl_seconds: int = 300
    bus: BusRuntimeConfig | None = None
    observability: ObservabilityRuntimeConfig = field(default_factory=ObservabilityRuntimeConfig)
    registry_database_url_env: str = "MM_DATABASE_URL"

    def resolved_registry_database_url(self) -> str | None:
        return os.environ.get(self.registry_database_url_env, "").strip() or None

    def resolved_strategy_ids(self, assignment_strategy_ids: tuple[str, ...]) -> tuple[str, ...]:
        if self.strategy_ids:
            ids = self.strategy_ids
        else:
            ids = assignment_strategy_ids
        if self.mode == "single" and ids:
            return (ids[0],)
        return ids

    def as_contract(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "assignment_id": self.assignment_id,
            "strategy_ids": list(self.strategy_ids),
            "mode": self.mode,
            "backoff": {
                "initial_s": self.backoff.initial_s,
                "max_s": self.backoff.max_s,
            },
            "heartbeat_interval_s": self.heartbeat_interval_s,
            "lease_ttl_seconds": self.lease_ttl_seconds,
            "bus": None if self.bus is None else {"kind": self.bus.kind, "consumer_group": self.bus.consumer_group},
            "observability": {
                "service_name": self.observability.service_name,
                "metrics_enabled": self.observability.metrics_enabled,
                "metrics_port": self.observability.metrics_port,
            },
        }


def load_worker_config(path: str | Path) -> WorkerRuntimeConfig:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("worker config file must contain a JSON object")
    return worker_config_from_mapping(payload)


def worker_config_from_mapping(payload: dict[str, Any]) -> WorkerRuntimeConfig:
    worker_id = str(payload.get("worker_id", "")).strip()
    if not worker_id:
        raise ValueError("worker.worker_id is required")

    strategy_ids_payload = payload.get("strategy_ids")
    strategy_ids: tuple[str, ...] = ()
    if isinstance(strategy_ids_payload, str):
        strategy_ids = (strategy_ids_payload.strip(),)
    elif isinstance(strategy_ids_payload, list):
        strategy_ids = tuple(str(item).strip() for item in strategy_ids_payload if str(item).strip())

    backoff_payload = payload.get("backoff")
    backoff = StrategyBackoffConfig()
    if isinstance(backoff_payload, dict):
        backoff = StrategyBackoffConfig(
            initial_s=max(0.1, float(backoff_payload.get("initial_s", 1.0))),
            max_s=max(0.1, float(backoff_payload.get("max_s", 60.0))),
        )

    bus_payload = payload.get("bus")
    bus = _bus_from_mapping(bus_payload) if isinstance(bus_payload, dict) else None
    registry_payload = payload.get("registry")
    registry_database_url_env = "MM_DATABASE_URL"
    if isinstance(registry_payload, dict):
        registry_database_url_env = str(
            registry_payload.get("database_url_env", registry_database_url_env)
        )

    mode = str(payload.get("mode", "multi")).strip().lower()
    if mode not in ("multi", "single"):
        raise ValueError("worker.mode must be 'multi' or 'single'")

    return WorkerRuntimeConfig(
        worker_id=worker_id,
        assignment_id=str(payload.get("assignment_id", "")).strip() or None,
        strategy_ids=strategy_ids,
        mode=mode,  # type: ignore[arg-type]
        backoff=backoff,
        heartbeat_interval_s=max(5, int(payload.get("heartbeat_interval_s", 30))),
        lease_ttl_seconds=max(30, int(payload.get("lease_ttl_seconds", 300))),
        bus=bus,
        observability=_observability_from_mapping(payload.get("observability")),
        registry_database_url_env=registry_database_url_env,
    )
