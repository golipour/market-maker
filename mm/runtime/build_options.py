"""Runtime composition flags for medium-scale decomposition.

``RuntimeMode`` documents the intended deployment shape. ``RuntimeBuildOptions``
gates which subsystems ``build_runtime_components()`` wires so later phases can
extract ingestors, persistence workers, and multi-strategy pods without changing
the monolith default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RuntimeMode = Literal["monolith", "consumer_only", "ingestor_only", "persistence_only"]


@dataclass(frozen=True, slots=True)
class RuntimeBuildOptions:
    """Select which runtime subsystems to compose in one process."""

    mode: RuntimeMode = "monolith"
    enable_public_ingestion: bool = True
    enable_strategy: bool = True
    enable_private_state: bool = True
    enable_execution: bool = True
    enable_persistence: bool = True
    enable_observability: bool = True

    @classmethod
    def monolith(cls) -> RuntimeBuildOptions:
        return cls(mode="monolith")

    @classmethod
    def for_mode(cls, mode: RuntimeMode) -> RuntimeBuildOptions:
        if mode == "monolith":
            return cls.monolith()
        if mode == "consumer_only":
            return cls(
                mode=mode,
                enable_public_ingestion=False,
                enable_persistence=False,
            )
        if mode == "ingestor_only":
            return cls(
                mode=mode,
                enable_public_ingestion=True,
                enable_strategy=False,
                enable_private_state=False,
                enable_execution=False,
                enable_persistence=False,
            )
        if mode == "persistence_only":
            return cls(
                mode=mode,
                enable_public_ingestion=False,
                enable_strategy=False,
                enable_private_state=False,
                enable_execution=False,
                enable_persistence=True,
                enable_observability=True,
            )
        raise ValueError(f"unsupported runtime mode: {mode!r}")
