"""Protocols for online-scored models."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OnlineScoredModel(Protocol):
    """Model that exposes params and optional observation scoring."""

    def get_params(self) -> dict[str, Any]:
        ...
