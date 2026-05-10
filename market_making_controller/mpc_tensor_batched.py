"""Lazy facade for MPC symbols; implementation lives in ``mm.mpc.mpc_tensor_batched``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mm.mpc.mpc_tensor_batched import MPCOptimizer, MonteCarloMPCTensorBatched

__all__ = ["MPCOptimizer", "MonteCarloMPCTensorBatched"]


def __getattr__(name: str):
    if name == "MPCOptimizer":
        from mm.mpc.mpc_tensor_batched import MPCOptimizer as _MPCOptimizer

        return _MPCOptimizer
    if name == "MonteCarloMPCTensorBatched":
        from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched as _MC

        return _MC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
