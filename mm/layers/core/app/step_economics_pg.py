"""Compatibility shim: canonical implementation lives in ``mm.persistence.step_economics_pg``."""

from mm.persistence.step_economics_pg import keep_runner_pnl_series_memory, persist_step_economics_row

__all__ = ["keep_runner_pnl_series_memory", "persist_step_economics_row"]
