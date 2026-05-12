"""Backward-compatible entry for diagnostics projection (see ``mm.observability.diagnostics``)."""

from mm.observability.diagnostics import (
    controller_decision_to_diagnostics_frame,
    emit_controller_diagnostics,
)

__all__ = ["controller_decision_to_diagnostics_frame", "emit_controller_diagnostics"]
