"""Validator that rejects optimizer configurations that cannot fit the
``decision_timeout_ms`` budget.

Background
==========

The 2026-09-01 production failure on Bitwana and Nobitex was
triggered by a configuration where the Monte Carlo MPC optimizer was
expected to take longer than ``decision_timeout_ms`` on every frame.
Each call therefore hit the timeout; the previous
``asyncio.to_thread``-based ``_decide_with_timeout`` could not cancel
the in-flight optimizer work, so the executor was gradually saturated
and the actor stopped publishing diagnostics. The fix in the runner
(cooperative cancellation via ``threading.Event``) addresses the
secondary symptom, but the structural fix is to reject the offending
configuration at load time so the system never enters that regime.

Cost model
==========

For the only optimizer we currently run in production
(``monte_carlo_mpc_tensor_batched``), the per-call compute is
dominated by the inner Adam loop and is well approximated as

    predicted_compute_ms = n_samples * horizon * adam_epochs * per_step_cost_ms

``per_step_cost_ms`` defaults to ``DEFAULT_PER_STEP_COST_MS = 2.5``
which is a conservative upper bound measured on the production image
on a single CPU thread. The validator compares the prediction to
``decision_timeout_ms * safety_factor`` (default ``0.7``) so a
rejection leaves headroom for the rest of the strategy pipeline
(``_decide_with_timeout`` overhead, JSON marshalling, etc.).
"""

from __future__ import annotations

from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Default per-inner-step cost for the Monte Carlo MPC tensor path.
#: Measured on the production image, single CPU thread.
DEFAULT_PER_STEP_COST_MS: float = 2.5

#: Default fraction of ``decision_timeout_ms`` that the optimizer is
#: allowed to consume. The remaining headroom covers the
#: ``_decide_with_timeout`` overhead, JSON marshalling, and the
#: per-frame publishing.
DEFAULT_SAFETY_FACTOR: float = 0.7

#: Optimizer kinds the validator understands. Anything else is
#: treated as "no cost model" and passed through.
_VALIDATED_KINDS = frozenset({"monte_carlo_mpc_tensor_batched"})


class OptimizerSizingError(ValueError):
    """Raised when the configured optimizer cannot fit the decision budget.

    The exception carries ``details`` so the config loader can log
    the offending fields and the caller can decide which knobs to
    relax.
    """

    def __init__(self, message: str, *, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_optimizer_compute_ms(
    optimizer: Mapping[str, Any] | None,
    *,
    per_step_cost_ms: float = DEFAULT_PER_STEP_COST_MS,
) -> tuple[int, int, int, int]:
    """Return ``(n_samples, horizon, adam_epochs, predicted_ms)``.

    Returns all zeros for missing / unknown optimizers, since the
    validator cannot make a prediction in that case.
    """
    if not optimizer:
        return (0, 0, 0, 0)
    if optimizer.get("kind") not in _VALIDATED_KINDS:
        return (0, 0, 0, 0)
    try:
        n_samples = int(optimizer.get("n_samples", 0) or 0)
        horizon = int(optimizer.get("horizon", 0) or 0)
        adam_epochs = int(optimizer.get("adam_epochs", 0) or 0)
    except (TypeError, ValueError):
        return (0, 0, 0, 0)
    if n_samples <= 0 or horizon <= 0 or adam_epochs <= 0:
        return (0, 0, 0, 0)
    inner_steps = n_samples * horizon * adam_epochs
    predicted = int(round(inner_steps * float(per_step_cost_ms)))
    return (n_samples, horizon, adam_epochs, predicted)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_optimizer_sizing(
    optimizer: Mapping[str, Any] | None,
    *,
    decision_timeout_ms: int | float,
    per_step_cost_ms: float = DEFAULT_PER_STEP_COST_MS,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> None:
    """Raise :class:`OptimizerSizingError` if the optimizer cannot fit.

    No-op for unknown / missing optimizers, since we have no cost
    model for them.
    """
    n_samples, horizon, adam_epochs, predicted = predict_optimizer_compute_ms(
        optimizer, per_step_cost_ms=per_step_cost_ms
    )
    if predicted == 0:
        return
    timeout = float(decision_timeout_ms)
    if timeout <= 0.0:
        return
    effective_budget = int(round(timeout * float(safety_factor)))
    if predicted <= effective_budget:
        return
    details = {
        "kind": optimizer.get("kind"),
        "n_samples": n_samples,
        "horizon": horizon,
        "adam_epochs": adam_epochs,
        "per_step_cost_ms": float(per_step_cost_ms),
        "predicted_compute_ms": predicted,
        "decision_timeout_ms": int(timeout),
        "safety_factor": float(safety_factor),
        "effective_budget_ms": effective_budget,
    }
    raise OptimizerSizingError(
        (
            f"Optimizer {details['kind']!r} predicted compute {predicted} ms "
            f"exceeds the effective budget of {effective_budget} ms "
            f"(decision_timeout_ms={int(timeout)} * safety_factor="
            f"{float(safety_factor)}) for n_samples={n_samples}, "
            f"horizon={horizon}, adam_epochs={adam_epochs}, "
            f"per_step_cost_ms={float(per_step_cost_ms)}. "
            f"Raise decision_timeout_ms, lower n_samples/horizon/adam_epochs, "
            f"or override per_step_cost_ms if your hardware is faster."
        ),
        details=details,
    )


def assert_optimizer_fits_budget(
    controller: Mapping[str, Any] | None,
    *,
    per_step_cost_ms: float = DEFAULT_PER_STEP_COST_MS,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> None:
    """Validator entry-point used by the config loader.

    The controller mapping must contain ``decision_timeout_ms`` and
    optionally ``optimizer``; missing ``optimizer`` or non-positive
    ``decision_timeout_ms`` are no-ops (the watchdog is disabled in
    that mode, so the validator has nothing to check).
    """
    if not controller:
        return
    timeout = controller.get("decision_timeout_ms", 0)
    try:
        timeout_f = float(timeout)
    except (TypeError, ValueError):
        return
    if timeout_f <= 0:
        return
    optimizer = controller.get("optimizer")
    if not optimizer:
        return
    validate_optimizer_sizing(
        optimizer,
        decision_timeout_ms=timeout_f,
        per_step_cost_ms=per_step_cost_ms,
        safety_factor=safety_factor,
    )
