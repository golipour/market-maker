"""Regression tests for the Bitwana ``OverflowError: math range error`` bug.

Observed on 2026-09-01 23:42:30 UTC, the Bitwana actor emitted
``OverflowError: math range error`` on every frame. The root cause
was unbounded additive updates to the viability dual state ``theta``,
so that ``math.exp(theta)`` overflowed on the next call.

These tests assert that:

1. ``viability_lambda_from_theta`` never raises for any finite input.
2. ``update_viability_theta`` clamps ``theta`` to a finite range so
   the dual state cannot run away under a sustained positive gap.
3. The end-to-end update/lambda cycle is stable under pathological
   inputs (a sequence of very large positive gaps).
"""

from __future__ import annotations

import math

import pytest

from mm.mpc.objective_terms import (
    VIABILITY_THETA_MAX,
    VIABILITY_THETA_MIN,
    update_viability_theta,
    viability_lambda_from_theta,
    viability_theta_from_lambda_init,
)


class _FakeParams:
    """Minimal stand-in for ``ViabilityParams`` used by these tests."""

    def __init__(self, eta: float = 1.0, target: float = 1.0) -> None:
        self.lambda_learning_rate = eta
        self.availability_target = target


# ---------------------------------------------------------------------------
# lambda_from_theta: must never raise, must return finite for any finite input
# ---------------------------------------------------------------------------


class TestLambdaFromTheta:
    @pytest.mark.parametrize("theta", [-1e6, -1.0, 0.0, 0.5, 1.0, 30.0, 100.0, 1e6, 1e300])
    def test_does_not_raise(self, theta: float) -> None:
        v = viability_lambda_from_theta(theta)
        assert math.isfinite(v) or v == 0.0

    def test_huge_theta_returns_zero(self) -> None:
        # Was: OverflowError("math range error"). Now: 0.0.
        assert viability_lambda_from_theta(1e6) == 0.0

    def test_inf_theta_returns_zero(self) -> None:
        assert viability_lambda_from_theta(math.inf) == 0.0

    def test_neg_inf_theta_returns_zero(self) -> None:
        assert viability_lambda_from_theta(-math.inf) == 0.0

    def test_nan_theta_returns_zero(self) -> None:
        assert viability_lambda_from_theta(math.nan) == 0.0

    def test_normal_theta_returns_exp(self) -> None:
        # Below the overflow threshold, exp(theta) is the answer.
        assert viability_lambda_from_theta(2.0) == pytest.approx(math.exp(2.0), rel=1e-9)


# ---------------------------------------------------------------------------
# theta_from_lambda_init: must be well-defined for any finite input
# ---------------------------------------------------------------------------


class TestThetaFromLambdaInit:
    def test_zero_init(self) -> None:
        assert viability_theta_from_lambda_init(0.0) == -math.inf

    def test_negative_init(self) -> None:
        assert viability_theta_from_lambda_init(-1.0) == -math.inf

    def test_positive_init(self) -> None:
        assert viability_theta_from_lambda_init(1.0) == pytest.approx(0.0, abs=1e-9)

    def test_large_init(self) -> None:
        # log(huge) is fine; the function never raises.
        theta = viability_theta_from_lambda_init(1e6)
        assert math.isfinite(theta)
        assert math.exp(theta) == pytest.approx(1e6, rel=1e-6)


# ---------------------------------------------------------------------------
# update_viability_theta: clamps to [theta_min, theta_max] under all inputs
# ---------------------------------------------------------------------------


class TestUpdateViabilityTheta:
    def test_none_params_returns_current(self) -> None:
        assert update_viability_theta(0.5, {}, params=None) == 0.5

    def test_missing_components_returns_current(self) -> None:
        # ``_signed_availability_gap`` returns None when the key is missing.
        params = _FakeParams(eta=1.0, target=1.0)
        assert update_viability_theta(0.5, {}, params=params) == 0.5

    def test_sustained_positive_gap_clamps_to_max(self) -> None:
        """The original Bitwana bug: a positive gap accumulates forever."""
        params = _FakeParams(eta=1.0, target=1.0)
        # The components dict needs ``viability_min_availability`` so the
        # gap function returns ``target - min`` = 1.0.
        components = {"viability_min_availability": 0.0}
        theta = 0.0
        for _ in range(10_000):
            theta = update_viability_theta(theta, components, params=params)
        # Without the clamp this would be 10_000, well past ``math.exp``
        # territory. With the clamp it must be the configured ceiling.
        assert theta == VIABILITY_THETA_MAX
        # And the subsequent lambda computation must be finite.
        assert math.isfinite(viability_lambda_from_theta(theta))

    def test_sustained_negative_gap_clamps_to_min(self) -> None:
        params = _FakeParams(eta=1.0, target=1.0)
        # Negative gap: target - min_availability = 1 - 100 = -99
        components = {"viability_min_availability": 100.0}
        theta = 0.0
        for _ in range(10_000):
            theta = update_viability_theta(theta, components, params=params)
        assert theta == VIABILITY_THETA_MIN

    def test_custom_clamp_bounds(self) -> None:
        """Callers can tighten or relax the clamp via kwargs."""
        params = _FakeParams(eta=1.0, target=1.0)
        components = {"viability_min_availability": 0.0}
        theta = update_viability_theta(
            0.0,
            components,
            params=params,
            theta_min=-1.0,
            theta_max=1.0,
        )
        assert theta == 1.0

    def test_end_to_end_stability(self) -> None:
        """Run the actual production cycle: update, then lambda, repeatedly."""
        params = _FakeParams(eta=0.5, target=1.0)
        # Force a positive gap to stress the system.
        components = {"viability_min_availability": 0.0}
        theta = 0.0
        for _ in range(100):
            theta = update_viability_theta(theta, components, params=params)
            lam = viability_lambda_from_theta(theta)
            assert math.isfinite(lam)
            assert 0.0 <= lam
        # Final theta is the clamp; lambda is finite.
        assert theta <= VIABILITY_THETA_MAX
        assert math.isfinite(viability_lambda_from_theta(theta))


# ---------------------------------------------------------------------------
# Leaky decay and reset
# ---------------------------------------------------------------------------


class TestDecay:
    def test_zero_decay_is_pure_dual_ascent(self) -> None:
        params = _FakeParams(eta=1.0, target=1.0)
        components = {"viability_min_availability": 0.0}
        # With decay=0 the function is unchanged from the original.
        assert update_viability_theta(
            0.0, components, params=params, decay_rate=0.0
        ) == 1.0
        assert update_viability_theta(
            5.0, components, params=params, decay_rate=0.0
        ) == 6.0

    def test_decay_shrinks_state_each_call(self) -> None:
        params = _FakeParams(eta=0.0, target=1.0)  # no ascent
        components = {"viability_min_availability": 1.0}  # gap = 0
        # With zero eta and zero gap, the only motion is decay.
        theta = 10.0
        for _ in range(10):
            theta = update_viability_theta(
                theta, components, params=params, decay_rate=0.01
            )
        # After 10 calls at 1% per call: 10 * 0.99 ≈ 9.56
        assert theta == pytest.approx(10.0 * (0.99 ** 10), rel=1e-9)

    def test_decay_does_not_override_gap(self) -> None:
        """A positive gap must still grow the state even with decay."""
        params = _FakeParams(eta=1.0, target=1.0)
        components = {"viability_min_availability": 0.0}
        # Start at 5.0; with decay=0.1 and eta=1, gap=1: 5*0.9 + 1 = 5.5
        theta = update_viability_theta(
            5.0, components, params=params, decay_rate=0.1
        )
        assert theta == pytest.approx(5.5, rel=1e-9)

    def test_decay_still_clamps(self) -> None:
        """The clamp must still apply after a decay step."""
        params = _FakeParams(eta=0.0, target=1.0)
        components = {"viability_min_availability": 1.0}  # gap=0
        # Start above VIABILITY_THETA_MAX; decay alone would still cap it.
        theta = update_viability_theta(
            VIABILITY_THETA_MAX + 100.0,
            components,
            params=params,
            decay_rate=0.5,
        )
        assert theta == VIABILITY_THETA_MAX


class TestReset:
    def test_reset_fraction_default(self) -> None:
        from mm.mpc.objective_terms import reset_viability_theta

        # default reset_fraction = 0.5
        assert reset_viability_theta(current_theta=10.0) == 5.0

    def test_reset_fraction_zero(self) -> None:
        from mm.mpc.objective_terms import reset_viability_theta

        # 0.0 fraction means "no change"
        assert reset_viability_theta(current_theta=10.0, reset_fraction=0.0) == 10.0

    def test_reset_fraction_one(self) -> None:
        from mm.mpc.objective_terms import reset_viability_theta

        # 1.0 fraction means "fully reset to zero"
        assert reset_viability_theta(current_theta=10.0, reset_fraction=1.0) == 0.0

    def test_reset_invalid_fraction_raises(self) -> None:
        from mm.mpc.objective_terms import reset_viability_theta

        with pytest.raises(ValueError):
            reset_viability_theta(current_theta=1.0, reset_fraction=-0.1)
        with pytest.raises(ValueError):
            reset_viability_theta(current_theta=1.0, reset_fraction=1.5)
