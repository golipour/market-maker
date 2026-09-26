"""Regression tests for the Bitwana dual overflow / lockup.

The old log-space dual ``λ = exp(θ)`` overflowed and then crushed the CE
objective at ``λ ~ 1e13``. These tests assert that the linear dual:

1. Never produces a non-finite multiplier.
2. Clamps to ``lambda_max`` under a sustained positive shortfall gap.
3. Stays at zero under a sustained healthy (zero) shortfall.
"""

from __future__ import annotations

import math

import pytest

from mm.mpc.objective_terms import (
    ViabilityParams,
    reset_viability_lambda,
    update_viability_lambda,
    viability_lambda_from_init,
)


def _params(*, eta: float = 1.0, epsilon: float = 0.0, lambda_max: float = 100.0) -> ViabilityParams:
    return ViabilityParams(
        min_viable_quote=100.0,
        shortfall_tolerance=epsilon,
        lambda_init=0.0,
        lambda_learning_rate=eta,
        lambda_max=lambda_max,
        dual_ewma_decay=0.0,
        recovery_fraction=0.0,
    )


class TestLambdaFromInit:
    @pytest.mark.parametrize("value", [-1e6, -1.0, 0.0, 0.5, 1.0, 30.0, 100.0, 1e6, 1e300])
    def test_does_not_raise(self, value: float) -> None:
        v = viability_lambda_from_init(value, lambda_max=100.0)
        assert math.isfinite(v)
        assert 0.0 <= v <= 100.0

    def test_huge_init_clips_to_max(self) -> None:
        assert viability_lambda_from_init(1e6, lambda_max=100.0) == 100.0

    def test_inf_init_clips_to_zero(self) -> None:
        assert viability_lambda_from_init(math.inf, lambda_max=100.0) == 0.0

    def test_nan_init_clips_to_zero(self) -> None:
        assert viability_lambda_from_init(math.nan, lambda_max=100.0) == 0.0


class TestUpdateViabilityLambda:
    def test_none_params_returns_current(self) -> None:
        lam, ewma = update_viability_lambda(0.5, {}, params=None, ewma=0.1)
        assert lam == 0.5
        assert ewma == 0.1

    def test_missing_components_returns_current(self) -> None:
        params = _params()
        lam, ewma = update_viability_lambda(0.5, {}, params=params, ewma=0.1)
        assert lam == 0.5
        assert ewma == 0.1

    def test_sustained_positive_gap_clamps_to_max(self) -> None:
        params = _params(eta=1.0, epsilon=0.0, lambda_max=100.0)
        components = {"viability_shortfall": 1.0}
        lam = 0.0
        ewma: float | None = None
        for _ in range(10_000):
            lam, ewma = update_viability_lambda(lam, components, params=params, ewma=ewma)
        assert lam == 100.0
        assert math.isfinite(lam)

    def test_sustained_healthy_wallet_clamps_to_zero(self) -> None:
        params = _params(eta=1.0, epsilon=0.05, lambda_max=100.0)
        components = {"viability_shortfall": 0.0}
        lam = 50.0
        ewma = 0.0
        for _ in range(10_000):
            lam, ewma = update_viability_lambda(lam, components, params=params, ewma=ewma)
        assert lam == 0.0

    def test_end_to_end_stability(self) -> None:
        params = _params(eta=5.0, epsilon=0.05, lambda_max=100.0)
        components = {"viability_shortfall": 1.0}
        lam = 0.0
        ewma: float | None = None
        for _ in range(100):
            lam, ewma = update_viability_lambda(lam, components, params=params, ewma=ewma)
            assert math.isfinite(lam)
            assert 0.0 <= lam <= 100.0
        assert lam == 100.0


class TestReset:
    def test_reset_fraction_default(self) -> None:
        assert reset_viability_lambda(current_lambda=10.0) == 5.0

    def test_reset_fraction_zero(self) -> None:
        assert reset_viability_lambda(current_lambda=10.0, reset_fraction=0.0) == 10.0

    def test_reset_fraction_one(self) -> None:
        assert reset_viability_lambda(current_lambda=10.0, reset_fraction=1.0) == 0.0

    def test_reset_invalid_fraction_raises(self) -> None:
        with pytest.raises(ValueError):
            reset_viability_lambda(current_lambda=1.0, reset_fraction=-0.1)
        with pytest.raises(ValueError):
            reset_viability_lambda(current_lambda=1.0, reset_fraction=1.5)
