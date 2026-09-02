"""Tests for the optimizer sizing validator.

The validator runs at config-load time and rejects any configuration
where the predicted MPC compute exceeds ``decision_timeout_ms *
safety_factor``. Without this guard the system enters a regime where
the timeout is the expected path and ``asyncio.to_thread`` leaks work
into the thread pool, which is exactly the failure mode that stopped
Bitwana and Nobitex on 2026-09-01.
"""

from __future__ import annotations

import math

import pytest

from mm.mpc.optimizer_sizing import (
    DEFAULT_PER_STEP_COST_MS,
    DEFAULT_SAFETY_FACTOR,
    OptimizerSizingError,
    assert_optimizer_fits_budget,
    predict_optimizer_compute_ms,
    validate_optimizer_sizing,
)


class TestPredictCompute:
    def test_returns_zero_for_missing_optimizer(self) -> None:
        assert predict_optimizer_compute_ms(None) == (0, 0, 0, 0)

    def test_returns_zero_for_unknown_kind(self) -> None:
        opt = {"kind": "linear_quadratic", "n_samples": 100, "horizon": 10, "adam_epochs": 100}
        assert predict_optimizer_compute_ms(opt) == (0, 0, 0, 0)

    def test_computes_inner_steps(self) -> None:
        opt = {
            "kind": "monte_carlo_mpc_tensor_batched",
            "n_samples": 64,
            "horizon": 2,
            "adam_epochs": 40,
        }
        n, h, e, predicted = predict_optimizer_compute_ms(opt)
        assert (n, h, e) == (64, 2, 40)
        # 64 * 2 * 40 = 5120 inner steps * 2.5 ms = 12800 ms
        assert predicted == 12800

    def test_respects_per_step_cost_override(self) -> None:
        opt = {
            "kind": "monte_carlo_mpc_tensor_batched",
            "n_samples": 4,
            "horizon": 1,
            "adam_epochs": 8,
        }
        # 4 * 1 * 8 = 32 * 1.0 = 32 ms
        assert predict_optimizer_compute_ms(opt, per_step_cost_ms=1.0)[3] == 32
        # 4 * 1 * 8 = 32 * 0.5 = 16 ms
        assert predict_optimizer_compute_ms(opt, per_step_cost_ms=0.5)[3] == 16

    def test_zero_optimizer_is_zero_compute(self) -> None:
        # Even with the validated kind, zero compute is zero compute.
        opt = {
            "kind": "monte_carlo_mpc_tensor_batched",
            "n_samples": 0,
            "horizon": 0,
            "adam_epochs": 0,
        }
        assert predict_optimizer_compute_ms(opt) == (0, 0, 0, 0)


class TestValidateOptimizerSizing:
    def test_rejects_oversized_config(self) -> None:
        opt = {
            "kind": "monte_carlo_mpc_tensor_batched",
            "n_samples": 64,
            "horizon": 2,
            "adam_epochs": 40,
        }
        # 64*2*40*2.5 = 12800 ms; budget 3000*0.7 = 2100 ms
        with pytest.raises(OptimizerSizingError) as exc:
            validate_optimizer_sizing(opt, decision_timeout_ms=3000)
        details = exc.value.details
        assert details["n_samples"] == 64
        assert details["horizon"] == 2
        assert details["adam_epochs"] == 40
        assert details["predicted_compute_ms"] == 12800
        assert details["effective_budget_ms"] == 2100

    def test_accepts_config_that_fits(self) -> None:
        opt = {
            "kind": "monte_carlo_mpc_tensor_batched",
            "n_samples": 8,
            "horizon": 2,
            "adam_epochs": 10,
        }
        # 8*2*10*2.5 = 400 ms; budget 3000*0.7 = 2100 ms
        validate_optimizer_sizing(opt, decision_timeout_ms=3000)

    def test_no_op_for_unknown_kind(self) -> None:
        # The validator only knows the cost model for the Monte Carlo
        # batched tensor path. Other optimizers are passed through.
        opt = {
            "kind": "linear_quadratic",
            "n_samples": 10_000,  # would obviously fail the check
            "horizon": 10_000,
            "adam_epochs": 10_000,
        }
        validate_optimizer_sizing(opt, decision_timeout_ms=100)

    def test_no_op_for_missing_optimizer(self) -> None:
        validate_optimizer_sizing(None, decision_timeout_ms=100)

    def test_per_step_cost_override(self) -> None:
        opt = {
            "kind": "monte_carlo_mpc_tensor_batched",
            "n_samples": 4,
            "horizon": 1,
            "adam_epochs": 8,
        }
        # 4*1*8 = 32 inner steps
        # With cost 100 ms, predicted = 3200; budget 3000*0.7 = 2100 -> reject
        with pytest.raises(OptimizerSizingError):
            validate_optimizer_sizing(opt, decision_timeout_ms=3000, per_step_cost_ms=100.0)
        # With cost 1.0 ms, predicted = 32; budget 2100 -> accept
        validate_optimizer_sizing(opt, decision_timeout_ms=3000, per_step_cost_ms=1.0)

    def test_safety_factor_override(self) -> None:
        opt = {
            "kind": "monte_carlo_mpc_tensor_batched",
            "n_samples": 8,
            "horizon": 1,
            "adam_epochs": 8,
        }
        # 8*1*8*2.5 = 160 ms
        # safety_factor=0.01 on timeout=2000: budget=20 ms; predicted=160 -> reject
        with pytest.raises(OptimizerSizingError):
            validate_optimizer_sizing(opt, decision_timeout_ms=2000, safety_factor=0.01)
        # safety_factor=1.0 on timeout=2000: budget=2000 ms; predicted=160 -> accept
        validate_optimizer_sizing(opt, decision_timeout_ms=2000, safety_factor=1.0)

    def test_error_message_includes_fix(self) -> None:
        opt = {
            "kind": "monte_carlo_mpc_tensor_batched",
            "n_samples": 64,
            "horizon": 2,
            "adam_epochs": 40,
        }
        with pytest.raises(OptimizerSizingError) as exc:
            validate_optimizer_sizing(opt, decision_timeout_ms=3000)
        msg = str(exc.value)
        # The message should name the fields to change.
        assert "decision_timeout_ms" in msg
        assert "n_samples" in msg
        assert "horizon" in msg
        assert "adam_epochs" in msg


class TestAssertOptimizerFitsBudget:
    def test_passes_for_safe_controller(self) -> None:
        controller = {
            "decision_timeout_ms": 5000,
            "optimizer": {
                "kind": "monte_carlo_mpc_tensor_batched",
                "n_samples": 8,
                "horizon": 2,
                "adam_epochs": 10,
            },
        }
        # 8*2*10*2.5 = 400 ms; budget 5000*0.7 = 3500 -> accept
        assert_optimizer_fits_budget(controller)

    def test_raises_for_oversized_controller(self) -> None:
        controller = {
            "decision_timeout_ms": 3000,
            "optimizer": {
                "kind": "monte_carlo_mpc_tensor_batched",
                "n_samples": 64,
                "horizon": 2,
                "adam_epochs": 40,
            },
        }
        with pytest.raises(OptimizerSizingError):
            assert_optimizer_fits_budget(controller)

    def test_no_op_when_no_optimizer(self) -> None:
        assert_optimizer_fits_budget({"decision_timeout_ms": 100})

    def test_no_op_when_no_timeout(self) -> None:
        # A non-positive ``decision_timeout_ms`` disables the watchdog
        # and the validator skips the check.
        assert_optimizer_fits_budget(
            {
                "decision_timeout_ms": 0,
                "optimizer": {
                    "kind": "monte_carlo_mpc_tensor_batched",
                    "n_samples": 10_000,
                    "horizon": 10_000,
                    "adam_epochs": 10_000,
                },
            }
        )

    def test_no_op_for_non_mpc_kind(self) -> None:
        assert_optimizer_fits_budget(
            {
                "decision_timeout_ms": 100,
                "optimizer": {
                    "kind": "linear_quadratic",
                    "n_samples": 10_000,
                },
            }
        )


class TestDefaults:
    def test_default_per_step_cost(self) -> None:
        # Sanity check on the documented default.
        assert DEFAULT_PER_STEP_COST_MS == pytest.approx(2.5, rel=1e-9)

    def test_default_safety_factor(self) -> None:
        assert DEFAULT_SAFETY_FACTOR == pytest.approx(0.7, rel=1e-9)


class TestRealisticConfigurations:
    """End-to-end tests on production-style configurations."""

    def test_nobitex_production_fails(self) -> None:
        """The configuration that was running on 2026-09-01 must be rejected.

        This is the regression test: the bitwana/nobitex production
        config that timed out every frame should no longer be loadable.
        """
        controller = {
            "decision_timeout_ms": 3000,
            "optimizer": {
                "kind": "monte_carlo_mpc_tensor_batched",
                "n_samples": 64,
                "horizon": 2,
                "adam_epochs": 40,
            },
        }
        with pytest.raises(OptimizerSizingError) as exc:
            assert_optimizer_fits_budget(controller)
        assert "exceeds" in str(exc.value).lower()

    def test_updated_example_config_passes(self) -> None:
        """The updated example config (decision_timeout_ms=10000, adam_epochs=20) must load."""
        controller = {
            "decision_timeout_ms": 10000,
            "optimizer": {
                "kind": "monte_carlo_mpc_tensor_batched",
                "n_samples": 64,
                "horizon": 2,
                "adam_epochs": 20,
            },
        }
        # 64*2*20*2.5 = 6400 ms; budget 10000*0.7 = 7000 -> accept
        assert_optimizer_fits_budget(controller)
