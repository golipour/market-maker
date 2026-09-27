"""Tests for the configurable optimizer dtype and torch thread control.

Two measured facts drive this module, both established on the Bitwana
optimizer configuration on CPU:

1. ``tensor_dtype`` must be honored end to end. ``float32`` produces
   numerically equivalent objectives (relative difference ~1e-5) but is
   **not** meaningfully faster than ``float64`` (measured ratio 0.96-1.04x),
   so it stays opt-in and is not presented as a performance win.

2. The rollout tensors are small enough that torch's default intra-op thread
   fan-out costs more in synchronization than it recovers:

   ============  ============
   threads       wall time
   ============  ============
   1             178.7 ms
   2             186.7 ms
   4 (default)   195.1 ms
   8             215.9 ms
   ============  ============

   So the optimizer pins the pool to a single thread by default.
"""

from __future__ import annotations

import pytest
import torch

from mm.mpc.mpc_state import OrderbookRelative, State
from mm.mpc.mpc_tensor_batched import (
    SUPPORTED_TENSOR_DTYPES,
    MonteCarloMPCTensorBatched,
    _resolve_tensor_dtype,
)
from mm.mpc.torch_device import resolve_torch_threads


class _FixedFlowModel:
    def sample(self, sample_shape, device=None, dtype=None):
        dt = dtype or torch.float64
        return (
            torch.full(sample_shape, 2.0, dtype=dt, device=device),
            torch.full(sample_shape, 1.0, dtype=dt, device=device),
        )


class _FixedPriceModel:
    def sample_residual(self, sample_shape, device=None, dtype=None):
        z = torch.zeros(sample_shape, dtype=dtype or torch.float64, device=device)
        return z, z


class _FixedExecutionModel:
    maker_fee = 0.0


def _minimal_state() -> State:
    return State(
        mid=100.0,
        ob_rel=OrderbookRelative(
            bids=((1.0, 50.0), (2.0, 30.0)),
            asks=((1.0, 40.0), (2.0, 25.0)),
        ),
        buy_vol=2.0,
        sell_vol=1.0,
        inventory=0.0,
        cash=10_000.0,
        buy_vwap=99.5,
        sell_vwap=100.5,
    )


def _optimizer(**kwargs) -> MonteCarloMPCTensorBatched:
    params = dict(
        local_price_model=_FixedPriceModel(),
        local_market_flow_model=_FixedFlowModel(),
        execution_model=_FixedExecutionModel(),
        n_components=2,
        n_samples=4,
        horizon=2,
        spread_max=10.0,
        spread_step=1.0,
        adam_lr=0.05,
        adam_epochs=3,
        reference_price_source="local",
    )
    params.update(kwargs)
    return MonteCarloMPCTensorBatched(**params)


class TestResolveTensorDtype:
    def test_float64_is_the_default(self) -> None:
        assert _resolve_tensor_dtype(torch, None) is torch.float64
        assert _resolve_tensor_dtype(torch, "float64") is torch.float64

    def test_float32_is_supported(self) -> None:
        assert _resolve_tensor_dtype(torch, "float32") is torch.float32
        assert _resolve_tensor_dtype(torch, "FLOAT32") is torch.float32

    def test_aliases_are_accepted(self) -> None:
        assert _resolve_tensor_dtype(torch, "double") is torch.float64
        assert _resolve_tensor_dtype(torch, "f64") is torch.float64
        assert _resolve_tensor_dtype(torch, "float") is torch.float32
        assert _resolve_tensor_dtype(torch, "f32") is torch.float32

    def test_unknown_dtype_raises_rather_than_silently_falling_back(self) -> None:
        # Silently coercing a typo would quietly change live numerics.
        with pytest.raises(ValueError, match="unsupported tensor_dtype"):
            _resolve_tensor_dtype(torch, "float16")

    def test_supported_list_is_documented(self) -> None:
        assert SUPPORTED_TENSOR_DTYPES == ("float64", "float32")


class TestOptimizerDtypeConfig:
    def test_default_constructor_dtype_is_float64(self) -> None:
        assert _optimizer().tensor_dtype == "float64"

    def test_constructor_rejects_unknown_dtype_eagerly(self) -> None:
        # Failure at construction, not on the first live decision.
        with pytest.raises(ValueError, match="unsupported tensor_dtype"):
            _optimizer(tensor_dtype="bfloat16")

    def test_float32_optimizer_runs_and_reports_dtype(self) -> None:
        _, diagnostics = _optimizer(
            tensor_dtype="float32", fixed_mc_samples=True
        ).optimize_with_diagnostics(_minimal_state())
        assert diagnostics.config["tensor_dtype"] == "float32"

    def test_float64_and_float32_agree_on_objective(self) -> None:
        state = _minimal_state()
        _, f64 = _optimizer(
            tensor_dtype="float64", adam_epochs=20, fixed_mc_samples=True
        ).optimize_with_diagnostics(state)
        _, f32 = _optimizer(
            tensor_dtype="float32", adam_epochs=20, fixed_mc_samples=True
        ).optimize_with_diagnostics(state)

        assert f64.best is not None and f32.best is not None
        # float32 accumulation over 20 Adam epochs drifts by well under 1%.
        # The plan's acceptance gate is on executable orders (tick/step), not
        # on objective equality, so this tolerance is deliberately loose.
        assert f64.best.objective_value == pytest.approx(
            f32.best.objective_value, rel=1e-2
        )
        assert f64.best_epoch == f32.best_epoch

    def test_float32_does_not_produce_nonfinite_values(self) -> None:
        _, diagnostics = _optimizer(
            tensor_dtype="float32", adam_epochs=20, fixed_mc_samples=True
        ).optimize_with_diagnostics(_minimal_state())
        assert diagnostics.gradients["nonfinite_loss_count"] == 0
        assert diagnostics.gradients["nonfinite_grad_count"] == 0

    def test_float32_and_float64_produce_similar_ladders(self) -> None:
        state = _minimal_state()
        _, f64 = _optimizer(
            tensor_dtype="float64", adam_epochs=20, fixed_mc_samples=True
        ).optimize_with_diagnostics(state)
        _, f32 = _optimizer(
            tensor_dtype="float32", adam_epochs=20, fixed_mc_samples=True
        ).optimize_with_diagnostics(state)

        assert f64.action_best is not None and f32.action_best is not None
        bid64 = f64.action_best.effective_depth["bid"]
        bid32 = f32.action_best.effective_depth["bid"]
        assert len(bid64) == len(bid32)
        for a, b in zip(bid64, bid32):
            assert a["spread_quote"] == pytest.approx(b["spread_quote"], rel=1e-6)
            assert a["cumulative_qty"] == pytest.approx(b["cumulative_qty"], rel=1e-3)


class TestResolveTorchThreads:
    def test_env_override_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MM_TORCH_THREADS", "1")
        assert resolve_torch_threads(None) == 1

    def test_explicit_argument_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MM_TORCH_THREADS", "8")
        assert resolve_torch_threads(2) == 2

    def test_unset_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MM_TORCH_THREADS", raising=False)
        assert resolve_torch_threads(None) is None

    def test_invalid_values_return_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MM_TORCH_THREADS", "not-a-number")
        assert resolve_torch_threads(None) is None
        monkeypatch.setenv("MM_TORCH_THREADS", "0")
        assert resolve_torch_threads(None) is None
        monkeypatch.setenv("MM_TORCH_THREADS", "-2")
        assert resolve_torch_threads(None) is None

    def test_blank_env_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MM_TORCH_THREADS", "   ")
        assert resolve_torch_threads(None) is None

    def test_optimizer_pins_thread_pool(self) -> None:
        original = torch.get_num_threads()
        try:
            optimizer = _optimizer(torch_threads=1)
            assert optimizer.torch_threads == 1
            assert torch.get_num_threads() == 1
            # The optimizer must still work with a single thread.
            _, diagnostics = optimizer.optimize_with_diagnostics(_minimal_state())
            assert diagnostics.timing["optimize_ms"] > 0.0
        finally:
            torch.set_num_threads(original)
