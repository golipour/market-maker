"""Tests for optimizer stage-timing instrumentation.

Two properties are covered:

1. ``OptimizerStageTimer`` accumulates and renders per-stage milliseconds
   without double counting nested stages.
2. ``MonteCarloMPCTensorBatched`` actually populates the ``*_ms`` timing keys
   that the persistence layer reads into ``mm_optimizer_facts``.

The second point matters because ``optimize_ms`` was historically always null:
the optimizer emitted ``total_sec`` (seconds) while the fact extractors looked
for ``optimize_ms``/``optimizer_ms``/``total_ms``. Every latency question then
required scanning ``mm_events.payload`` JSON.
"""

from __future__ import annotations

import pytest
import torch

from mm.mpc.mpc_state import OrderbookRelative, State
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched
from mm.mpc.optimizer_timing import (
    NESTED_STAGES,
    STAGE_KEYS,
    OptimizerStageTimer,
    stage_timing_payload,
)
from mm.persistence.parquet_writer import _controller_columns  # type: ignore[attr-defined]
from mm.persistence.postgres import _optimizer_fact_row  # type: ignore[attr-defined]


class _FixedFlowModel:
    def sample(self, sample_shape, device=None, dtype=None):
        mod = torch
        dt = dtype or mod.float64
        return (
            mod.full(sample_shape, 2.0, dtype=dt, device=device),
            mod.full(sample_shape, 1.0, dtype=dt, device=device),
        )


class _FixedPriceModel:
    def sample_residual(self, sample_shape, device=None, dtype=None):
        mod = torch
        z = mod.zeros(sample_shape, dtype=dtype or mod.float64, device=device)
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


class TestOptimizerStageTimer:
    def test_empty_timer_reports_zero_total(self) -> None:
        timer = OptimizerStageTimer()
        payload = timer.as_payload()
        assert payload["optimize_ms"] == 0.0
        assert payload["epochs_run"] == 0

    def test_record_accumulates_repeated_stages(self) -> None:
        timer = OptimizerStageTimer()
        timer.record("forward", 10.0)
        timer.record("forward", 5.0)
        assert timer.totals_ms["forward"] == pytest.approx(15.0)
        assert timer.counts["forward"] == 2

    def test_non_positive_measurements_are_ignored(self) -> None:
        timer = OptimizerStageTimer()
        timer.record("forward", 0.0)
        timer.record("forward", -3.0)
        assert "forward" not in timer.totals_ms

    def test_nested_stages_excluded_from_total(self) -> None:
        timer = OptimizerStageTimer()
        timer.record("forward", 100.0)
        timer.record("objective", 40.0)
        assert "objective" in NESTED_STAGES
        # objective is a breakdown of forward, so it must not inflate the total.
        assert timer.total_ms() == pytest.approx(100.0)

    def test_total_sums_all_non_nested_stages(self) -> None:
        timer = OptimizerStageTimer()
        for stage in ("setup", "sample", "forward", "backward", "adam", "diagnostics"):
            timer.record(stage, 10.0)
        assert timer.total_ms() == pytest.approx(60.0)

    def test_payload_uses_ms_suffix_keys(self) -> None:
        timer = OptimizerStageTimer()
        for stage in STAGE_KEYS:
            timer.record(stage, 1.0)
        payload = timer.as_payload()
        for stage in STAGE_KEYS:
            assert f"{stage}_ms" in payload

    def test_measure_context_manager_records_even_on_exception(self) -> None:
        timer = OptimizerStageTimer()
        with pytest.raises(RuntimeError):
            with timer.measure("setup"):
                raise RuntimeError("boom")
        assert timer.totals_ms["setup"] > 0.0

    def test_bump_epochs_never_goes_negative(self) -> None:
        timer = OptimizerStageTimer()
        timer.bump_epochs(3)
        timer.bump_epochs(-10)
        assert timer.epochs_run == 0

    def test_stage_timing_payload_preserves_legacy_keys(self) -> None:
        timer = OptimizerStageTimer()
        timer.record("forward", 20.0)
        payload = stage_timing_payload(
            timer,
            legacy={"total_sec": 1.25, "train_sec": 1.0},
        )
        assert payload["total_sec"] == 1.25
        assert payload["train_sec"] == 1.0
        assert payload["forward_ms"] == pytest.approx(20.0)
        assert payload["optimize_ms"] == pytest.approx(20.0)


class TestOptimizerPopulatesStageTiming:
    def test_optimizer_emits_stage_milliseconds(self) -> None:
        _, diagnostics = _optimizer().optimize_with_diagnostics(_minimal_state())
        timing = diagnostics.timing

        for stage in ("setup_ms", "forward_ms", "backward_ms", "adam_ms", "diagnostics_ms"):
            assert stage in timing, f"missing {stage}"
            assert timing[stage] > 0.0

        assert timing["optimize_ms"] > 0.0
        assert timing["epochs_run"] == 3

    def test_legacy_second_keys_are_preserved(self) -> None:
        _, diagnostics = _optimizer().optimize_with_diagnostics(_minimal_state())
        timing = diagnostics.timing
        assert timing["total_sec"] > 0.0
        assert timing["train_sec"] > 0.0
        assert "sample_draw_sec" in timing

    def test_stage_times_do_not_exceed_total(self) -> None:
        _, diagnostics = _optimizer().optimize_with_diagnostics(_minimal_state())
        timing = diagnostics.timing
        total = timing["optimize_ms"]
        non_nested = sum(
            v for k, v in timing.items()
            if k.endswith("_ms") and k != "optimize_ms" and k[:-3] not in NESTED_STAGES
        )
        # The total is the sum of the non-nested stages, so it cannot be less.
        assert total == pytest.approx(non_nested, rel=1e-9)

    def test_epochs_run_reflects_actual_epochs(self) -> None:
        _, diagnostics = _optimizer(adam_epochs=5).optimize_with_diagnostics(_minimal_state())
        assert diagnostics.timing["epochs_run"] == 5

    def test_cancelled_run_still_reports_timing(self) -> None:
        import threading

        optimizer = _optimizer(adam_epochs=4)
        event = threading.Event()
        event.set()
        optimizer.set_cancel_event(event)
        _, diagnostics = optimizer.optimize_with_diagnostics(_minimal_state())

        assert optimizer.was_cancelled() is True
        assert diagnostics.timing["epochs_run"] < 4
        assert diagnostics.timing["optimize_ms"] > 0.0


class TestSampledEpochDiagnostics:
    """``optimizer_diagnostics_every`` must not change published results."""

    @staticmethod
    def _run(every: int) -> "OptimizerRunDiagnostics":
        _, diagnostics = _optimizer(
            adam_epochs=20, optimizer_diagnostics_every=every
        ).optimize_with_diagnostics(_minimal_state())
        return diagnostics

    def test_sampling_preserves_initial_best_and_final(self) -> None:
        baseline = self._run(1)
        for every in (2, 3, 5, 20):
            sampled = self._run(every)
            assert sampled.initial is not None
            assert sampled.best is not None
            assert sampled.final is not None
            assert sampled.initial.objective_value == pytest.approx(
                baseline.initial.objective_value, rel=1e-9
            )
            assert sampled.best.objective_value == pytest.approx(
                baseline.best.objective_value, rel=1e-9
            )
            assert sampled.best_epoch == baseline.best_epoch

    def test_sampling_preserves_best_action(self) -> None:
        baseline = self._run(1)
        sampled = self._run(7)
        assert baseline.action_best is not None and sampled.action_best is not None
        assert sampled.action_best.effective_depth["bid"] == (
            baseline.action_best.effective_depth["bid"]
        )
        assert sampled.action_best.effective_depth["ask"] == (
            baseline.action_best.effective_depth["ask"]
        )

    def test_sampling_does_not_increase_diagnostics_cost(self) -> None:
        # Skipping per-epoch scalar extraction must not cost more time.
        every_epoch = min(self._run(1).timing["diagnostics_ms"] for _ in range(1))
        sampled = min(self._run(20).timing["diagnostics_ms"] for _ in range(1))
        assert sampled <= every_epoch + 5.0

    def test_every_epoch_sampling_is_always_materialized(self) -> None:
        # Sampling must never disable metrics entirely.
        for every in (1, 5, 100):
            diagnostics = self._run(every)
            assert diagnostics.initial is not None
            assert diagnostics.final is not None


class TestEpochNeedsMetrics:
    def test_always_records_first_and_last(self) -> None:
        from mm.mpc.mpc_tensor_batched import _epoch_needs_metrics

        assert _epoch_needs_metrics(0, 9, 5) is True
        assert _epoch_needs_metrics(9, 9, 5) is True

    def test_records_sampled_epochs(self) -> None:
        from mm.mpc.mpc_tensor_batched import _epoch_needs_metrics

        assert _epoch_needs_metrics(5, 9, 5) is True
        assert _epoch_needs_metrics(4, 9, 5) is False

    def test_every_one_keeps_legacy_behavior(self) -> None:
        from mm.mpc.mpc_tensor_batched import _epoch_needs_metrics

        for epoch in range(10):
            assert _epoch_needs_metrics(epoch, 9, 1) is True
            assert _epoch_needs_metrics(epoch, 9, 0) is True
            assert _epoch_needs_metrics(epoch, 9, -5) is True


class TestTimingReachesFactTables:
    """The point of the instrumentation: facts must be populated, not null."""

    @staticmethod
    def _envelope(timing: dict) -> "EventEnvelope":
        from mm.core.events import envelope_event

        return envelope_event(
            "strategy:diagnostics:mpc-controller",
            {
                "type": "ControllerDiagnostics",
                "actor_id": "mpc-controller",
                "exchange": "bitwana",
                "symbol": "BTC_IRT",
                "optimizer": {
                    "config": {"kind": "monte_carlo_mpc_tensor_batched"},
                    "timing": timing,
                },
            },
        )

    def test_parquet_optimizer_fact_columns_include_stage_timings(self) -> None:
        envelope = self._envelope(
            {
                "optimize_ms": 1234.5,
                "setup_ms": 10.0,
                "sample_ms": 20.0,
                "forward_ms": 900.0,
                "objective_ms": 30.0,
                "backward_ms": 200.0,
                "adam_ms": 5.0,
                "diagnostics_ms": 1.0,
                "epochs_run": 10,
            }
        )
        columns = _controller_columns(envelope, envelope.payload)
        assert columns["optimize_ms"] == pytest.approx(1234.5)
        assert columns["forward_ms"] == pytest.approx(900.0)
        assert columns["backward_ms"] == pytest.approx(200.0)
        assert columns["epochs_run"] == 10

    def test_postgres_optimizer_fact_row_appends_stage_timings(self) -> None:
        envelope = self._envelope(
            {
                "optimize_ms": 1234.5,
                "forward_ms": 900.0,
                "epochs_run": 10,
            }
        )
        row = _optimizer_fact_row(envelope)
        assert row is not None
        # The stage columns are appended at the end of the row so the existing
        # positional contract used by other tests stays valid. Order matches
        # the INSERT list: setup, sample, forward, objective, backward, adam,
        # diagnostics, epochs_run.
        assert row[-8] is None  # setup_ms
        assert row[-6] == pytest.approx(900.0)  # forward_ms
        assert row[-1] == 10  # epochs_run

    def test_postgres_row_appends_all_eight_stage_columns(self) -> None:
        envelope = self._envelope(
            {
                "setup_ms": 1.0,
                "sample_ms": 2.0,
                "forward_ms": 3.0,
                "objective_ms": 4.0,
                "backward_ms": 5.0,
                "adam_ms": 6.0,
                "diagnostics_ms": 7.0,
                "epochs_run": 8,
            }
        )
        row = _optimizer_fact_row(envelope)
        assert row is not None
        assert row[-8:] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8)

    def test_missing_stage_timings_stay_null(self) -> None:
        envelope = self._envelope({"controller_total_ms": 50.0})
        columns = _controller_columns(envelope, envelope.payload)
        assert columns["optimize_ms"] is None
        assert columns["forward_ms"] is None
        assert columns["epochs_run"] is None
