"""Run one real MPC decide with the profiler attached.

The unit tests prove the hook is a no-op when disabled and emits a valid
trace when enabled. This exercises it against the actual optimizer to confirm
the wrapper sits on the hot path (not just that it compiles) and that the
emitted trace contains attributable operators.

Run: OMP_NUM_THREADS=2 MM_PROFILE_DECIDE=1 python3 scripts/verify_decide_profiler.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mm.mpc.l_param import ReservationSkewParams
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched, VenueOptimizerParams
from mm.mpc.objective_terms import ViabilityParams


class _FixedPriceModel:
    def sample_residual(self, sample_shape, device=None, dtype=None):
        z = torch.zeros(sample_shape, dtype=dtype or torch.float64, device=device)
        return z, z


class _FixedFlowModel:
    def sample(self, sample_shape, device=None, dtype=None):
        dt = dtype or torch.float64
        return (
            torch.full(sample_shape, 2.0, dtype=dt, device=device),
            torch.full(sample_shape, 1.0, dtype=dt, device=device),
        )


class _FixedExecutionModel:
    maker_fee = 0.0


def _minimal_state():
    from mm.mpc.mpc_state import OrderbookRelative, State

    return State(
        mid=265_000.0,
        ob_rel=OrderbookRelative(
            bids=tuple((float(i), 50.0) for i in range(1, 6)),
            asks=tuple((float(i), 50.0) for i in range(1, 6)),
        ),
        buy_vol=2.0,
        sell_vol=1.0,
        inventory=3_000.0,
        cash=200_000_000.0,
        buy_vwap=265_000.0,
        sell_vwap=265_000.0,
    )


def main() -> None:
    from mm.mpc.decide_profiler import profiling_enabled

    trace = Path(os.environ.get("MM_PROFILE_DECIDE_PATH", "/tmp/decide_trace.json"))
    os.environ["MM_PROFILE_DECIDE"] = str(trace)
    print(f"profiling_enabled (pre-import path): {profiling_enabled()}")

    local_opt = VenueOptimizerParams(
        n_components=1,
        reservation_skew=ReservationSkewParams(max_abs_skew=0.002),
        viability=ViabilityParams(min_viable_quote=50_000.0, recovery_fraction=0.1),
        cumulative_qty_floor_at_spread_max=1.0,
        cumulative_qty_cap_at_spread_max=10.0,
    )
    mpc = MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        _FixedFlowModel(),
        _FixedExecutionModel(),
        n_components=1,
        n_samples=64,
        horizon=2,
        spread_max=10.0,
        spread_step=5.0,
        adam_epochs=20,
        fixed_mc_samples=True,
        device="cpu",
        local_optimizer=local_opt,
    )

    print("=== decide WITH profiler ===")
    t0 = time.perf_counter()
    mpc.optimize_with_diagnostics(_minimal_state())
    with_prof = (time.perf_counter() - t0) * 1000.0
    print(f"  wall: {with_prof:.0f} ms")

    print("=== decide WITHOUT profiler (flag auto-cleared) ===")
    t0 = time.perf_counter()
    mpc.optimize_with_diagnostics(_minimal_state())
    without_prof = (time.perf_counter() - t0) * 1000.0
    print(f"  wall: {without_prof:.0f} ms")

    print(f"\ntrace exists: {trace.exists()} ({trace.stat().st_size} bytes)" if trace.exists() else "trace MISSING")
    summary = trace.with_suffix("")
    summary = Path(str(trace).rsplit(".", 1)[0] + "_summary.txt")
    if summary.exists():
        print("\n=== top operators ===")
        for line in summary.read_text().splitlines()[:18]:
            print(" ", line)
    print(f"\nprofiler overhead: {with_prof / max(without_prof, 1e-9):.2f}x")
    print("NOTE: a ratio near 1.0 means the hook is on the hot path but the")
    print("      profiler itself is cheap; a large ratio means profiling cost")
    print("      dominates and the trace should be read for kernel time only.")


if __name__ == "__main__":
    main()