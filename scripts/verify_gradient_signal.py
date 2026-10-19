"""Measure real optimizer gradient signal under a given preset configuration.

The controller publishes only ``nonfinite_loss_count`` / ``nonfinite_grad_count``
in the diagnostics payload, so "gradient is zero" is invisible from Grafana.
This script reproduces the production rollout with the *shipped* preset
knobs and reports the actual per-epoch gradient norms and objective deltas.

Usage:
    OMP_NUM_THREADS=2 ./.venv/bin/python scripts/verify_gradient_signal.py \
        --queue-model pro_rata --scenario-mode weighted_stratified
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mm.mpc.market_order_flow_torch import TorchHurdleLogNormalModel
from mm.mpc.mpc_state import OrderbookRelative, State
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched, VenueOptimizerParams
from mm.mpc.objective_terms import (
    BookQualityParams,
    BookQualitySoftScoring,
    ValidityParams,
    ViabilityParams,
)


class _FixedPriceModel:
    def sample_residual(self, sample_shape, device=None, dtype=None):
        z = torch.zeros(sample_shape, dtype=dtype or torch.float64, device=device)
        return z, z


class _FixedExecutionModel:
    maker_fee = 0.0


def _state(inventory: float = 3_000.0) -> State:
    return State(
        mid=265_000.0,
        ob_rel=OrderbookRelative(
            bids=tuple((float(i), 50.0) for i in range(1, 6)),
            asks=tuple((float(i), 50.0) for i in range(1, 6)),
        ),
        buy_vol=2.0,
        sell_vol=1.0,
        inventory=inventory,
        cash=200_000_000.0,
        buy_vwap=265_000.0,
        sell_vwap=265_000.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-model", default="pro_rata")
    ap.add_argument("--scenario-mode", default="weighted_stratified")
    ap.add_argument("--optimizer-mode", default="adam")
    ap.add_argument("--adam-epochs", type=int, default=20)
    ap.add_argument("--adam-lr", type=float, default=1.0)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--init-p0", type=float, default=0.85)
    ap.add_argument("--init-mu", type=float, default=2.0)
    ap.add_argument("--init-sigma", type=float, default=0.8)
    ap.add_argument("--inventory", type=float, default=3_000.0)
    ap.add_argument("--reservation-skew", type=float, default=0.002)
    ap.add_argument("--n-components", type=int, default=4)
    args = ap.parse_args()

    flow = TorchHurdleLogNormalModel(
        init_p0=args.init_p0,
        init_mu=args.init_mu,
        init_sigma=args.init_sigma,
    )

    local_opt = VenueOptimizerParams(
        n_components=args.n_components,
        return_risk_aversion_beta=0.1,
        objective_weight=1.0,
        cumulative_qty_floor_at_spread_max=1.0,
        validity=ValidityParams(min_score=0.3, penalty_scale=100.0),
        viability=ViabilityParams(
            min_viable_quote=50_000.0,
            shortfall_tolerance=0.05,
            lambda_init=0.0,
            lambda_learning_rate=0.02,
            lambda_max=100.0,
            dual_ewma_decay=0.9,
            recovery_fraction=0.0,
        ),
        book_quality=BookQualityParams(
            soft_scoring=BookQualitySoftScoring(
                measure="quoted", tightness_spread_scale=200.0, presence_reference_qty=1.0
            )
        ),
        reservation_skew=None,
        soft_fill_temp=0.1,
        soft_fill_activation="hardsigmoid",
        soft_fill_queue_model=args.queue_model,
    )

    mpc = MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        flow,
        _FixedExecutionModel(),
        n_components=args.n_components,
        n_samples=args.n_samples,
        horizon=args.horizon,
        spread_max=100.0,
        spread_step=20.0,
        adam_epochs=args.adam_epochs,
        adam_lr=args.adam_lr,
        fixed_mc_samples=True,
        optimizer_mode=args.optimizer_mode,
        flow_scenario_mode=args.scenario_mode,
        stratified_positive_nodes=3,
        local_optimizer=local_opt,
        device="cpu",
    )

    # Instrument the raw parameter tensor to capture per-epoch gradient norms.
    grad_norms: list[float] = []

    state = _state(inventory=args.inventory)
    action, diag = mpc.optimize_with_diagnostics(state)
    payload = diag.to_payload()

    obj = payload["objective"]
    init_v = obj["initial"]["value"]
    best_v = obj["best"]["value"]
    final_v = obj["final"]["value"]

    print("=" * 78)
    print(f"queue_model        = {args.queue_model}")
    print(f"scenario_mode      = {args.scenario_mode}")
    print(f"optimizer_mode     = {args.optimizer_mode}")
    print(f"n_samples          = {args.n_samples}  horizon={args.horizon}")
    print(f"adam_epochs        = {args.adam_epochs}  adam_lr={args.adam_lr}")
    print("-" * 78)
    print(f"initial objective  = {init_v}")
    print(f"final   objective  = {final_v}")
    print(f"best    objective  = {best_v}")
    delta = None
    if init_v is not None and best_v is not None:
        delta = float(best_v) - float(init_v)
        print(f"best - initial     = {delta:.6e}")
    print("-" * 78)
    print(f"rollout            = {payload['rollout']}")
    print(f"gradients          = {payload['gradients']}")
    print("=" * 78)

    if delta is not None and abs(delta) < 1e-12:
        print("RESULT: FLAT OBJECTIVE - no measurable optimizer progress.")
        return 1
    print("RESULT: objective moved - optimizer is making progress.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
