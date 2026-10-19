from __future__ import annotations

import math
import pytest
import torch

from mm.mpc.market_order_flow_torch import TorchHurdleLogNormalModel
from mm.mpc.mpc_state import OrderbookRelative, State
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched, VenueOptimizerParams
from mm.mpc.reward_tensor import objective_entropic_return, terminal_objective


class _FixedPriceModel:
    def sample_residual(self, sample_shape, device=None, dtype=None):
        mod = torch
        z = mod.zeros(sample_shape, dtype=dtype or mod.float64, device=device)
        return z, z


class _FixedExecutionModel:
    maker_fee = 0.0


def _sample_state(inventory: float = 0.0) -> State:
    return State(
        mid=100.0,
        ob_rel=OrderbookRelative(
            bids=((1.0, 10.0), (2.0, 20.0), (3.0, 30.0)),
            asks=((1.0, 10.0), (2.0, 20.0), (3.0, 30.0)),
        ),
        buy_vol=0.0,
        sell_vol=0.0,
        inventory=inventory,
        cash=10_000.0,
        buy_vwap=99.0,
        sell_vwap=101.0,
    )


def test_stratified_scenarios_exact_weights() -> None:
    flow = TorchHurdleLogNormalModel(
        init_p0=0.85,
        init_mu=1.5,
        init_sigma=0.6,
    )
    strat = flow.stratified_scenarios(horizon=2, positive_nodes=3)
    assert strat.buy_vol.shape[0] == 16  # (1 + 3) * (1 + 3)
    assert strat.buy_vol.shape[1] == 2
    assert strat.sell_vol.shape == strat.buy_vol.shape
    assert strat.weights.shape == (16,)
    assert float(strat.weights.sum().item()) == pytest.approx(1.0, rel=1e-6)

    # Node 0 has zero buy and sell volume with weight p0_buy * p0_sell
    assert float(strat.buy_vol[0, 0].item()) == 0.0
    assert float(strat.sell_vol[0, 0].item()) == 0.0
    expected_w0 = 0.85 * 0.85
    assert float(strat.weights[0].item()) == pytest.approx(expected_w0, rel=1e-5)


def test_weighted_certainty_equivalent() -> None:
    returns = torch.tensor([0.0, 10.0, -5.0], dtype=torch.float64)
    weights = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    beta = 0.5
    ce = objective_entropic_return(returns, beta=beta, weights=weights)

    # Theoretical: min(R) - (1/beta) * ln(sum_i w_i * exp(-beta * (R_i - min(R))))
    r_min = -5.0
    shifted = returns - r_min
    expected = r_min - (1.0 / beta) * math.log(
        float(torch.sum(weights * torch.exp(-beta * shifted)).item())
    )
    assert float(ce.item()) == pytest.approx(expected, rel=1e-6)


def test_discrete_trust_region_optimizer_monotonic_progress() -> None:
    flow = TorchHurdleLogNormalModel(
        init_p0=0.85,
        init_mu=2.0,
        init_sigma=0.8,
    )
    mpc = MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        flow,
        _FixedExecutionModel(),
        n_components=1,
        horizon=1,
        spread_max=10.0,
        spread_step=1.0,
        device="cpu",
        optimizer_mode="discrete_trust_region",
        flow_scenario_mode="weighted_stratified",
        stratified_positive_nodes=2,
        trust_region_iterations=3,
        trust_region_radius_ticks=2.0,
        soft_fill_queue_model="pro_rata",
    )
    state = _sample_state(inventory=0.0)
    action, diagnostics = mpc.optimize_with_diagnostics(state)

    payload = diagnostics.to_payload()
    initial_obj = payload["objective"]["initial"]["value"]
    best_obj = payload["objective"]["best"]["value"]

    assert initial_obj is not None
    assert best_obj is not None
    # Monotonic progress guarantee: best is never worse than initial
    assert float(best_obj) >= float(initial_obj) - 1e-9

    top_fractions = payload["objective"]["contribution_fractions"]
    assert sum(float(v) for v in top_fractions.values()) == pytest.approx(1.0, rel=1e-5)
    assert payload["timing"]["trust_region_ms"] >= 0.0


def test_discrete_trust_region_adapts_to_inventory_imbalance() -> None:
    flow = TorchHurdleLogNormalModel(
        init_p0=0.80,
        init_mu=2.5,
        init_sigma=0.5,
    )
    mpc = MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        flow,
        _FixedExecutionModel(),
        n_components=1,
        horizon=1,
        spread_max=10.0,
        spread_step=1.0,
        device="cpu",
        optimizer_mode="discrete_trust_region",
        flow_scenario_mode="weighted_stratified",
        stratified_positive_nodes=2,
        trust_region_iterations=3,
        soft_fill_queue_model="pro_rata",
    )
    # Long inventory (+50 units)
    state_long = _sample_state(inventory=50.0)
    action_long, diag_long = mpc.optimize_with_diagnostics(state_long)
    pay_long = diag_long.to_payload()

    # Short inventory (-50 units)
    state_short = _sample_state(inventory=-50.0)
    action_short, diag_short = mpc.optimize_with_diagnostics(state_short)
    pay_short = diag_short.to_payload()

    # When short inventory, buying (bid) should be tightened relative to asking (ask) to cover the short
    assert action_short.bid_x0[0] < action_short.ask_x0[0]


def _build_adam_mpc(*, queue_model: str, adam_eps: float) -> MonteCarloMPCTensorBatched:
    flow = TorchHurdleLogNormalModel(init_p0=0.85, init_mu=2.0, init_sigma=0.8)
    local_opt = VenueOptimizerParams(
        n_components=1,
        cumulative_qty_floor_at_spread_max=1.0,
        soft_fill_queue_model=queue_model,
    )
    return MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        flow,
        _FixedExecutionModel(),
        n_components=1,
        n_samples=16,
        horizon=1,
        spread_max=10.0,
        spread_step=1.0,
        adam_epochs=20,
        adam_lr=1.0,
        adam_eps=adam_eps,
        fixed_mc_samples=True,
        device="cpu",
        optimizer_mode="adam",
        flow_scenario_mode="weighted_stratified",
        stratified_positive_nodes=2,
        local_optimizer=local_opt,
    )


def _deep_book_state(inventory: float = 20.0) -> State:
    """A book deep enough that a market order never reaches our quotes.

    This is the production condition on an illiquid venue: the resting book
    ahead of our quote absorbs the whole market order, so back-of-queue
    matching leaves us with a zero fill and a zero gradient.
    """
    return State(
        mid=100.0,
        ob_rel=OrderbookRelative(
            bids=((1.0, 5_000.0), (2.0, 5_000.0), (3.0, 5_000.0)),
            asks=((1.0, 5_000.0), (2.0, 5_000.0), (3.0, 5_000.0)),
        ),
        buy_vol=0.0,
        sell_vol=0.0,
        inventory=inventory,
        cash=10_000.0,
        buy_vwap=99.0,
        sell_vwap=101.0,
    )


def test_adam_gradient_is_nonzero_with_pro_rata_queue_model() -> None:
    """Regression: ``back_of_queue`` makes the fill identically zero.

    Under back-of-queue matching our quote only fills once the incoming
    market order has consumed the entire resting book ahead of it, so
    ``d fill / d q_ours`` is exactly 0 and every epoch has a zero gradient.
    ``pro_rata`` gives a strictly positive, continuous share.
    """
    mpc = _build_adam_mpc(queue_model="pro_rata", adam_eps=1e-12)
    _, diag = mpc.optimize_with_diagnostics(_deep_book_state())
    gradients = diag.to_payload()["gradients"]

    assert gradients["grad_norm_epoch_count"] > 0
    assert gradients["grad_norm_zero_epochs"] == 0
    assert gradients["grad_norm_max"] > 0.0
    assert gradients["nonfinite_grad_count"] == 0


def test_back_of_queue_queue_model_yields_identically_zero_gradient() -> None:
    mpc = _build_adam_mpc(queue_model="back_of_queue", adam_eps=1e-12)
    _, diag = mpc.optimize_with_diagnostics(_deep_book_state())
    gradients = diag.to_payload()["gradients"]

    assert gradients["grad_norm_epoch_count"] > 0
    assert gradients["grad_norm_zero_epochs"] == gradients["grad_norm_epoch_count"]
    assert gradients["grad_norm_max"] == 0.0


def test_adam_eps_is_scaled_to_the_objective_gradient_scale() -> None:
    """Regression: PyTorch's default ``eps=1e-8`` attenuates every step.

    The rollout objective is a *relative* wealth change (``dW / W_0``), so
    its gradient magnitude is ~1e-8..1e-10. With ``eps=1e-8`` the Adam
    denominator ``sqrt(v_hat) + eps`` is dominated by ``eps`` and each step
    is attenuated by ~100x, which is indistinguishable from a dead optimizer.
    """
    mpc = _build_adam_mpc(queue_model="pro_rata", adam_eps=1e-12)
    _, diag = mpc.optimize_with_diagnostics(_deep_book_state())
    payload = diag.to_payload()

    assert mpc.adam_eps == 1e-12
    assert payload["gradients"]["grad_norm_max"] > 0.0
    # The optimizer must actually be constructed with the configured eps.
    for group in mpc._optimizer.param_groups:
        assert group["eps"] == 1e-12


def test_adam_eps_default_is_not_the_pytorch_default() -> None:
    """The engine default must already be scaled; 1e-8 would re-break it."""
    default_mpc = MonteCarloMPCTensorBatched(
        _FixedPriceModel(),
        TorchHurdleLogNormalModel(init_p0=0.85),
        _FixedExecutionModel(),
        n_components=1,
        n_samples=8,
        horizon=1,
        spread_max=10.0,
        spread_step=1.0,
        device="cpu",
    )
    assert default_mpc.adam_eps < 1e-8
