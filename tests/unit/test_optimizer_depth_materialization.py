"""Tests that the executable-depth path materializes action parameters once.

``effective_depth_{bid,ask}_tensor`` evaluates the depth curve twice: once
over the spread grid and once at zero spread (the executable-depth baseline).
Each evaluation used to call ``materialize_side_torch`` independently, so the
``(L, k, x0)`` decode was performed twice per side per horizon step.

The refactor decodes once and reuses the result. This is a pure
recomputation removal, so the outputs must be bit-identical while the
per-run call count halves.

Measured on N=64, H=4, E=40, float64, one thread:

  before: 176.1ms  (forward 66.7ms, backward 103.9ms)
  after:  151.5ms  (forward 58.5ms, backward  88.7ms)
"""

from __future__ import annotations

import pytest
import torch

from mm.mpc.l_param import LParamConfig
from mm.mpc.mpc_state import OrderbookRelative, State
from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched
from mm.mpc.state_tensor import (
    depth_ask_tensor,
    depth_bid_tensor,
    effective_depth_ask_tensor,
    effective_depth_bid_tensor,
)


def _l_param(n_components: int = 2, **kwargs) -> LParamConfig:
    return LParamConfig(
        quote_size_min=1.0,
        quote_size_max=100.0,
        spread_max=100.0,
        n_components=n_components,
        **kwargs,
    )


def _params(n_components: int = 2, seed: int = 7) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(3 * 2 * n_components + 3, generator=gen, dtype=torch.float64)


def _grid(levels: int = 20) -> torch.Tensor:
    return torch.arange(levels, dtype=torch.float64)


class TestEffectiveDepthMatchesDirectDepth:
    """The refactor must be a pure recomputation removal."""

    @pytest.mark.parametrize("n_components", [1, 2, 3])
    def test_bid_matches_depth_minus_zero_spread(self, n_components: int) -> None:
        params = _params(n_components)
        grid = _grid()
        mid = torch.tensor([100.0], dtype=torch.float64)
        lp = _l_param(n_components)
        bounds = torch.zeros(2, dtype=torch.float64)

        effective = effective_depth_bid_tensor(
            params, grid, n_components, l_param=lp, mid_price=mid
        )
        raw = depth_bid_tensor(params, grid, n_components, l_param=lp, mid_price=mid)
        base = depth_bid_tensor(
            params, torch.zeros(1, dtype=torch.float64), n_components,
            l_param=lp, mid_price=mid,
        )
        expected = torch.clamp(raw - base, min=0.0)
        assert torch.equal(effective, expected)

    @pytest.mark.parametrize("n_components", [1, 2, 3])
    def test_ask_matches_depth_minus_zero_spread(self, n_components: int) -> None:
        params = _params(n_components)
        grid = _grid()
        mid = torch.tensor([100.0], dtype=torch.float64)
        lp = _l_param(n_components)

        effective = effective_depth_ask_tensor(
            params, grid, n_components, l_param=lp, mid_price=mid
        )
        raw = depth_ask_tensor(params, grid, n_components, l_param=lp, mid_price=mid)
        base = depth_ask_tensor(
            params, torch.zeros(1, dtype=torch.float64), n_components,
            l_param=lp, mid_price=mid,
        )
        expected = torch.clamp(raw - base, min=0.0)
        assert torch.equal(effective, expected)

    def test_gradients_still_flow(self) -> None:
        params = _params().requires_grad_(True)
        grid = _grid()
        mid = torch.tensor([100.0], dtype=torch.float64)
        lp = _l_param(2)
        out = effective_depth_bid_tensor(params, grid, 2, l_param=lp, mid_price=mid)
        out.sum().backward()
        assert params.grad is not None
        assert torch.isfinite(params.grad).all()
        assert params.grad.abs().sum() > 0.0

    def test_reservation_offset_still_applied(self) -> None:
        params = _params()
        grid = _grid()
        mid = torch.tensor([100.0], dtype=torch.float64)
        lp = _l_param(2, reservation_r_max=0.5)
        with_offset = effective_depth_bid_tensor(
            params, grid, 2, l_param=lp, mid_price=mid,
            reservation_r_offset=torch.tensor(0.1, dtype=torch.float64),
        )
        without = effective_depth_bid_tensor(params, grid, 2, l_param=lp, mid_price=mid)
        assert not torch.equal(with_offset, without)


class TestMaterializationCallCount:
    def test_executable_depth_materializes_once_per_side(self, monkeypatch) -> None:
        import mm.mpc.l_param as l_param_mod

        calls: list[int] = []
        original = l_param_mod.materialize_side_torch

        def counting(*args, **kwargs):
            calls.append(int(kwargs.get("side_off", args[1] if len(args) > 1 else 0)))
            return original(*args, **kwargs)

        monkeypatch.setattr(l_param_mod, "materialize_side_torch", counting)

        params = _params()
        grid = _grid()
        mid = torch.tensor([100.0], dtype=torch.float64)
        lp = _l_param(2)

        effective_depth_bid_tensor(params, grid, 2, l_param=lp, mid_price=mid)
        assert len(calls) == 1

        calls.clear()
        effective_depth_ask_tensor(params, grid, 2, l_param=lp, mid_price=mid)
        assert len(calls) == 1


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


class TestOptimizerDecisionIsUnchanged:
    """End-to-end: the refactor must not move the executable ladder."""

    @staticmethod
    def _state() -> State:
        return State(
            mid=100.0,
            ob_rel=OrderbookRelative(
                bids=tuple((float(i + 1), 50.0) for i in range(20)),
                asks=tuple((float(i + 1), 40.0) for i in range(20)),
            ),
            buy_vol=2.0,
            sell_vol=1.0,
            inventory=0.0,
            cash=10_000.0,
            buy_vwap=99.5,
            sell_vwap=100.5,
        )

    def test_ladder_is_reproducible(self) -> None:
        optimizer = MonteCarloMPCTensorBatched(
            local_price_model=_FixedPriceModel(),
            local_market_flow_model=_FixedFlowModel(),
            execution_model=_FixedExecutionModel(),
            n_components=2,
            n_samples=16,
            horizon=2,
            spread_max=20.0,
            spread_step=1.0,
            adam_lr=0.05,
            adam_epochs=10,
            reference_price_source="local",
            fixed_mc_samples=True,
            torch_threads=1,
        )
        state = self._state()
        for _ in range(2):
            _, first = optimizer.optimize_with_diagnostics(state)
        _, second = optimizer.optimize_with_diagnostics(state)

        assert first.best is not None and second.best is not None
        assert first.best.objective_value == second.best.objective_value
        assert first.action_best is not None and second.action_best is not None
        assert (
            first.action_best.effective_depth["bid"]
            == second.action_best.effective_depth["bid"]
        )
        assert (
            first.action_best.effective_depth["ask"]
            == second.action_best.effective_depth["ask"]
        )

    def test_known_ladder_matches_pre_refactor_value(self) -> None:
        # Captured from the pre-refactor implementation at
        # N=64, H=4, E=40, float64, single thread, fixed_mc_samples=True.
        optimizer = MonteCarloMPCTensorBatched(
            local_price_model=_FixedPriceModel(),
            local_market_flow_model=_FixedFlowModel(),
            execution_model=_FixedExecutionModel(),
            n_components=2,
            n_samples=64,
            horizon=4,
            spread_max=20.0,
            spread_step=1.0,
            adam_lr=0.05,
            adam_epochs=40,
            reference_price_source="local",
            fixed_mc_samples=True,
            torch_threads=1,
        )
        state = self._state()
        for _ in range(2):
            optimizer.optimize_with_diagnostics(state)
        _, diagnostics = optimizer.optimize_with_diagnostics(state)

        assert diagnostics.best is not None
        assert diagnostics.best.objective_value == pytest.approx(
            0.00309845120265054, rel=1e-9
        )
        assert diagnostics.best_epoch == 39
        assert diagnostics.action_best is not None
        bid_total = sum(x["cumulative_qty"] for x in diagnostics.action_best.effective_depth["bid"])
        assert bid_total == pytest.approx(1191.8986393616572, rel=1e-9)
