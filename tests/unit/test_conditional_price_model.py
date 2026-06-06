from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mm.core.models.market import PriceLevel
from mm.mpc.price_torch import TorchConditionalMechanicalMixturePriceModel


@dataclass(frozen=True, slots=True)
class _Book:
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]
    mid: float


def test_conditional_price_model_uses_sigmoid_flow_gate_in_rollout() -> None:
    torch = pytest.importorskip("torch")

    model = TorchConditionalMechanicalMixturePriceModel(
        reference_vwap_depth=100.0,
        point_mode="mean",
    )
    model._gate_beta = np.array([0.0, 1.0], dtype=np.float64)
    model._market_beta["mid"] = np.array([10.0, 0.0], dtype=np.float64)
    model._refresh_mean["mid"] = -10.0

    prev_mid = torch.tensor([[100.0, 100.0]], dtype=torch.float64)
    prev_spread = torch.full_like(prev_mid, 2.0)
    prev_shape = torch.full_like(prev_mid, 3.0)
    mech_mid = prev_mid
    mech_spread = prev_spread
    total_flow = torch.tensor([[0.0, 10.0]], dtype=torch.float64)
    depth = torch.full_like(prev_mid, 100.0)

    buy_vwap, sell_vwap = model.sample_dynamic_vwap(
        prev_mid=prev_mid,
        prev_spread=prev_spread,
        prev_buy_shape=prev_shape,
        prev_sell_shape=prev_shape,
        mech_mid=mech_mid,
        mech_spread=mech_spread,
        mech_buy_shape=prev_shape,
        mech_sell_shape=prev_shape,
        total_flow=total_flow,
        buy_depth=depth,
        sell_depth=depth,
    )

    np.testing.assert_allclose(
        buy_vwap.cpu().numpy(),
        np.array([[104.0, 112.33333333333333]]),
    )
    np.testing.assert_allclose(
        sell_vwap.cpu().numpy(),
        np.array([[96.0, 104.33333333333333]]),
    )


def test_conditional_price_model_updates_sigmoid_gate_with_decay() -> None:
    book = _Book(
        bids=(PriceLevel(99.0, 10.0),),
        asks=(PriceLevel(101.0, 10.0),),
        mid=100.0,
    )
    fast = TorchConditionalMechanicalMixturePriceModel(
        decay=0.5,
        reference_vwap_depth=1.0,
    )
    slow = TorchConditionalMechanicalMixturePriceModel(
        decay=0.99,
        reference_vwap_depth=1.0,
    )
    before_fast = np.asarray(fast.get_params()["gate_beta"], dtype=np.float64)
    before_slow = np.asarray(slow.get_params()["gate_beta"], dtype=np.float64)

    fast.update((0, book, 0.0, 0.0), (1, book, 0.0, 0.0))
    slow.update((0, book, 0.0, 0.0), (1, book, 0.0, 0.0))

    fast_params = fast.get_params()
    slow_params = slow.get_params()
    after_fast = np.asarray(fast_params["gate_beta"], dtype=np.float64)
    after_slow = np.asarray(slow_params["gate_beta"], dtype=np.float64)

    assert "gate_kind" not in fast_params
    assert "gate_bins" not in fast_params
    assert "gate_bin_probs" not in fast_params
    assert np.linalg.norm(after_fast - before_fast) > np.linalg.norm(after_slow - before_slow)
