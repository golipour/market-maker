"""
Controller: build State from OB + flow/position, call MPC -> optimal action. Modular.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .state import (
    Action,
    DiscretizedOrderAction,
    State,
    SigmoidMixtureAction,
    ZERO_DISCRETIZED_ACTION,
    sigmoid_action_to_discretized_action,
)
from .mpc_tensor_batched import MPCOptimizer


@runtime_checkable
class Controller(Protocol):
    """Given current state, return action a_t*. Estimators are provided at construction (to the optimizer)."""

    def act(
        self,
        state: State,
        *,
        rng: np.random.Generator | None = None,
    ) -> Action | DiscretizedOrderAction:
        ...


class SigmoidController:
    """
    Controller that uses an optimizer returning SigmoidMixtureAction (e.g. MonteCarloMPCTensorBatched),
    then converts to DiscretizedOrderAction so the execution model can compute fills
    from the discrete ladder (cumulative depth approximating the sigmoid curves).

    act(state) returns DiscretizedOrderAction: each side is a list of (spread_ticks, qty)
    where qty is MARGINAL (size at that level). Cumulative depth at spread s = sum of qty
    over all levels with spread_ticks <= s.
    """

    def __init__(
        self,
        optimizer,  # optimizer.optimize(state) -> SigmoidMixtureAction; estimators at optimizer construction
        *,
        s_b_min: float = 0.0,
        s_b_max: float = 100.0,
        s_a_min: float = 0.0,
        s_a_max: float = 100.0,
        lot_size: int = 1,
        spread_max: float | None = None,
        zero_action_if_negative_J_mean: bool = False,
    ):
        self.optimizer = optimizer
        self.spread_max = spread_max
        self.zero_action_if_negative_J_mean = zero_action_if_negative_J_mean
        self._discretization = {
            "s_b_min": s_b_min,
            "s_b_max": s_b_max,
            "s_a_min": s_a_min,
            "s_a_max": s_a_max,
            "lot_size": lot_size,
        }

    def act(
        self,
        state: State,
        *,
        rng: np.random.Generator | None = None,
    ) -> DiscretizedOrderAction:
        sigmoid_action = self.optimizer.optimize(state)
        if self.zero_action_if_negative_J_mean:
            J_mean = getattr(self.optimizer, "_last_E_dW", None)
            if J_mean is not None and J_mean < 0:
                return ZERO_DISCRETIZED_ACTION
        return sigmoid_action_to_discretized_action(
            sigmoid_action,
            total_q_min=getattr(self.optimizer, "total_q_min", None),
            **self._discretization,
        )


class DefaultController:
    """
    Build State = [mid, ob_rel, buy_vol, sell_vol, inventory, cash] from OB and position,
    call MPC optimizer (which uses ExecutionModel for sampling), return a_t*.
    """

    def __init__(
        self,
        optimizer: MPCOptimizer | None = None,
        spread_max: float | None = None,
    ):
        self.optimizer = optimizer
        self.spread_max = spread_max

    def act(
        self,
        state: State,
        *,
        rng: np.random.Generator | None = None,
    ) -> Action:
        return self.optimizer.optimize(state)
