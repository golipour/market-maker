"""
Modular market-making controller: MDP state, dynamics, reward, and MPC.
Each component (price dynamics, order flow, execution, reward, optimizer) is
swappable so it can be changed independently.
"""

from .state import (
    MarketState,
    State,
    Action,
    SigmoidMixtureAction,
    sigmoid_mixture_action_bounds,
    sigmoid_action_to_action,
    OrderbookRelative,
    state_from_orderbook,
    state_from_step,
)
from .models import (
    PriceModel,
    GaussianPriceModel,
    OrderFlowModel,
    HurdleOrderFlow,
    ExecutionRule,
    DefaultExecutionRule,
    ExecutionModel,
    DiscretizedExecutionModel,
    ObservedStepExecutionModel,
)
from .state_tensor import (
    StateTensor,
    ActionTensor,
    DEFAULT_OB_LEVELS,
    state_to_tensor,
    state_tensor_to_state,
    action_tensor_from_sigmoid,
    sigmoid_from_action_tensor,
    depth_bid_tensor,
    depth_ask_tensor,
    depth_bid_derivative_tensor,
    depth_ask_derivative_tensor,
    slippage_penalty_tensor,
)
from .reward_tensor import (
    delta_pnl_detailed_tensor,
    inventory_pnl_term_tensor,
    slippage_pnl_term_tensor,
    reward_delta_inventory_slippage_tensor,
    REWARD_MODES,
    terminal_objective,
)
from .mpc_tensor_batched import MPCOptimizer, MonteCarloMPCTensorBatched
from .controller import Controller, DefaultController, SigmoidController

__all__ = [
    "MarketState",
    "State",
    "Action",
    "SigmoidMixtureAction",
    "sigmoid_mixture_action_bounds",
    "sigmoid_action_to_action",
    "OrderbookRelative",
    "state_from_orderbook",
    "state_from_step",
    "PriceModel",
    "GaussianPriceModel",
    "OrderFlowModel",
    "HurdleOrderFlow",
    "ExecutionRule",
    "DefaultExecutionRule",
    "ExecutionModel",
    "DiscretizedExecutionModel",
    "ObservedStepExecutionModel",
    "StateTensor",
    "ActionTensor",
    "DEFAULT_OB_LEVELS",
    "state_to_tensor",
    "state_tensor_to_state",
    "action_tensor_from_sigmoid",
    "sigmoid_from_action_tensor",
    "depth_bid_tensor",
    "depth_ask_tensor",
    "depth_bid_derivative_tensor",
    "depth_ask_derivative_tensor",
    "slippage_penalty_tensor",
    "delta_pnl_detailed_tensor",
    "inventory_pnl_term_tensor",
    "slippage_pnl_term_tensor",
    "reward_delta_inventory_slippage_tensor",
    "REWARD_MODES",
    "terminal_objective",
    "MPCOptimizer",
    "MonteCarloMPCTensorBatched",
    "Controller",
    "DefaultController",
    "SigmoidController",
]
