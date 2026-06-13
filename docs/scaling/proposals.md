# Order Book Quality and Risk Improvements

Based on the review of the MPC optimization and state parameterization, here are concrete, incrementally implementable proposals to achieve a balanced, profitable, and less risky order book. The proposals focus on balancing the risk-adjusted return objective with explicit book-quality objectives, utilizing both tunable scalar weights and structural hard constraints.

## 1. Explicit Terminal Inventory Risk Penalty (Soft Constraint)

**Problem:** Currently, the terminal objective (`objective_entropic_return`) manages inventory risk indirectly via dynamic quote capacity limits, imbalance skew, and side-aware liquidation marking. This can lead to the optimizer drifting into high-inventory states if the short-term expected return from marking is marginally positive, increasing tail risk.
**Solution:** Incorporate an explicit quadratic terminal inventory penalty into the objective function, controlled by a scalar weight `w_inv`. This explicitly penalizes holding large absolute inventory at the horizon.

*   **Files to Change:** `mm/mpc/reward_tensor.py` and `mm/mpc/joint_mpc.py`
*   **Function:** `terminal_objective` and `joint_terminal_objective`
*   **Pseudocode / Math:**
    ```python
    # In terminal_objective:
    # Add parameter: w_inv: float = 0.0
    # Add parameter: terminal_inventory: Tensor (shape: n_samples)
    
    term_return = objective_entropic_return(returns, beta=return_risk_aversion_beta)
    
    # Calculate quadratic inventory penalty
    inv_penalty = -w_inv * torch.mean(terminal_inventory ** 2)
    
    total_objective = term_return + inv_penalty
    components = {
        "risk_adjusted_return": term_return,
        "inventory_penalty": inv_penalty,
        "total": total_objective
    }
    return total_objective, components
    ```
*   **Implementation Effort:** Low. The `inventory_pnl_term_tensor` already exists as a training helper but needs to be wired into the live rollout's terminal evaluation. The rollout loop in `mm/mpc/mpc_tensor_batched.py` must pass the terminal state's inventory to the objective.
*   **Prerequisites:** Exposing `w_inv` in the `ControllerRuntimeConfig` and strategy preset JSON.

---

## 2. Top-of-Book Spread Penalty (Soft Constraint)

**Problem:** The optimizer maximizes wealth. If market taker flow is sparse or toxic, the optimizer might place the majority of the depth far away from the mid (widening the spread) to avoid adverse selection, resulting in poor top-of-book depth and quality.
**Solution:** Introduce a book-quality objective that penalizes the deployed depth's distance from the mid-price. This creates a tunable trade-off: earn higher expected return by quoting wide, or pay a small "quality penalty" to quote tight.

*   **Files to Change:** `mm/mpc/reward_tensor.py` (and calling sites in `joint_mpc.py` / `mpc_tensor_batched.py`)
*   **Function:** Add a new function `book_quality_penalty_tensor` and update `terminal_objective`.
*   **Pseudocode / Math:**
    ```python
    def book_quality_penalty_tensor(action: ActionTensor, w_spread: float) -> Tensor:
        # Penalize the distance of the sigmoid centers (x0) from the mid.
        # action.bid_x0 and action.ask_x0 represent the spread offsets of the mixture components.
        # We can penalize the spread of the closest component (the top of book).
        
        bid_top_spread = action.bid_x0[0] # assuming first component is closest
        ask_top_spread = action.ask_x0[0]
        
        # Penalize linear or quadratic distance
        return -w_spread * torch.mean(bid_top_spread + ask_top_spread)
    ```
*   **Implementation Effort:** Medium. Requires computing the metric over the horizon actions during the rollout loop in `mm/mpc/mpc_tensor_batched.py` and adding it to the cumulative reward or terminal objective. 
*   **Prerequisites:** Expose `w_spread` in the runtime config.

---

## 3. Independent Per-Side Maximum Spread (Hard Constraint)

**Problem:** The current `spread_structured` constraint in `LParamConfig` enforces a joint spread cap: `x0_1_bid + x0_1_ask <= quote_spread_max`. While this guarantees tight *combined* spreads, the softmax split allows one side to be placed at `quote_spread_max` while the other is at `0`. In heavily skewed inventory states, the bot might completely abandon the top-of-book on one side, violating the requirement for a balanced order book.
**Solution:** Apply an independent hard maximum spread to *both* sides simultaneously, guaranteeing that the bot always maintains a presence near the mid-price on both the bid and the ask, regardless of the objective function's weights.

*   **Files to Change:** `mm/mpc/l_param.py`
*   **Function:** `_spread_centers_from_flat` and `materialize_side_torch`
*   **Pseudocode / Math:**
    ```python
    # In materialize_side_torch, replace the joint constraint:
    # Current:
    # centers = torch.clamp(scale * spread_cap * spread_w, min=min_center)
    
    # Proposed:
    # Add per_side_spread_cap to LParamConfig
    per_side_cap = torch.tensor(l_param.quote_spread_max_per_side, ...)
    
    # Enforce both the joint cap (if desired) and the strict independent cap
    centers = scale * spread_cap * spread_w
    centers = torch.clamp(centers, min=min_center, max=per_side_cap)
    
    # Adjust scale if the clamp reduced the values, or simply let the clamp enforce the hard boundary.
    ```
*   **Implementation Effort:** Low. It is a direct mathematical clamp on the tensor and numpy decoding paths. Because it is a hard constraint applied before the Adam closure, the optimizer will naturally route gradients within the feasible space without requiring a tuning weight in the objective.
*   **Prerequisites:** Add `quote_spread_max_per_side` to `LParamConfig` and the JSON config schema.

---

## 4. Minimum Dynamic Symmetrical Depth (Hard Constraint)

**Problem:** Dynamic quote capacity limits the maximum order size based on available cash/inventory, but it can allow the quoted depth to drop to very low levels on one side if inventory is imbalanced.
**Solution:** Enforce a hard minimum structural depth on the *first* layer (top of book) that ignores the imbalance skew, ensuring that "good depth" is always visible. The system already has `quote_spread_min_depth`, but we can enforce that this minimum depth is completely symmetrical and immune to the skew strength reduction.

*   **Files to Change:** `mm/mpc/l_param.py`
*   **Function:** `materialize_side_torch` (in the `L_eff` calculation for the first component)
*   **Pseudocode / Math:**
    ```python
    # When calculating L1 (the weight of the first sigmoid component):
    # Current: L1 = torch.maximum(L1, q_spread_t / d1_c)
    # Proposed: Ensure the minimum depth is strictly enforced regardless of q_total reduction from risk bounds.
    
    absolute_min_depth_t = torch.tensor(l_param.absolute_min_top_depth)
    L1 = torch.maximum(L1, absolute_min_depth_t / d1_c)
    ```
*   **Implementation Effort:** Low. Modifies the lower bound of the `L1` variable during action materialization.
*   **Prerequisites:** Add `absolute_min_top_depth` to `LParamConfig`.
