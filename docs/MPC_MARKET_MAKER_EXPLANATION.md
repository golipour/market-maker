# MPC Market Maker: Technical Explanation for Experts

This document describes how our **Monte Carlo Model Predictive Control (MC-MPC)** market maker works in detail, so an AI/RL/Control or market-making expert can review the design and suggest improvements.

---

## 1. High-Level Architecture

The system is a **receding-horizon, sample-based MPC** for two-sided limit order placement:

1. **State** is observed from the order book and position (mid, OB relative to mid, inventory, cash, and optionally recent flow).
2. **Action** is a **continuous parameterization of depth curves** (sigmoid mixtures), not discrete (δ_b, δ_a, q_b, q_a).
3. **Dynamics** are simulated over a finite horizon using **learned/estimated models** for price (residual on VWAPs) and **market order flow**; the synthetic book evolves by MOF consumption only; fills use our depth vs. book and sampled flow.
4. **Objective** is a **terminal objective** over the horizon: expected wealth change (ΔW) plus optional **risk terms** (variance, std, inventory penalty, or CARA). It is approximated by **Monte Carlo sampling** and optimized with **gradient-based search** (Adam) over action parameters.
5. Only the **first-step action** is executed; then we re-observe and re-solve (classic MPC).

**Pipeline flow:**  
`OrderFlowRunner.advance()` → get step (OB, buy_vol, sell_vol) → apply previous action via `ExecutionModel.state_after_step()` → update estimators → build `State` → `SigmoidController.act(state)` → `MonteCarloMPCTensorBatched.optimize(state)` → returns `SigmoidMixtureAction` → converted to `DiscretizedOrderAction` (ladder of (spread_ticks, qty)) → sent to execution; pending (state, action) stored for next step.

---

## 2. State Representation

**State** (see `market_making_controller/state.py`):

- **mid**: Mid price (e.g. in ticks).
- **ob_rel**: Order book **relative to mid** — list of `(spread_ticks, qty)` per side.  
  - Bids: `spread_ticks = mid - price_ticks` (positive = below mid).  
  - Asks: `spread_ticks = price_ticks - mid` (positive = above mid).  
  Used for depth and fill logic (how much flow “reaches” our quotes).
- **buy_vol, sell_vol**: Realized (or placeholder) market buy/sell volume in the current step (for state construction; inside MPC these are **sampled**).
- **inventory**: Integer (lots).
- **cash**: In consistent units (e.g. tick_size × lot_size).
- **buy_ref, sell_ref**: Representative ask/bid VWAP prices in ticks from the **current** book (same depth cap as the price model); used by MPC for initial refs and mark-to-market in the rollout.

So state is **s_t = (mid, ob_rel, buy_vol, sell_vol, inventory, cash, buy_ref, sell_ref)**. The controller sees the current OB and position; the MPC then **samples** future flow and price noise to evaluate actions.

---

## 3. Action Parameterization: Sigmoid Mixture

We do **not** optimize discrete (δ_b, δ_a, q_b, q_a). We optimize a **continuous depth curve** per side.

**SigmoidMixtureAction** (per side, then replicated for bid/ask):

- **Bid depth (cumulative volume from mid):**  
  **D_b(δ) = Σ_i L_i / (1 + exp(-k_i (δ - x0_i)))**  
  with **δ = spread in ticks** (1, 2, …, spread_max).
- **Ask depth:** same form with parameters (ask_L_i, ask_k_i, ask_x0_i).

Parameters per side: **n_components** triples **(L_i, k_i, x0_i)**:
- **L_i**: Asymptotic total quantity (softplus in PyTorch so L > 0).
- **k_i**: Steepness (softplus).
- **x0_i**: Inflection point (ticks from mid), unconstrained.

**Flat vector:** `[bid_L1, bid_k1, bid_x0_1, …, bid_Ln, …; ask_L1, …, ask_x0_n]` → **6 × n_components** dimensions for one step.  
Over a horizon **H**, we optimize **H** such vectors (full sequence **a_0, a_1, …, a_{H-1}**), but only **a_0** is used; the rest are for multi-step consistency in the objective.

**Execution:** The chosen `SigmoidMixtureAction` is converted to **DiscretizedOrderAction**: a ladder of (spread_ticks, qty) so that cumulative depth at each level approximates the sigmoid curves (see `sigmoid_action_to_discretized_action`). Fills are then computed per level: at each spread, “reach” = flow − competing depth − our cumulative depth before that level; fill = min(our qty at level, reach).

---

## 4. Dynamics Models (Used Inside MPC)

Two probabilistic model families are **fitted online** (EWMA/closed-form or similar) and **sampled** in the MPC loop (market flow + price residual). Their **parameters are frozen** during each `optimize(state)` call (no online learning inside the optimizer). All models live in **`market_making_controller/models/`**; we use **"model"** (not "dynamics" or "estimator") for naming.

### 4.1 Price model

- **GaussianPriceModel** (execution / discrete mid sampling): used by `DiscretizedExecutionModel` for `sample_next_mid`, not the MPC rollout core.
- **TorchResidualPriceModel** (default in pipeline): two-sided **residual on representative VWAPs**. From previous and current order books plus window MOF, it forms instantaneous buy/sell VWAPs to a target depth on (i) the realized book and (ii) the MOF-only post-flow synthetic book; **e_buy**, **e_sell** are the gaps. **μ** and **σ** per side are EWMA-updated from those residuals (the VWAP levels themselves are not EWMA-smoothed). **In MPC:** after MOF consumption updates the synthetic book, representative buy/sell prices are computed and **independent Gaussian noise** (`sample_residual`) is added to get next-step **buy_ref**, **sell_ref**; **mid** comes from the post-flow book touch.

### 4.2 Market order flow (buy/sell volume)

- **Model:** `TorchHurdleLogNormalModel` — hurdle + LogNormal per side; params from EWMA.
- **In MPC:** Draw **(buy_vol, sell_vol)** per sample per step. **buy_vol** consumes ask depth; **sell_vol** consumes bid depth.

### 4.3 Limit order flow (LOF)

- **Not used in MPC rollouts** (book evolves by MOF consumption only). Optional `limit_order_flow_model` on the runner remains for future use / animation helpers; `limit_order_flow_observed` in `flow_implied.py` supports observed cumulative LOF plots.

---

## 5. Execution / Fill Model (Differentiable)

Fills are computed in a **differentiable** way so we can backprop through the objective to the action parameters.

**Idea:**  
- **Book cumulative depth** at grid points: `book_cum_bid(δ)`, `book_cum_ask(δ)` from current (or evolved) OB.  
- **Our depth** at same grid: from sigmoid mixture **D_b(δ)**, **D_a(δ)** (using softplus for L, k).  
- **Reach:** On bid side, “how much sell flow gets to our level” = **reach_bid = max(0, sell_vol − book_cum_bid)** (per level or aggregated); similarly **reach_ask = max(0, buy_vol − book_cum_ask)**.  
- **Soft fill:** To keep gradients, we use a **soft** fill: e.g. **fill_bid_level = our_depth × sigmoid(reach / temp)** with a temperature parameter. So when flow barely reaches us, we get partial fill; when it exceeds, we get full.  
- **Cash and inventory delta:**  
  - **cash_delta** = (fill_ask side revenue) − (fill_bid side cost) with maker fee **(1−f)**.  
  - **inv_delta** = fill_bid×(1−f) − fill_ask.

So **next state** (refs, inv_next, cash_next) is computed from (refs, inv, cash), action params, and sampled (buy_vol, sell_vol); **OB_next** from OB and MOF only. All in PyTorch so **∇_params (reward)** exists.

---

## 6. Reward (Objective)

**Per-step reward** is **ΔW** (change in wealth):

- **Wealth** = cash + inventory × **sell_ref** (mark-to-market at bid-side representative price in ticks).  
- **ΔW = (cash_next + inv_next × sell_ref_next) − (cash + inv × sell_ref)**.

**Per-step quantity:** Each trajectory yields **ΔW** per step; summed over **H** steps gives **total ΔW** per trajectory (used inside `terminal_objective`). Differentiable tensor variants still use mid-based `delta_pnl_detailed_tensor` elsewhere; batched MPC uses `_delta_wealth_mtm_batched` in `mpc_tensor_batched.py`.

**Terminal objective (risk-aware):** The MPC uses **`terminal_objective`** (`reward_tensor.py`) which builds a scalar **J** from **delta_W** (n_samples,) and **inv_terminal** (n_samples,). **Reward modes** (`REWARD_MODES`):

| mode | J (maximized) | Parameters |
|------|----------------|------------|
| **step_pnl** / **terminal_pnl** | J = **J_mean** = E[ΔW] | — |
| **terminal_pnl_inventory** | J = J_mean − **γ** · J_inv | **reward_gamma** (γ); J_inv = E[q_{t+H}²] |
| **terminal_mean_variance** | J = J_mean − **λ** · J_var | **reward_lambda** (λ); J_var = Var(ΔW) |
| **terminal_mean_std** | J = J_mean − **λ** · J_std | **reward_lambda** (λ); J_std = √Var(ΔW) |
| **terminal_cara** | J = **J_entropic** = −(1/η) log(E[e^{−η ΔW}]) | **reward_eta** (η) |

So we can penalize **terminal inventory squared** (γ), **variance** or **std of ΔW** (λ), or use **CARA** (η). Pipeline passes **reward_mode**, **reward_gamma**, **reward_lambda**, **reward_eta** into the MPC; e.g. `reward_mode="terminal_mean_std"`, `reward_lambda=1.0`.

**Optional safety:** **zero_action_if_negative_J_mean**: if True (controller), after optimization we check **J_mean** = E[ΔW] for the best action; if J_mean < 0 we return **no quotes** (zero action).

---

## 7. Optimization Algorithm (Monte Carlo MPC)

**Class:** `MonteCarloMPCTensorBatched` (`market_making_controller/mpc_tensor_batched.py`).

**Parameters (typical from config):**

- **n_components**: 1 (or 2) — number of sigmoid components per side.  
- **n_samples**: 32 — number of MC trajectories per epoch.  
- **horizon**: 5 — number of steps H.  
- **spread_max**, **n_grid**, **ob_levels**: e.g. 100 — spread grid and OB level count.  
- **adam_lr**, **adam_epochs**, **adam_grad_clip**: e.g. 1e1, 33, 1e10 — Adam for action vector.  
- **total_q_min**, **total_q_max**: optional box on **sum of L** per side (projected each epoch).  
- **fixed_mc_samples**: if True, draw **one** set of (buy_vol, sell_vol) for all steps and epochs; if False, redraw every epoch.  
- **warm_start**: if True, initialize from previous solution (last action replicated over H).  
- **adam_normalize_inputs**: scale L by a volume scale (from OB) for numerics.  
- **reward_mode**: one of `REWARD_MODES` (step_pnl, terminal_pnl, terminal_pnl_inventory, terminal_mean_variance, terminal_mean_std, terminal_cara).  
- **reward_gamma**, **reward_lambda**, **reward_eta**: parameters for inventory penalty (γ), variance/std penalty (λ), and CARA (η).

**Decision variable:**  
- One vector **x** of length **H × (6 × n_components)**.  
- Bounds: per-component [L_low, L_high], [k_low, k_high], [x0_low, x0_high] (from `sigmoid_mixture_action_bounds`), replicated for each of H steps.  
- Optional projection so that **sum(bid_L)** and **sum(ask_L)** lie in [total_q_min, total_q_max] (same factor on all L per side).

**Loop:**

1. Build initial state tensors from **state**: mid_0, buy_ref_0, sell_ref_0, inv_0, cash_0, **book_cum_bid**, **book_cum_ask**, **ob_bids_0**, **ob_asks_0** (replicated to n_samples).  
2. **(Optional)** Pre-draw all MC samples: **buy_vol**, **sell_vol** (n_samples, H).  
3. **For epoch = 1 … adam_epochs:**  
   - **Closure:**  
     - Clamp **x** to bounds and project total L.  
     - If not fixed_mc_samples, draw new samples.  
     - For **h = 0 … H−1:**  
       - **book_cum** at h: at h=0 use current OB; at h>0 use evolved OB from previous step.  
       - **action_h** = slice of **x** for step h.  
       - Compute **fill_bid, fill_ask, cash_delta, inv_delta** with `_fill_and_cash_batched(action_h, …)` (differentiable).  
       - **inv_next = inv + inv_delta**, **cash_next = cash + cash_delta**.  
       - **ob_bids, ob_asks** ← MOF consumption only (`_consume_mof_only_torch`).  
       - **buy_ref_next**, **sell_ref_next** ← representative VWAPs on post-flow book + `price_model` noise; **mid_next** = average of the two refs.  
       - **Step reward** = `_delta_wealth_mtm_batched(sell_ref, inv, cash, sell_ref_next, inv_next, cash_next)`.  
       - Accumulate **total_reward** per sample.  
       - Roll **buy_ref**, **sell_ref**, **mid**, **inv**, **cash** forward.  
     - **Loss** = −J (we maximize J, so minimize −J).  
     - **loss.backward()**, clip gradient, then **optimizer.step(closure)**.  
4. Return **best** action (first-step slice of **x**) as **SigmoidMixtureAction**; store for warm start and for E[ΔW] (J_mean) logging. **J** is from `terminal_objective(total_reward, inv_t, reward_mode, reward_gamma, reward_lambda, reward_eta)`.

So we have **receding-horizon MPC**: at each real step we solve a finite-horizon stochastic control problem with MC expectation and gradient ascent, then apply the first action only.

---

## 8. Controller and Execution in the Loop

- **SigmoidController**: Holds the `MonteCarloMPCTensorBatched` optimizer. `act(state)` calls `optimizer.optimize(state)` and converts the returned `SigmoidMixtureAction` to `DiscretizedOrderAction`. Optional **zero_action_if_negative_J_mean**: if True and the optimizer’s **J_mean** (E[ΔW]) for the best action is < 0, returns **no quotes** (ZERO_DISCRETIZED_ACTION).  
- **Execution**: When the next step arrives, `DiscretizedExecutionModel.state_after_step(prev_state, prev_action, curr_step)` uses **realized** (buy_vol, sell_vol) from the step and computes integer fills from the discrete ladder, then updates inventory, cash, and exposes **ΔW** and PnL decomposition (spread, inventory, cross).  
- **Models** (price, market flow, limit order flow) are updated in the runner **after** applying the previous action and **before** calling the controller, so the MPC always uses the latest fitted models. The runner stores **expected_mid_hist** (flow-implied mid per step) when using the residual price model, for plotting alongside realized mid.

Execution timing and provenance (important):

- action decided at step `t` is stored as pending
- realized transition happens at step `t+1` via `state_after_step(...)`
- runner `inventory`, `cash`, `wealth`, realized `ΔW`, and cumulative spread/inventory/cross PnL plots are derived from that execution output path
- optimizer `E[ΔW]` plot is separate: predicted value from MPC at action time, not realized execution

---

## 9. Design Choices (Summary)

| Aspect | Choice | Rationale / note |
|--------|--------|-------------------|
| **Action space** | Continuous sigmoid mixture over depth curves | Smooth, differentiable; avoids discrete (δ, q) grid; one action describes full ladder. |
| **Horizon** | Finite H (e.g. 5) | Classic MPC; only a_0 executed. |
| **Uncertainty** | Monte Carlo over MOF and price residuals | Simple; no explicit belief state; expectation approximated by sample mean. |
| **Optimizer** | Adam on full H-step sequence | Gradient-based; reparameterized sampling where used (e.g. Normal, LogNormal). |
| **Objective** | J = terminal_objective(ΔW, inv_terminal; mode, γ, λ, η) | step_pnl: J=E[ΔW]. terminal_mean_std: J=E[ΔW]−λ Std(ΔW). terminal_pnl_inventory: J=E[ΔW]−γ E[q²]. terminal_cara: J_entropic. Optional zero_action_if_negative_J_mean. |
| **Price** | TorchResidualPriceModel: two-sided VWAP residual + noise | MOF-only synthetic book; residual ~ N per side in MPC sampling. |
| **Flow** | Hurdle-LogNormal, independent per side | Fitted from data; sampled per step. |
| **OB evolution** | Consumption by MOF only in MPC | Deterministic given flow sample. |
| **Fills** | Soft fill (sigmoid of reach) for gradients | Differentiable; temperature controls “hardness”. |

---

## 10. Possible Questions for an Expert

These can guide the discussion and improvements:

1. **Objective:** The implementation already supports **risk-aware** objectives (terminal_mean_variance, terminal_mean_std, terminal_pnl_inventory, terminal_cara). How should we tune **reward_lambda** (λ) or **reward_gamma** (γ) / **reward_eta** (η) in practice? Is **terminal_mean_std** a good default vs variance or CARA?

2. **Dynamics:** With the residual price model, mid is coupled to flow/LOF; the residual is exogenous. Should we also model **price impact** of our own orders (e.g. temporary impact when we absorb flow), and if so, how to keep it differentiable and tractable?

3. **LOF and OB:** Is the current “consumption + Gaussian LOF” evolution a reasonable approximation for the book? Should LOF be **correlated** with flow or mid moves?

4. **Action parameterization:** Are sigmoid mixtures a good choice for depth curves, or would other bases (e.g. splines, piecewise linear) give better expressiveness vs. overfitting and gradient stability?

5. **Optimization:** Is Adam on the full H-step sequence a good choice, or would **stage-wise** (optimize a_0 with a_1,…,a_{H-1} fixed, then roll) or **differentiable shooting** improve stability/speed? Any benefit from **multiple restarts** or **smoothing** of the MC objective?

6. **Samples and variance:** With **n_samples=32**, **H=5**, how much variance in the gradient should we expect? Is **fixed_mc_samples** (same scenarios every epoch) better for convergence or worse for overfitting to a small set of paths?

7. **Soft vs hard fills:** The soft fill (sigmoid(reach/temp)) is for differentiability. How to choose **soft_fill_temp** so that gradients are useful but the policy is close to real execution? Should we use **hard** fills and a **smoothing** (e.g. Convex Relaxation or SPSA) instead?

8. **Warm start:** Reusing the previous solution over H steps as initial guess — does this bias the solver toward “slow” updates, and when would it be better to re-initialize (e.g. after large mid or OB moves)?

9. **RL / learning:** Could we **learn** the dynamics (price, flow, LOF) or a value function **end-to-end** while still using this MPC as a policy (e.g. MPC as inner loop, outer loop tuning models or weights)? Any prior work that combines MPC with learned models in market making?

10. **Control theory:** From an MPC perspective, is there a natural **terminal cost** or **stabilizing constraint** (e.g. on inventory) that would make the receding-horizon problem well-posed and reduce end-of-horizon effects?

---

**Cross-exchange hedging**

Suppose there is another **larger spot exchange** where we can hedge inventory (the ideal hedge might be a derivative exchange, but assume cross-exchange spot hedging for now). The other exchange has its own **spot price**, **spread**, and **trading fee**.

11. **Hedging in the objective:** How could we add hedging to the MPC objective? Should we hedge **every trade** or use a **threshold** (e.g. hedge when |inventory| > I_max)? What threshold or rule is reasonable, and could the **objective alone** (e.g. expected wealth including hedge PnL and costs) answer "when and how much to hedge" without a separate threshold, or do we need an explicit inventory penalty / constraint?

12. **Two-exchange mid difference:** If the **mid price** on the two exchanges is not the same (basis, latency, or liquidity), how should we **model** that? For example: spread between exchange A mid and exchange B mid as a random process (e.g. mean-reverting or random walk), or a separate state variable for "hedge exchange mid" with its own dynamics and correlation to the primary exchange. How to incorporate this into the MPC state and reward (e.g. hedge execution at hedge_mid ± spread and fee)?

---

**Probability distributions and heavy tails**

13. **Alternative distributions for market order flow:** Which **probability distributions** could replace or complement the current **hurdle LogNormal** (or discrete Gamma) for buy/sell volume? Should we consider **heavy-tailed** models (e.g. Pareto, Student-t, or mixture of LogNormal with a heavy-tail component) to better capture large flow events? How to keep **reparameterized sampling** and **gradients** tractable for such distributions in the batched MPC?

14. **Alternative distributions for limit order flow:** If LOF is reintroduced into MPC rollouts, what distributions are natural for **order book replenishment** or **cumulative order flow** (e.g. heavy-tailed, skewed, or truncated)?

15. **Heavy tails in the residual price model:** VWAP residuals per side are currently **Gaussian**. Should we use **heavy-tailed** distributions (e.g. Student-t, mixture) for better tail risk, and what are the trade-offs for **differentiability** and **EWMA-style parameter updates**?

---

## 11. File Reference

- **Pipeline / entry:** `order_flow_pipeline.py` — builds models, testers, `MonteCarloMPCTensorBatched`, `SigmoidController`, execution model/rule; `create_runner`, `main`.  
- **MPC core:** `market_making_controller/mpc_tensor_batched.py` — `MonteCarloMPCTensorBatched.optimize(state)`, batched rollouts, Adam, fill/dynamics helpers.  
- **MPC helpers:** `market_making_controller/mpc_tensor_batched.py` — bounds, total_q projection, book cumulative from OB, normalization (inlined; single MPC implementation).  
- **State / action:** `market_making_controller/state.py` — `State`, `OrderbookRelative`, `SigmoidMixtureAction`, `DiscretizedOrderAction`, conversion and bounds.  
- **Tensor state/depth/reward:** `market_making_controller/state_tensor.py`, `market_making_controller/reward_tensor.py` — differentiable depth and ΔW.  
- **Reference VWAPs:** `market_making_controller/ob_reference_prices.py` — shared buy/sell ref ticks from OB and transitions.  
- **Execution:** `market_making_controller/models/execution.py` — `DiscretizedExecutionModel`, `state_after_step`, fill rules.  
- **Controller:** `market_making_controller/controller.py` — `SigmoidController.act(state)`.  
- **Dynamics:** `market_making_controller/models/` — **price_torch.py** (GaussianPriceModel, TorchResidualPriceModel), **market_order_flow_torch.py** (TorchHurdleLogNormalModel), **flow_implied.py** (Step, `limit_order_flow_observed`), **execution.py**.
- **Runner:** `order_flow_runner.py` — `advance()`, state construction (`buy_ref`/`sell_ref` from OB), controller call, execution model, **model** (market order flow), **price_model**, optional **limit_order_flow_model**, **expected_mid_hist**.

This should give an expert enough context to suggest concrete improvements to the formulation, models, or implementation.
