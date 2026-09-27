This plan focuses on improving the existing differentiable MPC optimizer’s latency and variability without changing its market-making economics: no change to the reward, risk terms, fill model, execution logic, number of Monte Carlo paths, horizon, or Adam epochs as the primary means of making it faster.

The principal implementation targets are `mm/mpc/mpc_tensor_batched.py` and `mm/mpc/price_torch.py`, with configuration, diagnostics, persistence, and tests as supporting work.

## Phase 0 — Establish a benchmark and correctness baseline

1. Build a deterministic replay harness for optimizer decisions.
   - Use `fixed_mc_samples: true` so every implementation sees the same market-order-flow samples and price innovations.
   - Persist/reconstruct:
     - input MPC state;
     - public book;
     - optimizer config;
     - initial / warm-start action;
     - flow samples;
     - price innovations;
     - random seed;
     - initial, final, and best action;
     - target executable ladder;
     - post-risk target book.
   - The harness must run the current float64 eager implementation as the reference and candidate variants against exactly identical inputs.

2. Create a representative Bitwana corpus.
   - Collect roughly 100–1,000 historical decision frames for timing statistics.
   - Include named edge cases:
     - quiet book with little or no flow;
     - large bid/ask depth asymmetry;
     - high inventory;
     - low cash or low base inventory;
     - sharp book-shape / price refresh;
     - frame where optimization materially improved the initial action;
     - historically slow optimizer invocation;
     - frames close to viability or validity boundaries.

3. Define decision-level parity criteria before changing code.
   - Candidate and reference must agree on risk approval/rejection.
   - Same order-side distribution and order count, except for explicitly approved ladder simplifications.
   - Price differences at most one venue tick.
   - Quantity differences at most one venue step, or a documented relative tolerance.
   - No increase in non-finite losses or gradients.
   - No violation of cash, inventory, or open-notional limits.
   - No material deterioration in expected objective, first-step target book, or replay markout.
   - Exact optimizer parameter equality is not a float32 requirement; operationally equivalent orders are.

4. Add fine-grained timing instrumentation.
   - Break current broad timing into:
     - setup / cache preparation;
     - Monte Carlo sampling;
     - rollout forward;
     - objective and penalty computation;
     - backward;
     - Adam update;
     - diagnostics and snapshot construction;
     - compilation / recompilation time.
   - Record each phase at decision level, and optionally aggregate per epoch without serializing full diagnostics each time.

5. Fix normalized persistence.
   - `optimize_ms` is currently null despite timing information existing in diagnostics/payloads.
   - Persist timing fields as queryable fact columns rather than relying on expensive JSON scans:
     - `setup_ms`, `sample_draw_ms`, `forward_ms`, `objective_ms`,
       `backward_ms`, `adam_step_ms`, `diagnostics_ms`, `train_ms`,
       `optimize_ms`;
     - optimizer dtype;
     - eager/compiled mode;
     - compile-cache hit/miss;
     - compile duration;
     - graph-break/recompile count;
     - best epoch and actual epoch count;
     - non-finite loss/gradient counters.

## Phase 1 — Make dtype configurable and safely enable float32

1. Add validated optimizer configuration.
   - Introduce:
     ```json
     {
       "tensor_dtype": "float64"
     }
     ```
   - Supported values initially:
     ```json
     "float64"
     "float32"
     ```
   - Default remains `float64` until validation is complete.
   - Do not silently fall back between dtypes.
   - Emit resolved dtype in decision diagnostics and persistence.

2. Apply dtype consistently.
   - Remove the hard-coded `torch.float64` in `MonteCarloMPCTensorBatched.optimize`.
   - Ensure the chosen dtype is used for:
     - action vector and bounds;
     - spread grid;
     - public-book tensors;
     - cash, inventory, capacity, and state tensors;
     - market-flow samples;
     - price innovations;
     - model coefficients;
     - objective / viability / validity constants.
   - Avoid mixed float32/float64 arithmetic, which can silently promote hot-path work back to float64.

3. Normalize optimizer-internal numerical scales.
   - Float32 has roughly seven meaningful decimal digits. Raw large IRT cash/notional values can make useful quote-size differences disappear.
   - At each decision, define positive scales:
     \[
     P_{\text{scale}} = \max(m_0, \epsilon)
     \]
     \[
     Q_{\text{scale}} =
     \max\left(
       Q_{\text{inventory}},
       \frac{C_0}{P_{\text{scale}}},
       Q_{\text{cap}},
       1
     \right)
     \]
     \[
     W_{\text{scale}} = P_{\text{scale}} Q_{\text{scale}}
     \]
   - Use dimensionless internal state:
     \[
     \tilde m = m / P_{\text{scale}},
     \quad
     \tilde q = q / Q_{\text{scale}},
     \quad
     \tilde c = c / W_{\text{scale}}
     \]
   - Convert to raw venue prices and quantities only at the optimizer boundary and when constructing final diagnostics/orders.

4. Scale thresholds consistently.
   - Normalizing state but leaving raw thresholds untouched would change strategy behavior.
   - Audit and transform all raw-unit values:
     - quantity floors/caps;
     - wallet capacity bounds;
     - viability cash and inventory buffers;
     - minimum viable quote quantities;
     - cumulative depth caps/floors;
     - book-quality reference quantities;
     - objective normalizers and penalty thresholds.
   - Ensure every objective term remains economically equivalent after scaling.

5. Preserve numerical stability.
   - Keep existing clamps for logs, divisions, variance, and exponentials.
   - Replace arbitrary tiny constants with dtype-aware values only where mathematically appropriate.
   - Maintain gradient clipping.
   - Record non-finite values separately by stage: price model, fill model, objective, backward, optimizer update.
   - Benchmark pure float32 first; do not add mixed precision unless replay proves it is needed.

6. Validate float32 in stages.
   1. Float64 eager with normalized internal quantities.
   2. Float32 eager with normalized internal quantities.
   3. Deterministic replay comparison.
   4. Repeated-seed distribution comparison.
   5. Shadow execution against live frames, without sending candidate orders.
   6. Only then consider live float32.

## Phase 2 — Extract a compile-friendly pure tensor kernel

1. Separate the differentiable hot path from Python orchestration.
   - Extract a pure tensor function conceptually shaped as:
     \[
     \mathcal{L}(\theta, S_0, B_0, I, K)
       \rightarrow (-J, \mathbf c, \mathbf r)
     \]
     where:
     - \(\theta\) is the action parameter tensor;
     - \(S_0\) is normalized initial state;
     - \(B_0\) is initial book tensors;
     - \(I\) is fixed Monte Carlo flow and price innovations;
     - \(K\) is static numeric/model parameter tensors;
     - \(J\) is scalar objective;
     - \(\mathbf c\) is a fixed-size objective-component tensor;
     - \(\mathbf r\) is fixed-shape rollout summary data.

2. Keep only tensor operations inside the compiled kernel.
   - No Python dictionaries.
   - No JSON construction.
   - No `.item()`, `.cpu()`, `.numpy()`, or Python scalar conversion.
   - No target-book or order objects.
   - No event bus, risk-engine, or persistence calls.
   - No `hasattr` checks or dynamic model capability branches.

3. Move orchestration outside the kernel.
   - State construction.
   - Normalization and denormalization.
   - Grid construction/caching.
   - Monte Carlo sample generation.
   - Adam state management.
   - Best-action tracking.
   - Diagnostics conversion.
   - Action snapshots.
   - Target-book construction.
   - Risk and execution flow.

4. Replace mutable component dictionaries with a fixed tensor layout.
   - Example:
     ```python
     COMPONENT_INDEX = {
         "terminal": 0,
         "risk_adjustment": 1,
         "validity": 2,
         "viability": 3,
         "book_score": 4,
         "viability_shortfall": 5,
     }
     ```
   - Convert component tensors to named diagnostics only after the best action is known.

5. Specialize kernels by price-model variant.
   - Resolve the current conditional mechanical-mixture model once at optimizer construction.
   - Use dedicated variants rather than a runtime branch inside the rollout:
     - conditional-mechanical-mixture kernel;
     - optional residual/legacy kernel.
   - This prevents graph breaks and enables stable compilation.

6. Add `torch.compile` behind a feature flag.
   - Initial configuration:
     ```python
     torch.compile(
         rollout_objective,
         dynamic=False,
         fullgraph=False,
         mode="reduce-overhead",
     )
     ```
   - Cache compiled kernels by shape and model characteristics:
     \[
     (\text{dtype}, N, H, G, C, \text{price model}, \text{queue model})
     \]
   - Do not compile on every decision.

7. Provide a safe eager fallback.
   - If compilation fails, graph-breaks excessively, or is slower than eager:
     - log a structured reason;
     - execute eager Torch safely;
     - continue strategy execution;
     - do not let compilation failure block live quoting.
   - Track compiled/eager mode, compile time, cache hit/miss, graph breaks, and candidate latency.

## Phase 3 — Remove repeated allocation and repeated tensor conversion

1. Create a decision-local immutable tensor bundle.
   - Build once before Adam:
     - spread grid;
     - initial cumulative bid/ask depth;
     - initial level depths;
     - lower/upper action bounds;
     - normalized initial state;
     - fees;
     - MC flow samples;
     - price innovations;
     - capacity and viability constants;
     - static sigmoid/depth configuration;
     - price-model tensors;
     - reference VWAP depth;
     - shape alpha;
     - Student-t degrees of freedom.

2. Refactor `price_torch.py` to expose decision-scoped tensor parameters.
   - The online model can retain NumPy/RLS state, but tensor conversion should happen once per decision.
   - Add an immutable structure similar to:
     ```python
     @dataclass(frozen=True)
     class ConditionalPriceTensorParams:
         market_beta: Tensor
         refresh_mean: Tensor
         market_sigma: Tensor
         refresh_sigma: Tensor
         gate_beta: Tensor
         reference_vwap_vol: Tensor
         shape_alpha: Tensor
         student_df: Tensor
         eps: Tensor
     ```
   - Add:
     ```python
     params = price_model.tensor_params(
         device=device,
         dtype=dtype,
         version=model_update_version,
     )
     ```
   - Invalidate cached tensors when the online model learns new parameters.

3. Add a pure tensor price-model API.
   - Refactor `sample_dynamic_vwap` into a tensor-only inner function such as:
     ```python
     sample_dynamic_vwap_tensor(..., params=params)
     ```
   - It must not construct static tensors from Python/NumPy values during every horizon step.

4. Precompute static book geometry.
   - Cache:
     - grid indices;
     - fixed price offsets;
     - cumulative-to-marginal transforms;
     - best-level masks;
     - grid-derived reference quantities.
   - Cache per optimizer instance when only configuration-dependent, otherwise per decision.

5. Audit `.clone()` and allocation use.
   - Preserve semantic reset of simulated book state each epoch.
   - Use `expand` for shared immutable views.
   - Use `clone()` only for state that is actually mutated or must be independent.
   - Benchmark functional updates versus in-place updates after compilation; compiler fusion can favor functional code.

## Phase 4 — Reduce optimized action dimension while preserving MPC simulation

1. Add horizon action modes:
   ```json
   {
     "horizon_action_mode": "independent"
   }
   ```
   Supported modes:
   - `independent`: current reference behavior;
   - `hold`: one action shared across all simulated horizons;
   - `linear_delta`: first action plus small low-dimensional future adjustment.

2. Implement `hold` first.
   - Instead of independently optimizing every \(\theta_h\):
     \[
     \theta_h = \theta_0 \quad \forall h
     \]
   - The optimizer still simulates every horizon step and values future inventory/book state.
   - It only removes unexecuted future-action degrees of freedom.
   - For `H = 2`, this approximately halves action dimension and Adam-state size.

3. Validate `hold` against `independent`.
   - Compare:
     - first executable action;
     - target book;
     - risk result;
     - viability shortfall;
     - first-step expected reward;
     - inventory direction;
     - replay markout.
   - Prioritize first-step quality because only the first action is actually executed.
   - Add `linear_delta` only if `hold` causes meaningful degradation in inventory-stressed cases.

4. Preserve backpropagation through time.
   - In hold mode, \(\theta_0\) must influence every simulated horizon:
     \[
     \frac{\partial J}{\partial \theta_0}
       =
       \sum_h
       \frac{\partial J}{\partial S_h}
       \frac{\partial S_h}{\partial \theta_0}
       +
       \frac{\partial J}{\partial \theta_0}
     \]
   - Do not detach later simulated price, book, cash, or inventory state for speed.

## Phase 5 — Remove unnecessary validity work

1. Confirm actual live validity behavior.
   - The observed Bitwana configuration has:
     ```json
     {
       "validity": {
         "min_score": 0.3,
         "penalty_scale": 0
       }
     }
     ```
   - Verify `combine_validity_term` behavior precisely.
   - If `penalty_scale = 0` means validity has no objective contribution or gradient, it should not be evaluated every epoch.

2. For inactive validity:
   - Evaluate validity for:
     - initial action;
     - best action;
     - final action.
   - Retain it for diagnostics and post-optimization validation.
   - Remove it from the differentiable inner loop only if it is mathematically non-contributory.

3. For active validity:
   - Keep it in the gradient path.
   - Include it in the compiled objective kernel as a fixed component tensor.
   - Reuse static geometry and avoid a second forward pass.
   - Specialize active/inactive validity kernels outside the epoch loop.

## Phase 6 — Minimize Python and diagnostics inside the Adam loop

1. Make the epoch loop deliberately small.
   - Per epoch, only:
     1. cancellation check;
     2. compiled or eager tensor objective call;
     3. backward;
     4. gradient clipping;
     5. Adam step and projection;
     6. best-loss comparison;
     7. early-stop decision;
     8. lightweight timing counters.

2. Reduce diagnostic extraction.
   - Do not convert every component to Python floats each epoch.
   - Do not create full `EpochObjectiveMetrics` / component maps each epoch.
   - Record complete diagnostics:
     - first epoch;
     - best epoch;
     - final epoch;
     - optionally every Nth epoch, such as every fifth epoch.
   - Convert tensors to Python values only when needed for control or final reporting.

3. Preserve best state safely.
   - Cloning a small action vector on an objective improvement is acceptable.
   - Avoid building full depth/action snapshots every epoch.
   - Construct snapshots after optimization for initial, best, and final action only.

4. Consider early stopping only after parity infrastructure exists.
   - It should be optional and conservative:
     - minimum epoch floor;
     - objective-improvement tolerance;
     - patience;
     - retain best action, not merely final action.
   - Treat it as a measured optimization, not an assumed latency win.

## Phase 7 — Compile the static horizon recurrence

1. Start with static unrolling.
   - With fixed `H` and `dynamic=False`, allow Torch Dynamo/Inductor to unroll:
     ```python
     for h in range(self.horizon):
     ```
   - This is low-risk, especially because production horizon is small.

2. Only consider scan-style approaches if profiling justifies them.
   - A pure transition is:
     \[
     S_{h+1} = F(S_h, \theta_h, \epsilon_h)
     \]
   - Consider `torch.func`/scan-like recurrence only if compiled static unrolling still shows material Python or recurrence overhead.
   - At `H = 2`, a scan abstraction may add complexity without measurable benefit.

## Phase 8 — Vectorize the conditional mechanical-mixture price model

1. Replace per-component Python loops.
   - Treat the four modeled components as a final dimension:
     \[
     \mathbf{x} \in \mathbb{R}^{N \times 1 \times 4}
     \]
   - Use tensors such as:
     \[
     B_{\text{market}} \in \mathbb{R}^{4 \times 2},
     \quad
     \mu_{\text{refresh}} \in \mathbb{R}^{4},
     \quad
     \sigma_{\text{market}}, \sigma_{\text{refresh}} \in \mathbb{R}^{4}
     \]
   - Compute all component means/scales in one vectorized operation.

2. Precompute Student-t constants.
   - Cache:
     \[
     c_\nu = \sqrt{\frac{\nu - 2}{\nu}}
     \]
   - Use:
     \[
     \sigma_{\text{refresh,effective}}
       = c_\nu \sigma_{\text{refresh}}
     \]
   - Do not rebuild this scalar inside every rollout call.

3. Preserve model semantics exactly.
   - Keep:
     - gate logic;
     - `point_mode`;
     - market versus refresh sampling;
     - log-space clamps;
     - dynamic VWAP conversion;
     - non-finite fallback behavior.
   - Validate each component’s sampled output against the float64 reference with fixed innovations.

## Phase 9 — Target post-refactor runtime structure

The intended execution path is:

```text
decide()
  ├─ update online flow and price models
  ├─ construct State
  ├─ prepare normalized optimizer inputs
  │    ├─ book/grid tensors
  │    ├─ fixed MC innovations
  │    ├─ model parameter tensors
  │    └─ warm-start action
  ├─ resolve cached compiled kernel
  ├─ Adam loop
  │    ├─ cancellation check
  │    ├─ tensor objective/rollout
  │    ├─ backward + clip + Adam update
  │    ├─ best-action tracking
  │    └─ early-stop policy
  ├─ evaluate diagnostics for initial/best/final only
  ├─ denormalize best first-step action
  ├─ construct executable target book
  └─ run existing risk and execution pipeline
```

The compiled kernel must have no awareness of events, JSON, orders, execution grids, target books, strategy decision objects, or risk-engine classes.

## Phase 10 — Benchmark, shadow, and safely roll out

1. Run a controlled benchmark matrix.

   - A: float64, eager, independent horizon actions — baseline.
   - B: float64, eager, hold horizon actions — isolate dimension reduction.
   - C: float64, compiled, hold — isolate compilation.
   - D: normalized float32, eager, hold — isolate dtype.
   - E: normalized float32, compiled, hold — target implementation.
   - F: normalized float32, compiled, linear delta — only if hold fails quality checks.

2. Measure for each variant:
   - p50/p95/p99 wall time;
   - timing breakdown;
   - peak RSS / allocation metrics where available;
   - compilation latency;
   - cache hit rate and recompilation count;
   - objective delta;
   - best epoch;
   - initial/final/best action delta;
   - target-book delta;
   - risk-result delta;
   - non-finite counter delta.

3. Deploy in shadow mode.
   - Continue sending current float64 eager decisions.
   - Run float32 compiled candidate on the identical state in parallel.
   - Do not execute candidate orders.
   - Classify each divergence:
     - parameter rounding only;
     - one-tick difference;
     - quantity-step difference;
     - qualitative side/ladder difference;
     - risk disagreement;
     - numerical failure.
   - Investigate every risk disagreement and every qualitative quote-side difference.

4. Controlled production rollout.
   - Enable via per-strategy flag:
     ```json
     {
       "tensor_dtype": "float32",
       "compile_rollout": true,
       "horizon_action_mode": "hold"
     }
     ```
   - Stages:
     1. observe-only;
     2. low notional / reduced order count;
     3. standard limits.
   - Roll back automatically to float64 eager if:
     - loss or gradient becomes non-finite;
     - compilation fails;
     - latency repeatedly violates the configured budget;
     - candidate differs materially from eager fallback;
     - risk result disagrees unexpectedly.

## Test coverage

Add tests for:

- float64 normalized versus legacy float64 parity;
- float32 versus float64 deterministic replay parity;
- large IRT balances and low-liquidity numerical stability;
- compiled versus eager output parity;
- compile failure falling back safely to eager;
- cached price-model tensor parameters;
- cache invalidation after an online model update;
- vectorized price-model output equivalence;
- independent / hold / linear-delta horizon-action behavior;
- preserved gradient flow from later horizons under hold mode;
- active versus inactive validity paths;
- no extra allocations/static tensor creation in the price-model hot path.

The safe implementation order is:

1. replay harness, parity contract, and timing facts;
2. dtype configuration and normalization;
3. price-model tensor parameter cache;
4. pure tensor rollout/objective extraction;
5. compile path and eager fallback;
6. invariant tensor caching/allocation reduction;
7. horizon `hold` mode;
8. validity and diagnostics specialization;
9. float32 compiled shadow deployment;
10. controlled live enablement with immediate float64-eager fallback.