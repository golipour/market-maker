# 13 — Glossary

| Term | Definition |
| --- | --- |
| **Actor** | The strategy instance. Identified by a name (e.g. `mm-btc-001`). Owns one or two venues. |
| **Bus** | The pub/sub backbone. Either in-process (`InMemoryEventBus`) or Redis streams. |
| **`ControllerDiagnostics`** | The telemetry payload published after every decision. Contains optimizer stats, validity/viability scores, and the decision hash. |
| **CE (certainty equivalent)** | Risk-adjusted return: `(1/β) · log E[exp(β · ΔW)]`. |
| **D(s)** | Cumulative depth at spread `s` from the sigmoid action. |
| **Effective depth** | `D(s) − D(0)`; the executable portion of the curve above the inside level. |
| **`EffectiveSigmoidMixtureAction`** | View of `SigmoidMixtureAction` whose depth methods return executable depth. |
| **EWMA** | Exponentially weighted moving average. Used for online model parameter updates. |
| **Fill** | A matched order. The fill size is `min(D(s) − D(0), V_side)` in the soft model. |
| **Frame** | A `StrategyInputFrame` or `JointStrategyFrame` produced by an assembler. |
| **Gate** | A boolean execution switch. Closed gates route to the preview bus, not the live order manager. |
| **Hurdle model** | A distribution with a discrete mass at zero and a continuous positive distribution. |
| **Joint** | A mode where the strategy optimizes both a local and a global venue's sigmoid action in one vector. |
| **`JointStrategyFrame`** | The bundle of input frame + joint ledger + open orders, used by the strategy runner. |
| **`JointTargetBook`** | The pair of local + global `VenueTargetBook` produced by the target book builder. |
| **`L, k, x0`** | The three sigmoid parameters per component: depth cap, steepness, and center. |
| **Ledger** | The canonical record of balances, open orders, and fills. Driven by the reconciler. |
| **Local venue** | The venue the actor is quoting. The optimizer's primary target. |
| **L-param decoder** | `materialize_side_torch` / `materialize_side_np`: maps a raw parameter vector to effective `(L, k, x0)`. |
| **MCE** | Monte Carlo Estimate. A sample-mean over `B` paths. |
| **MPC** | Model Predictive Control. The optimizer simulates `H` future steps under a model and chooses the action that maximizes the expected objective. |
| **`MarketEvent`** | A closed-window per-interval snapshot: trades + orderbook + aggregates. |
| **`NetDeltaRiskEngine`** | The risk gate. Validates proposed orders against ledger state and risk limits. |
| **`OptimizerResult`** | The output of `optimize`: the best action and its diagnostic trail. |
| **`OrderbookRelative`** | An orderbook in `(spread, quantity)` form, signed by side. |
| **Reservation skew** | A small spread shift that incentivizes inventory rebalancing. `r` shifts the action centers. |
| **`Reservation_r_offset`** | The same as `r` above; the variable name in code. |
| **Risk-adjusted return** | `CE` (certainty equivalent). The risk-neutral score the optimizer maximizes. |
| **Ruin probability** | In viability: the probability that the bot runs out of either side's margin over the horizon. The viability penalty is a smooth upper bound on this. |
| **Sigmoid action** | A `SigmoidMixtureAction`: sum of `K` sigmoids per side. |
| **`Soft fill`** | Differentiable fill function: `min(D(s) − D(0), V_side)` with a smooth min inside the optimizer. |
| **State** | The MPC's input: market data + ledger state + reference price + skew. |
| **Target book** | The venue-ready list of orders produced by the target book builder. |
| **Validity** | Book quality: tightness, presence, balance, uniformity. |
| **Viability** | Availability of margin to support the proposed exposure. |
| **Viability dual** | `θ`, the Lagrange multiplier for the viability constraint, updated by subgradient. |
| **VWAP** | Volume-weighted average price. `Σ price·qty / Σ qty`. |
| **Warm-start** | Initializing the next decision's raw parameter vector from the previous decision's vector. |
| **Window** | A closed interval `[t_start, t_end]` over which trades are aggregated. |
