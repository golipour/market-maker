# Model Lab — Experiment Plan 2

Extended research agenda for walk-forward flow and price model experiments on persisted Postgres data (e.g. Nobitex `USDT_IRT`, 30s intervals). This document restates the problem, evaluation contract, and exploration tracks **before implementation**. Use it to guide new notebooks and model implementations without modifying production code.

**Related docs:**

| Doc | Role |
|-----|------|
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | Original taxonomy, hyperparameter sweeps, tooling reference |
| [GUIDE.md](GUIDE.md) | Domain, production model formulas, quick start |
| [notebooks/production_models_walkforward.ipynb](notebooks/production_models_walkforward.ipynb) | Baseline walk-forward notebook (do not overwrite) |

**New experiments:** use a separate notebook (e.g. `notebooks/experiments_02_model_sweep.ipynb`).

---

## 1. Problem restatement

### 1.1 What we are modeling

Every `interval_ms` (e.g. 30,000 ms), the market-maker observes a discrete snapshot:

| Field | Meaning |
|-------|---------|
| `(buy_volume, sell_volume)` | Aggressive flow in the bucket (base lots, ≈ USDT for USDT_IRT) |
| `(mid, spread)` | Best-bid/ask-derived price and width (quote per base) |
| `orderbook` | Full depth snapshot (required for MOF / mechanical price transitions) |

We need **generative probabilistic models** for two families:

1. **Flow model** — joint distribution of `(V_buy, V_sell)` for the *next* interval, conditional on history.
2. **Price model** — how book shape (mid, spread, depth curves) transitions beyond what aggressive flow mechanically explains.

This lab evaluates those models **offline** on historical intervals in Postgres (`mm_events`, `mm_market_facts`).

### 1.2 Evaluation contract (strictly enforced)

There is **no segmented train/validation split**. Evaluation is **prequential** (predictive-sequential): one-step-forward prediction with online learning.

```
For each step t = 1..N:
  1. SCORE:  NLL_t = -log P(s_t | θ_{t-1})   ← predict BEFORE update
  2. UPDATE: θ_t = online_update(θ_{t-1}, s_{t-1}, s_t)
```

| Term | Meaning |
|------|---------|
| `θ_{t-1}` | Model state after observing steps `0..t-1` (hyperparameters fixed for the run) |
| `s_t` | Observation at step `t` |
| `NLL_t` | Negative log-likelihood of `s_t` under the predictive distribution |

**Critical rule (matches production):** scoring at step `t` uses **θ_{t-1}** only. The model must not peek at `s_t` before predicting it. After scoring, the model ingests the transition and moves to **θ_t**.

The running sum of NLLs is a proper scoring rule. **Lower mean NLL = better predictive model** (for the same observation space and scoring rule).

### 1.3 What this lab is NOT

- Not batch ML with random train/test shuffles (leaks future information).
- Not PnL or fill simulation (that is the MPC optimizer's job).
- Not hyperparameter tuning of the MPC objective — only the **flow/price generative models**.

### 1.4 Success metrics

| Aggregate | Formula | Use |
|-----------|---------|-----|
| `mean_nll` | \(\frac{1}{T}\sum_t \text{NLL}_t\) | **Primary leaderboard metric** — lower is better |
| `median_nll` | median of per-step NLL | Robust to burst intervals with tiny \(P\) |
| `total_nll` | sum of NLL | Comparable across runs with same \(T\) |

**Compare within family:** flow NLL and price NLL are not on the same scale. Compare `flow:*` against `flow:*`, and `price:*` against `price:*`.

### 1.5 Why non-stationarity matters

Current production models are adaptive **by design** — EWMA and RLS forget old observations to track regime changes.

**Core tension:**

> **Memory vs agility:** long memory (high decay / slow forgetting) fits a stationary regime better; short memory adapts faster but is noisier.

If observations can be represented in an **approximately stationary** way (e.g. mid returns instead of levels, volume normalized by recent activity), richer fixed-weight or small neural models may compete with or beat adaptive EWMA/RLS — because there is less distribution shift to track.

**Important:** a model may **improve over time** as θ accumulates information. Report NLL vs step index (cold start vs warm) in addition to global mean NLL.

---

## 2. Reference dataset (Nobitex USDT_IRT, 30s)

Example characteristics from a recent ~48h live run (~2,882 steps at `interval_ms=30000`):

| Property | Value | Implication |
|----------|-------|-------------|
| Zero-volume buckets | ~59% | Hurdle / zero-inflation essential for flow |
| P90 / P99 volume | ~861 / ~3,331 USDT | Heavy-tailed — tail model matters |
| Mid range | ~158,007–161,300 IRT | Slow drift; non-stationary in levels |
| Mid std | ~733 IRT (~0.46%) | Low volatility in that window |
| corr(volume, market_prob) | ~−0.42 | Clear quiet vs active regime structure |
| Intraday volume | Highly variable by hour | Non-stationarity within the window |
| Series length | ~2,882 steps | Tight budget for high-parameter models |

Typical end-of-run production params (for context only):

- **Flow (hurdle lognormal):** `p0_buy` ~0.15, `p0_sell` ~0.24; buy sizes more variable (`sigma_buy` ~1.6) than sell (`sigma_sell` ~0.65).
- **Price (conditional mechanical mixture):** `market_prob` ~0.20; negative `gate_beta` on log-flow; refresh mid drift ~+0.95 ticks/30s.

---

## 3. Exploration tracks

Experiments are grouped into three tracks. Each track can include **multiple model families** and **hyperparameter grids within a family**.

### Track A — Flow models (`V_buy`, `V_sell`)

**Production baseline:** `flow:hurdle_lognormal` — independent buy/sell sides, hurdle at zero, log-normal when \(V > 0\), EWMA on `p0`, `μ`, `σ`.

#### A1. Hyperparameter sweeps (existing families)

Built-in kinds: `hurdle_lognormal`, `hurdle_gamma`, `hurdle_weibull`, `gamma`, `weibull`.

| Hyperparameter | Role | Sweep ideas |
|----------------|------|-------------|
| `decay` | EWMA forgetting (γ) | 0.80, 0.90, 0.95, 0.99 — effective window ≈ 1/(1−γ) steps |
| `init_p0` | Prior P(zero) | Higher for sparse markets (faster warm-start) |
| Tail family | Distribution when V > 0 | log-normal vs gamma vs weibull vs log-Student-t |
| Per-side decay | Separate buy/sell memory | `decay_buy`, `decay_sell` if sides differ |

**Hypothesis:** optimal `decay` trades off cold-start noise vs regime-shift lag; may differ by hour-of-day slice.

#### A2. Joint buy/sell (remove independence)

Current: \(P(V_{\text{buy}}, V_{\text{sell}}) = P(V_{\text{buy}})\,P(V_{\text{sell}})\).

Alternatives:

- **Conditional:** \(P(V_{\text{sell}} \mid V_{\text{buy}})\) via online regression on log-volumes when both > 0.
- **Copula:** marginals + Gaussian/Clayton copula; online EWMA of rank correlation.
- **Bivariate hurdle:** explicit mass on (0,0), (>0,0), (0,>0), (>0,>0).

**Hypothesis:** joint density is sharper during information events; may not help on mostly-zero buckets.

#### A3. State-conditioned flow

Condition hurdle gate or (μ, σ) on observable state at \(t-1\):

- Bid-ask spread (illiquidity → lower flow)
- Book imbalance \((D_{\text{bid}} - D_{\text{ask}}) / (D_{\text{bid}} + D_{\text{ask}})\)
- Microprice vs mid divergence
- Recent flow momentum \(\sum (V_{\text{buy}} - V_{\text{sell}})\) over last \(k\) steps

**Implementation sketch:** logistic gate on features + EWMA/online SGD; continuous params still EWMA on log-volume when \(V > 0\).

#### A4. Count / tail alternatives

- **Negative Binomial / ZINB** — overdispersion and zero-inflation for discrete lots.
- **Discrete Pareto / Zipf** — power-law tails for extreme buckets (can dominate mean NLL if mis-modeled).

**Requirement:** tractable PMF for NLL at each step.

#### A5. Adaptive forgetting

- **Kalman-style** state on log-volume mean/variance; process noise from prediction error.
- **CUSUM / alarm:** reset or accelerate forgetting on sustained prediction errors.
- **RLS with forgetting** on flow parameters (analogous to price model market branch).

---

### Track B — Price models (mid, spread, shape)

**Production baseline:** `price:conditional_mechanical_mixture` — market vs refresh mixture gated on log-flow; RLS market branch; EWMA refresh branch with Student-t tails.

**Alternative baseline:** `price:residual` — MOF VWAP residuals with per-side EWMA Gaussian.

#### B1. Hyperparameter sweeps (existing)

**Conditional mechanical mixture:**

| Hyperparameter | Role | Sweep ideas |
|----------------|------|-------------|
| `decay` | RLS/EWMA rate | 0.95, 0.99, 0.995 |
| `reference_vwap_vol` | Fixed-depth VWAP | 300–2000 USDT |
| `shape_alpha` | Depth-shape power | 0.3, 0.5, 0.7, 1.0 |
| `student_df` | Refresh-branch tails | 3, 5, 10, ∞ (Gaussian) |
| `initial_sigma` | Prior mid uncertainty | Data-driven from first N steps |

**Residual:**

| Hyperparameter | Role |
|----------------|------|
| `residual_decay` | EWMA on \(e_{\text{buy}}, e_{\text{sell}}\) |
| `residual_sigma` | Initial σ floor |
| Asymmetric decay | Separate buy/sell residual memory |

See [GUIDE.md §2](GUIDE.md#2-current-production-models-examples) for exact formulas.

#### B2. Mid-only and simpler targets

- **Gaussian random walk on Δmid** — baseline `baseline:mid_random_walk`.
- **AR(p) on Δmid** — online RLS/EWMA on lags.
- **Flow-conditioned Δmid** — \(\Delta\text{mid} \sim \mathcal{N}(\beta_0 + \beta_1 \cdot \text{imbalance} + \beta_2 \log(1+V_{\text{total}}), \sigma^2)\).
- **Student-t noise** on mid changes for heavy tails.

#### B3. GARCH-style volatility

Online GARCH(1,1) on mid innovations:

\[
\sigma^2_t = \omega + \alpha \varepsilon^2_{t-1} + \beta \sigma^2_{t-1}
\]

Captures volatility clustering after large flow events.

#### B4. Autocorrelated residuals

Current models often treat innovations as i.i.d. after adaptation. **AR(1) on residuals:**

\[
\text{residual}_t \sim \mathcal{N}(\phi \cdot \text{residual}_{t-1}, \sigma^2)
\]

with φ updated online — tests mean-reversion vs momentum in book reshaping.

#### B5. Regime-switching (2-state online HMM)

Explicit quiet vs active states with persistent hidden state (not only flow at \(t\)):

- **Quiet:** small mid innovation, tight spread.
- **Active:** larger innovation, possible mean-reversion vs mechanical path.

Compare to conditional mixture gate, which is flow-instantaneous rather than Markov on regime.

#### B6. Microprice as target

\[
\text{microprice} = \text{mid} + \frac{\text{spread}}{2} \cdot \frac{D_{\text{ask}} - D_{\text{bid}}}{D_{\text{ask}} + D_{\text{bid}}}
\]

Model \(\Delta\text{microprice}\) instead of raw mid — may be more stationary and informative short-term.

---

### Track C — Stationarity + neural / sequence models

Only pursue if Track A/B show that **representation** reduces prequential NLL degradation over long windows.

#### C1. Stationarity transforms

| Transform | Definition | Rationale |
|-----------|------------|-----------|
| Log-return | \(r_t = \log(\text{mid}_t / \text{mid}_{t-1})\) | Level drift removed |
| Spread-normalized Δmid | \((\text{mid}_t - \text{mid}_{t-1}) / \text{spread}_{t-1}\) | Scale-invariant move |
| Normalized volume | \(v_t / \text{EWMA}(\|v\|)\) | Activity-level invariance |
| Imbalance | Bounded in [-1, 1] | Local supply/demand without level drift |
| Flow direction | \((V_{\text{buy}} - V_{\text{sell}}) / (V_{\text{buy}} + V_{\text{sell}})\) when total > 0 | Net pressure |

**Test:** compare adaptive EWMA on raw vs transformed targets under the same walk-forward contract.

#### C2. Small sequence models (conditional on C1)

With ~3k steps, large models are risky. Options if representations are stable:

| Approach | Idea | Online learning |
|----------|------|-----------------|
| Small GRU/RNN | Window of (V, spread, imbalance) → hurdle/Gaussian params | SGD on last W steps each step |
| Small Transformer | Attention over W-step context | Same; W ≈ 30–50 |
| Pre-train + fine-tune | Months of history → few online params | Meta-learning bias |

**Density head required:** output must define a proper \(\log P(\text{obs} \mid \theta)\) for NLL (e.g. hurdle log-normal parameters).

**Risk:** many parameters → high estimation variance on short windows; EWMA may win unless pre-training + good features reduce shift.

---

## 4. Experiment structure

### 4.1 Shared harness

- **Data:** e.g. `nobitex`, `USDT_IRT`, `interval_ms=30000`, `--since 48h` (or longer if available).
- **Source:** `events` or `auto` for price models (full orderbook); `market_facts` OK for flow-only rough tests.
- **Contract:** `research.model_lab.core.walk_forward.run_walk_forward` — score then update.
- **Primary metric:** `mean_nll` per family.
- **Secondary:** `median_nll`, NLL time series (cold vs warm), NLL by volume quintile.

### 4.2 Suggested groups (effort vs payoff)

| Group | Contents | Expected payoff | Effort |
|-------|----------|-----------------|--------|
| **A** | Hyperparameter sweeps on existing models | High ROI | Very low |
| **B** | New flow distributions (log-Student-t, NB, joint buy/sell) | Moderate | Low |
| **C** | State-conditioned flow gate | Moderate | Low–medium |
| **D** | GARCH, AR residual, HMM for price | Moderate | Medium |
| **E** | Stationarity preprocessing | Uncertain, foundational | Low |
| **F** | Small RNN/GRU + sliding window | High if (E) succeeds | High |
| **G** | Pre-trained transformer + online fine-tune | Long-shot; needs long history | Very high |

### 4.3 Example leaderboard shape

```
Family: flow
Model                              mean_nll   median_nll   n_scored
baseline:zero_flow                  ...
flow:hurdle_lognormal (production)  ...
flow:hurdle_lognormal (decay=0.99)  ...
flow:hurdle_log_student_t           ...
flow:joint_bivariate                ...

Family: price
Model                              mean_nll   median_nll   n_scored
baseline:mid_random_walk            ...
price:residual (production)         ...
price:conditional_mechanical_mixture ...
price:garch_mid                     ...
price:hmm_2state                    ...
```

### 4.4 Recommended execution order

1. Establish baseline NLL for all existing models on live data.
2. Sweep `decay` (and key price hyperparameters) — often largest lever.
3. Add log-Student-t flow (wrap existing production class if present).
4. Test stationarity transforms; compare fixed vs adaptive on same scoring rule.
5. Joint buy/sell flow model.
6. AR(1) and GARCH mid models for price.
7. 2-state HMM price model.
8. Regime-stratified NLL (zero-vol vs high-vol quintiles).
9. Small RNN only if steps 4 and 8 justify complexity.

**Notebooks:**

| Notebook | Purpose |
|----------|---------|
| `production_models_walkforward.ipynb` | Baseline reproduction (do not overwrite) |
| `experiments_02_flow.ipynb` | Track A experiments |
| `experiments_03_price.ipynb` | Track B experiments |
| `experiments_04_neural.ipynb` | Track C (only after stationarity tests) |

---

## 5. Interpretation caveats

### 5.1 Global NLL vs regimes

A model that excels on quiet (zero-flow) steps may beat one that excels on bursts — even if the latter is more useful for MPC when trades occur. **Report NLL stratified by volume quintile.**

### 5.2 Performance over time

Early steps (cold start) inflate NLL. Plot NLL vs step index to compare learning speed and regime-shift sensitivity. A model with higher early NLL but lower late NLL may still be preferable for long runs.

### 5.3 Statistical significance

~3k paired differences in `NLL_A - NLL_B` — use paired t-tests or bootstrap CIs before claiming a winner.

### 5.4 Family boundaries

Never rank flow NLL against price NLL.

### 5.5 NLL vs MPC utility

MPC uses models for **rollout sampling**, not only density scoring. Best NLL model must still produce valid, sampleable paths (hurdle zeros, mixture branches, etc.).

---

## 6. Mapping to production

| Lab | Production |
|-----|------------|
| `IntervalStep` | `MarketEvent` / frame local event |
| `run_walk_forward` | `score_frame_models` + `model.update` in `MpcStrategyEngine` |
| Hyperparameters | `controller.local_*_model` in runtime JSON |
| Learned state | `get_params()` after each update; persisted in `StrategyDecision` / `mm_model_score_facts` |
| Prequential NLL | Same contract as live `mm_model_score_facts` |

Production reference implementations:

- Flow: `mm/mpc/market_order_flow_torch.py`
- Price: `mm/mpc/price_torch.py`
- Scoring: `mm/models/scoring.py`, `research/model_lab/core/scoring.py`

---

## 7. Tooling reference

```bash
# CLI compare (multiple specs)
python -m research.model_lab.experiments.compare \
  --exchange nobitex --symbol USDT_IRT --interval-ms 30000 \
  --since 48h \
  --models flow:hurdle_lognormal,price:residual,price:conditional_mechanical_mixture,baseline:zero_flow

# List built-in models
python -m research.model_lab.experiments.compare --list-models

# Smoke tests (no DB)
pytest research/model_lab/tests/test_walk_forward.py -q
```

Outputs: `research/model_lab/outputs/*_summary.json` (leaderboard), optional `*_scores.jsonl`.

Data access: Postgres via `MM_DATABASE_URL` (see [GUIDE.md §4](GUIDE.md#4-prerequisites)) or MCP postgres server when configured.

---

## 8. Open questions (Plan 2 focus)

1. Does longer history (7d+) change the optimal `decay` or favor heavier tails?
2. Is joint buy/sell worth the complexity on USDT_IRT specifically?
3. Can spread-normalized or log-return mid targets beat full book mixture/residual NLL?
4. At what context length (if any) does a small GRU beat EWMA hurdle on walk-forward NLL?
5. Do winners on zero-volume quintiles differ from winners on P90+ volume quintiles?
6. Does NLL improve monotonically with step index for adaptive models, and does that differ by `decay`?

Document results of each experiment wave in `outputs/` and summarize findings in a dated `outputs/REPORT_*.md`.

---

## 9. Implementation checklist (when coding starts)

- [ ] Baseline leaderboard on Nobitex USDT_IRT 30s (`compare.py` or new notebook).
- [ ] Track A1: decay grid for `flow:hurdle_lognormal`.
- [ ] Track B1: decay / `reference_vwap_vol` grid for price models.
- [ ] Stratified NLL by volume quintile (shared analysis helper).
- [ ] NLL vs step index plots (cold start / warm performance).
- [ ] New `OnlineModel` implementations registered in `models/registry.py` (no production edits).
- [ ] Unit tests in `tests/test_walk_forward.py` for each new model.
- [ ] Paired significance test between top-2 candidates per family.
