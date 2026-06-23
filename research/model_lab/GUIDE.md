# Model Lab — Research Guide

This folder is a **standalone research sandbox** for fitting and comparing **market-flow** and **price** models on data already persisted by the market-maker runtime (Postgres `mm_events` / `mm_market_facts`).

It is designed so an agent **with no prior context** can run experiments, add models, and pick winners by **walk-forward negative log-likelihood (NLL)**.

**Read this first if you are new:**
1. [§1 Problem details](#1-problem-details-what-you-are-modeling) — domain, inputs, outputs, success metric
2. [§2 Current production models (examples)](#2-current-production-models-examples) — exact formulas + config used in the live bot
3. [§4 Quick start](#5-quick-start) — run your first benchmark

---

## 1. Problem details (what you are modeling)

### 1.1 Domain: interval market making

The parent repo is an **event-bus market maker** for crypto pairs (e.g. Nobitex `USDT_IRT`). Every `interval_ms` milliseconds (10s, 30s, …) the runtime:

1. Pulls exchange orderbook + trades
2. Builds a **`MarketEvent`** — aggregated buy/sell volume, mid, spread, full orderbook snapshot
3. Runs an **MPC optimizer** that simulates future orderbooks and picks bid/ask ladders
4. Optionally places live limit orders

The MPC simulator needs **predictive models** for two quantities on each interval:

| Model family | Question it answers | Used in MPC rollouts for |
|--------------|-------------------|--------------------------|
| **Market flow** | How much aggressive buy/sell volume arrives in the next interval? | Consuming depth on the simulated book |
| **Price** | How does the book's prices/spread/shape move, beyond what flow alone explains? | VWAP paths, reservation price, spread risk |

This lab evaluates those models **offline** on historical intervals stored in Postgres.

### 1.2 One observation = one interval step

Each **interval step** (`IntervalStep` in `core/types.py`) is one row of market state:

| Field | Meaning | Example (Nobitex USDT_IRT) |
|-------|---------|----------------------------|
| `ts` | End of interval (UTC) | `2026-06-22T11:38:00Z` |
| `interval_ms` | Bucket width | `30000` (30 seconds) |
| `buy_volume` | Aggressive buy volume in bucket (base lots, ≈ USDT) | `170.5` |
| `sell_volume` | Aggressive sell volume in bucket | `129.1` |
| `mid` | `(best_bid + best_ask) / 2` in quote per base (IRT per USDT) | `159719` |
| `spread` | `best_ask - best_bid` | `139` IRT |
| `orderbook` | Full depth snapshot (bids/asks with price, qty) | Required for price models |

**Typical market character (USDT_IRT, 30s, from recent runs):**
- ~59% of intervals have **zero** trade volume (sparse, bursty flow)
- When volume arrives, it is heavy-tailed (P90 ≈ 860 USDT, P99 ≈ 3,300 USDT per 30s)
- Mid price changes slowly (~700 IRT std over 48h on ~160k mid)
- Weak linear correlation between mid and volume (|ρ| < 0.05) — volume switches **regime**, not level

### 1.3 The prediction task (step by step)

At each transition **previous → current** (interval `t-1` to `t`):

```
History:  [s0, s1, …, s_{t-1}]  ──trained on──►  model parameters θ_{t-1}
                                                      │
                                                      ▼
Target:   observe s_t  ──score──►  NLL_t = -log P(s_t | θ_{t-1})
                                                      │
                                                      ▼
Update:   θ_t = Update(θ_{t-1}, s_{t-1}, s_t)
```

**Critical rule (matches production):** scoring at step `t` uses **θ_{t-1}** only. The model must not peek at `s_t` before predicting it. After scoring, the model ingests the transition and moves to **θ_t**.

This is **prequential / walk-forward** evaluation — the only honest metric for online models.

### 1.4 Success metric: negative log-likelihood (NLL)

For each scored step:

\[
\text{NLL}_t = -\log P(\text{observation}_t \mid \theta_{t-1})
\]

| Aggregate | Formula | How to use |
|-----------|---------|------------|
| `mean_nll` | \(\frac{1}{T}\sum_t \text{NLL}_t\) | **Primary leaderboard metric** — lower is better |
| `median_nll` | median of per-step NLL | Robust to burst intervals with tiny \(P\) |
| `total_nll` | sum of NLL | Comparable across runs with same \(T\) |

**Compare within family:** flow NLL and price NLL are not on the same scale. Compare `flow:*` models against each other, and `price:*` models against each other.

### 1.5 What this lab is NOT

- Not batch ML train/test with random shuffles (that leaks future information)
- Not PnL or fill simulation (that's the MPC optimizer's job)
- Not hyperparameter tuning of the MPC objective — only the **flow/price generative models**

### 1.6 Reference deployment (example bot)

The example runtime config (`mm/configs/market_maker.runtime.example.json`) runs a Nobitex `USDT_IRT` bot with:

```json
"interval_ms": 30000,
"local_market_flow_model": { "kind": "hurdle_lognormal", "decay": 0.95, ... },
"local_price_model": { "kind": "conditional_mechanical_mixture", "reference_vwap_vol": 1000.0, ... }
```

Your Postgres data from a live run uses these model kinds unless you changed the preset.

---

## 2. Current production models (examples)

These are the models **currently wired into the live runtime**. The lab wraps them via `models/adapters.py`. Use them as **reference implementations** when building new models.

Implementation files:
- Flow: `mm/mpc/market_order_flow_torch.py`
- Price: `mm/mpc/price_torch.py`
- Scoring: `mm/models/scoring.py`
- Config: `mm/configs/market_maker.runtime.example.json` → `strategy_presets.*.controller`

---

### 2.1 Market flow model: `hurdle_lognormal` (production default)

**Config example (local venue):**

```json
"local_market_flow_model": {
  "kind": "hurdle_lognormal",
  "decay": 0.95,
  "init_p0": 0.3,
  "init_mu": 0.0,
  "init_sigma": 1.0
}
```

**Lab spec:** `flow:hurdle_lognormal`

**Generative model (per side, buy and sell independent):**

Let \(V\) be integer volume in one interval (buy or sell side).

\[
P(V = 0) = p_0
\]
\[
P(V = k \mid V > 0) = (1 - p_0) \cdot \text{DiscreteLogNormalPMF}(k; \mu, \sigma), \quad k \geq 1
\]

where \(\log V \sim \mathcal{N}(\mu, \sigma^2)\) before rounding.

**Online parameter updates (EWMA, decay \(\gamma =\) `decay`):**

After observing volume \(x\) for a side:

```
z = 1 if x == 0 else 0
p0 ← γ·p0 + (1-γ)·z                    # tracks P(zero)

if x > 0:
    log_x = log(x)
    S_log  ← γ·S_log  + log_x          # EWMA of log-volume
    S2_log ← γ·S2_log + log_x²
    W      ← γ·W + 1
    μ ← S_log / W
    σ ← sqrt(max(ε², S2_log/W - μ²))
```

**Scoring (lab + runtime):** independent sides,

\[
\log P = \log P_{\text{buy}}(v_{\text{buy}}) + \log P_{\text{sell}}(v_{\text{sell}})
\]

Code: `mm/models/scoring.py` → `score_market_flow()`.

**Typical trained values (Nobitex USDT_IRT, 30s, ~48h):**

| Param | Buy | Sell | Interpretation |
|-------|-----|------|----------------|
| `p0` (P zero) | ~0.15 | ~0.24 | Sell side more often zero in a bucket |
| `mu` | ~5.62 | ~6.24 | Log-scale of size when trade happens |
| `sigma` | ~1.61 | ~0.65 | Buy sizes more variable |

Implied median size when \(V>0\): \(e^\mu \approx 276\) USDT (buy), \(\approx 512\) USDT (sell).

**Other flow kinds in lab:** `hurdle_gamma`, `hurdle_weibull`, `gamma`, `weibull` — same hurdle structure, different continuous distribution before rounding.

---

### 2.2 Price model: `residual` (alternative local price model)

**Config example:**

```json
"local_price_model": {
  "kind": "residual",
  "residual_decay": 0.95,
  "residual_sigma": 50.0
}
```

**Lab spec:** `price:residual`  
**Class:** `TorchResidualPriceModel` (`mm/mpc/price_torch.py`)

**Idea:** split price movement into (a) what aggressive flow alone would do to the book (MOF path), and (b) a **residual** from limit-order refresh, cancellations, and other effects.

**Step transition** from orderbook at \(t-1\) to \(t\):

1. **Realized VWAP** — VWAP on the **current** book at depth \(D\) (ticks/lots)
2. **MOF-only VWAP** — apply observed buy/sell flow to the **previous** book, then compute VWAP on that synthetic book (no replenishment)

\[
e_{\text{buy}} = \text{VWAP}^{\text{real}}_{\text{buy}} - \text{VWAP}^{\text{MOF}}_{\text{buy}}
\]
\[
e_{\text{sell}} = \text{VWAP}^{\text{real}}_{\text{sell}} - \text{VWAP}^{\text{MOF}}_{\text{sell}}
\]

Code: `mm/pricing/reference_prices_ticks.py` → `reference_prices_from_transition_ticks()`.

**EWMA on residuals** (per side, \(\gamma =\) `residual_decay`):

\[
\mu_{\text{side}} \leftarrow \gamma \mu_{\text{side}} + (1-\gamma)\, e_{\text{side}}
\]
\[
\sigma^2_{\text{side}} \leftarrow \gamma \sigma^2_{\text{side}} + (1-\gamma)\, (e_{\text{side}} - \mu_{\text{side}})^2
\]

**Scoring (lab):** Gaussian per side,

\[
\log P = \log \mathcal{N}(e_{\text{buy}}; \mu_{\text{buy}}, \sigma_{\text{buy}}) + \log \mathcal{N}(e_{\text{sell}}; \mu_{\text{sell}}, \sigma_{\text{sell}})
\]

Code: `research/model_lab/core/scoring.py` → `score_residual_transition()`.

**MPC rollout sampling:**

\[
\varepsilon_{\text{buy}} \sim \mathcal{N}(\mu_{\text{buy}}, \sigma_{\text{buy}}), \quad
\varepsilon_{\text{sell}} \sim \mathcal{N}(\mu_{\text{sell}}, \sigma_{\text{sell}})
\]

added to MOF-simulated VWAP paths.

**Note:** `vwap_depth` / `reference_vwap_vol` are **not** config keys for residual — depth \(D\) is derived from inventory in production. In the lab, default `reference_vwap_vol=1000` sets `_target_depth_lots`.

---

### 2.3 Price model: `conditional_mechanical_mixture` (production default for local price)

**Config example (matches live Nobitex preset):**

```json
"local_price_model": {
  "kind": "conditional_mechanical_mixture",
  "reference_vwap_vol": 1000.0,
  "shape_alpha": 0.5,
  "decay": 0.99,
  "initial_sigma": 50.0,
  "initial_log_sigma": 0.25,
  "student_df": 3.0,
  "point_mode": "sample"
}
```

**Lab spec:** `price:conditional_mechanical_mixture`  
**Class:** `TorchConditionalMechanicalMixturePriceModel` (`mm/mpc/price_torch.py`)

**Idea:** each interval, book changes decompose into **market-driven** (explained by mechanical flow impact) vs **refresh** (limit orders, book reshaping). A **sigmoid gate** on total flow picks the mixture weight.

**Book shape features** at fixed reference depth \(V_{\text{ref}} =\) `reference_vwap_vol`:

\[
\text{buy\_shape} = \text{buy\_vwap} - \text{mid} - \tfrac{1}{2}\text{spread}
\]
\[
\text{sell\_shape} = \text{mid} - \text{sell\_vwap} - \tfrac{1}{2}\text{spread}
\]

**Component deltas** \(c \in \{\text{mid}, \log\text{spread}, \log\text{buy\_shape}, \log\text{sell\_shape}\}\):

\[
\Delta^{\text{obs}}_c = \text{feature}_c(t) - \text{feature}_c(t-1)
\]
\[
\Delta^{\text{mech}}_c = \text{feature}_c(\text{MOF book from } t-1) - \text{feature}_c(t-1)
\]

**Mixture gate:**

\[
q_{\text{market}} = \sigma\!\left(\beta_0 + \beta_1 \cdot \min(\log(1 + V_{\text{buy}} + V_{\text{sell}}),\, 12)\right)
\]

**Market branch** (linear in mechanical delta, per component):

\[
\mathbb{E}[\Delta^{\text{obs}}_c \mid \text{market}] = \beta^c_0 + \beta^c_1 \cdot \Delta^{\text{mech}}_c
\]

Updated via weighted RLS; Gaussian innovation variance per component.

**Refresh branch:** EWMA mean per component with Student-t innovations (`student_df`).

**Scoring (lab):** mixture log-likelihood,

\[
\log P = \log\!\Big( e^{\log q_{\text{market}} + \ell_{\text{market}}} + e^{\log q_{\text{refresh}} + \ell_{\text{refresh}}} \Big)
\]

Code: `research/model_lab/core/scoring.py` → `score_conditional_mixture_transition()`.

**Typical trained values (Nobitex USDT_IRT, 30s, end of ~48h run):**

| Param | Value | Reading |
|-------|-------|---------|
| `gate_beta` | `[-0.033, -0.231]` | Higher flow → **lower** P(market-driven) |
| `market_prob` | ~0.20 | Only ~20% of recent steps classified as flow-explained |
| `market_beta.mid` | `[0.40, -0.20]` | Mechanical mid move **negatively** predicts observed mid move |
| `refresh_mean.mid` | ~+0.95 ticks/30s | Refresh regime drifts mid up slowly |
| `market_sigma.mid` | ~50 ticks | Near `initial_sigma` floor |

**Learned regime structure:** corr(volume, market_prob) ≈ **−0.42** — quiet intervals look like refresh; busy intervals look like flow-driven impact (with mean-reversion in mid).

---

### 2.4 Global models (not in default lab CLI, but same code)

The example preset also configures **global** (liquidity/reference venue) models:

```json
"global_market_flow_model": { "kind": "hurdle_lognormal", "decay": 0.95, ... },
"global_price_model": {
  "kind": "conditional_mechanical_mixture",
  "reference_vwap_vol": 100.0,
  ...
}
```

To benchmark global venue data, load a second exchange's intervals and register a model — or extend `compare.py`. Formulas are identical; only the input stream differs.

---

### 2.5 Baseline models (sanity checks)

| Lab spec | Purpose | When it wins |
|----------|---------|--------------|
| `baseline:zero_flow` | Always predict \(V=0\) | Very sparse markets (>50% zero-volume buckets) |
| `baseline:mid_random_walk` | EWMA Gaussian on \(\Delta\text{mid}\) only | Price moves without using flow/book structure |

Always include baselines in comparisons. If your model cannot beat `baseline:zero_flow` on flow NLL, it is not adding value.

---

### 2.6 Reproduce production defaults in the lab

```bash
# Flow — matches mm/configs/market_maker.runtime.example.json
python -m research.model_lab.experiments.compare \
  --models 'flow:hurdle_lognormal{"decay":0.95,"init_p0":0.3,"init_mu":0.0,"init_sigma":1.0}'

# Price — production conditional mixture
python -m research.model_lab.experiments.compare \
  --models 'price:conditional_mechanical_mixture{"reference_vwap_vol":1000.0,"decay":0.99,"initial_sigma":50.0,"student_df":3.0}'

# Price — residual alternative
python -m research.model_lab.experiments.compare \
  --models 'price:residual{"residual_decay":0.95,"residual_sigma":50.0}'
```

Defaults in `models/registry.py` mirror the example preset unless overridden by inline JSON.

---

## 3. What this lab implements (summary)

The live runtime updates models incrementally each interval and scores predictions using **pre-update parameters** (see `mm/models/scoring.py`). This lab reproduces that contract offline:

1. Load time-ordered interval steps (30s, 10s, …).
2. For each transition `previous → current`:
   - **Score** step `current` with parameters learned from all prior steps only.
   - **Update** the model with the transition.
3. Aggregate **mean/median NLL** across scored steps.

**Lower mean NLL = better predictive model** (for the same observation space and scoring rule).

---

## 4. Prerequisites

From the **repository root** (Python **3.10+**, same as the main `mm` package):

```bash
pip install -r requirements.txt
cp .env.example .env   # if not done already
```

Environment:

| Variable | Purpose |
|----------|---------|
| `MM_DATABASE_URL` | Postgres with `mm_events` and/or `mm_market_facts` |
| `.env` at repo root | Loaded automatically via `mm.env_loader` |

Postgres must contain market data for your symbol, e.g. from running:

```bash
docker compose up -d
python -m mm.runtime --config mm/configs/market_maker.runtime.example.json
# or the decomposed stack (ingestor + persistence)
```

---

## 5. Quick start

```bash
# List built-in model specs
python -m research.model_lab.experiments.compare --list-models

# Compare default model set on Nobitex USDT_IRT, 30s intervals, last 48h
python -m research.model_lab.experiments.compare \
  --exchange nobitex \
  --symbol USDT_IRT \
  --interval-ms 30000 \
  --since 48h \
  --models flow:hurdle_lognormal,price:residual,price:conditional_mechanical_mixture,baseline:zero_flow
```

Outputs land in `research/model_lab/outputs/`:

- `{timestamp}_summary.json` — leaderboard
- optional `{timestamp}_{model}_scores.jsonl` with `--write-scores`

---

## 6. Folder layout

```
research/model_lab/
  GUIDE.md                 ← you are here
  README.md                ← one-line pointer
  core/
    types.py               ← IntervalStep, StepScore
    protocol.py            ← OnlineModel interface
    walk_forward.py        ← evaluation loop
    scoring.py             ← NLL helpers (aligned with runtime)
    metrics.py             ← summarize_scores()
  data/
    loader.py              ← Postgres → IntervalStep list
  models/
    adapters.py            ← wrappers around production torch models
    baselines.py           ← simple benchmarks
    registry.py            ← build_model("flow:hurdle_lognormal")
  experiments/
    compare.py             ← CLI entry point
    configs/default.json   ← example experiment config
  tests/
    test_walk_forward.py   ← synthetic smoke tests (no DB)
  outputs/                 ← gitignored results
```

---

## 7. Evaluation contract (important)

### Walk-forward loop (`core/walk_forward.py`)

For steps `[s0, s1, s2, …, sN]`:

| Index | Action |
|-------|--------|
| 0 | No score (no previous step). Model may warm-start. |
| 1..N | Score `si` using params after updates on `s0..s(i-1)`, then `update(s(i-1), si)` |

This matches production: at frame `t`, `score_frame_models` uses **pre-update** params, then the engine calls `model.update`.

### What is being predicted?

| Family | Target observation | Scoring function |
|--------|-------------------|------------------|
| **flow** | `(buy_volume, sell_volume)` integer lots in the interval | `mm.models.scoring.score_market_flow` |
| **price:residual** | VWAP residuals `e_buy`, `e_sell` after MOF-only book evolution | `core/scoring.score_residual_transition` |
| **price:conditional_mechanical_mixture** | Mixture NLL on component deltas (mid, log-spread, log-shapes) | `core/scoring.score_conditional_mixture_transition` |
| **baseline:mid_random_walk** | `delta_mid` Gaussian | `score_local_mid_delta` |

---

## 8. Data sources

`data/loader.py` supports:

| `source` | Table | Orderbook depth | Use for |
|----------|-------|-----------------|---------|
| `auto` (default) | `mm_events` MarketEvent, else `mm_market_facts` | Full book when available | All models |
| `events` | `mm_events` | Full JSON orderbook | Price models (required) |
| `market_facts` | `mm_market_facts` | Best bid/ask only (synthetic 1-lot book) | Flow models; rough price baselines |

**Price models that use MOF/mechanical transitions need full orderbook depth.** Prefer `source=events` or `auto`.

Query filters:

- `--exchange`, `--symbol`, `--interval-ms`
- `--since 48h` | `7d` | ISO timestamp
- `--until` (default: now UTC)

---

## 9. Built-in models (catalog)

Full formulas and production config examples are in **[§2 Current production models](#2-current-production-models-examples)**.

Run `python -m research.model_lab.experiments.compare --list-models`.

### Flow (`family:flow`)

| Spec | Production class | Key params |
|------|------------------|------------|
| `flow:hurdle_lognormal` | `TorchHurdleLogNormalModel` | `decay`, `init_p0`, `init_mu`, `init_sigma` |
| `flow:hurdle_gamma` | `TorchHurdleGammaModel` | same |
| `flow:hurdle_weibull` | `TorchHurdleWeibullModel` | same |
| `flow:gamma` | `TorchDiscreteGammaModel` | same |
| `flow:weibull` | `TorchWeibullModel` | same |

Default config in `models/registry.py` → `DEFAULT_FLOW_CONFIG`.

### Price (`family:price`)

| Spec | Production class | Notes |
|------|------------------|-------|
| `price:residual` | `TorchResidualPriceModel` | MOF VWAP residual EWMA |
| `price:conditional_mechanical_mixture` | `TorchConditionalMechanicalMixturePriceModel` | Market vs refresh mixture |

### Baselines (`family:baseline`)

| Spec | Purpose |
|------|---------|
| `baseline:zero_flow` | Always predict zero volume |
| `baseline:mid_random_walk` | EWMA Gaussian on `delta_mid` |

### Inline JSON config

```bash
python -m research.model_lab.experiments.compare \
  --models 'flow:hurdle_lognormal{"decay":0.99,"init_p0":0.2}'
```

---

## 10. How to add a new model

1. **Implement `OnlineModel`** (`core/protocol.py`):
   - `reset()`, `get_params()`, `score(previous, current)`, `update(previous, current)`
   - `score` must use only state from prior updates.

2. **Place implementation** in `models/your_model.py`.

3. **Register** in `models/registry.py` → `build_model()`.

4. **Add a test** in `tests/test_walk_forward.py` with synthetic `IntervalStep` data.

5. **Run compare** and check `outputs/*_summary.json`.

### Minimal template

```python
from dataclasses import dataclass
from research.model_lab.core.types import IntervalStep, StepScore

@dataclass
class MyFlowModel:
    name: str = "my_flow"
    kind: str = "my_kind"
    family: str = "flow"

    def reset(self) -> None: ...
    def get_params(self) -> dict: ...
    def score(self, previous: IntervalStep | None, current: IntervalStep) -> StepScore: ...
    def update(self, previous: IntervalStep, current: IntervalStep) -> bool: ...
```

---

## 11. Experiment workflow for agents

### Step A — Smoke test (no DB)

```bash
pytest research/model_lab/tests/test_walk_forward.py -q
```

### Step B — Single-symbol benchmark

```bash
python -m research.model_lab.experiments.compare \
  --exchange nobitex --symbol USDT_IRT --interval-ms 30000 \
  --since 48h --write-scores
```

### Step C — Read leaderboard

Open `research/model_lab/outputs/*_summary.json`. Sort by `mean_nll` ascending.

### Step D — Parameter sweep (manual)

Repeat with inline JSON configs, e.g. decay grids:

```bash
for d in 0.90 0.95 0.99; do
  python -m research.model_lab.experiments.compare \
    --since 48h --interval-ms 30000 \
    --models "flow:hurdle_lognormal{\"decay\":$d}" \
    --output-dir research/model_lab/outputs/decay_sweep
done
```

### Step E — Validate on holdout window

Run twice with different `--since` / `--until` windows; the model that wins on both is more robust.

---

## 12. Interpreting results

| Metric | Meaning |
|--------|---------|
| `mean_nll` | Average `-log P(observation \| past)`. **Primary ranking metric.** |
| `median_nll` | Robust to burst intervals with tiny probability |
| `n_skipped` | Steps where scoring failed (missing book, invalid features) |
| `skip_reasons` | Count by reason — investigate if large |

**Caveats:**

- NLL scales differ between families (flow vs price). **Compare within family first.**
- Sparse markets (many zero-volume buckets) favour hurdle models; check `baseline:zero_flow`.
- Shallow books from `market_facts` distort price-model scores — use `mm_events` for price comparisons.
- Production runtime may use different config defaults; align `registry.py` defaults with your bot preset.

---

## 13. Mapping to production runtime

| Lab | Production |
|-----|------------|
| `IntervalStep` | `MarketEvent` / frame local event |
| `ProductionFlowModel` | `controller.local_market_flow_model` |
| `ProductionPriceModel` | `controller.local_price_model` |
| `run_walk_forward` | `score_frame_models` + `model.update` in `MpcStrategyEngine` |
| `mm_model_score_facts` table | Same NLL contract persisted at runtime |

Reference docs in repo:

- `docs/old/price-and-orderflow-models.md`
- `mm/models/scoring.py`
- `mm/mpc/price_torch.py`, `mm/mpc/market_order_flow_torch.py`

---

## 14. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Need at least 2 interval steps` | Persistence not running or wrong symbol/interval |
| `No database URL` | Set `MM_DATABASE_URL` in `.env` |
| `ModuleNotFoundError: psycopg2` | `pip install -r requirements.txt` |
| Price model high `n_skipped` | Use `--source events` for full orderbooks |
| All flow models tie | Try longer `--since`; market may be mostly zero-volume |

---

## 15. Suggested research agenda

1. **Flow family bake-off** — hurdle_lognormal vs hurdle_gamma vs weibull on 30s and 10s intervals.
2. **Price family bake-off** — residual vs conditional_mechanical_mixture on full `mm_events` books.
3. **Decay sensitivity** — sweep `decay` / `residual_decay` and plot mean NLL vs decay.
4. **Regime slices** — filter steps by volume quantile; score models per regime.
5. **New models** — implement AR features, microprice delta, or copula joint flow; register and compare.

---

## 16. CLI reference

```
python -m research.model_lab.experiments.compare \
  --exchange nobitex \
  --symbol USDT_IRT \
  --interval-ms 30000 \
  --since 48h \
  --until 2026-06-22T00:00:00Z \
  --source auto \
  --models flow:hurdle_lognormal,price:residual \
  --min-updates-before-score 1 \
  --database-url postgresql://... \
  --output-dir research/model_lab/outputs \
  --write-scores \
  --list-models
```

---

## 17. Contact points in code

- **Change evaluation logic:** `core/walk_forward.py`
- **Change NLL for a model type:** `core/scoring.py`
- **Change data SQL:** `data/loader.py`
- **Add models:** `models/registry.py` + new file under `models/`
- **Change CLI:** `experiments/compare.py`

Good luck — optimize for **lower walk-forward NLL** on holdout windows, not in-sample fit.
