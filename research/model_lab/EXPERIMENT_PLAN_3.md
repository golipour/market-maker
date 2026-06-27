---
name: Model Lab Experiments
overview: Design a new research notebook and experiment agenda for online walk-forward flow and price model comparison on Postgres market data, starting with Nobitex USDT_IRT at 30000ms. The plan preserves production code and the existing baseline notebook, and focuses on objective prequential NLL comparisons with regime-aware diagnostics.
todos:
  - id: baseline-context
    content: Load Nobitex USDT_IRT 30000ms data and reproduce production baseline flow and price NLL in a new notebook.
    status: completed
  - id: flow-sweeps
    content: Run existing flow-family hyperparameter sweeps and rank by prequential NLL plus regime-stratified NLL.
    status: completed
  - id: price-sweeps
    content: Run existing price-family hyperparameter sweeps and compare conditional mixture, residual, and mid-random-walk baselines.
    status: completed
  - id: analysis-layer
    content: Add rolling NLL, cumulative NLL difference, volume-regime stratification, and paired model-difference summaries.
    status: completed
  - id: new-model-shortlist
    content: "Prototype the highest-value lab-only candidates: log-Student-t or bivariate hurdle flow, plus Student-t or flow-conditioned mid price."
    status: completed
  - id: stationarity-check
    content: Evaluate stationary representations before considering small GRU or Transformer density models.
    status: completed
isProject: false
---

# Model Lab Experiment Plan

## Problem Restatement

We are modeling interval market dynamics for one venue/symbol/interval sequence, initially `nobitex` / `USDT_IRT` / `30000ms`, using persisted Postgres data from `mm_events` and `mm_market_facts`.

At each interval step `t`, the model has only history through `t-1`. It must assign probability to the next observation, record one-step-forward NLL, then update online using the new transition. This is prequential evaluation, not train/test splitting.

The two model families must be compared separately:

- Flow models predict the joint next-step distribution of aggressive `(buy_volume, sell_volume)`.
- Price models predict next-step price/book transition behavior, usually after separating the mechanical effect of aggressive flow from residual or refresh behavior.

The goal is not PnL optimization and not production refactoring. The goal is to find models and hyperparameters with lower walk-forward NLL, better regime behavior, and valid generative sampling semantics for possible later MPC use.

## Existing Research Surface

Use the existing lab rather than modifying production runtime:

- `research/model_lab/GUIDE.md` defines the problem, production model formulas, and notebook conventions.
- `research/model_lab/EXPERIMENT_PLAN.md` and `research/model_lab/EXPERIMENT_PLAN_2.md` already outline candidate model families.
- `research/model_lab/core/walk_forward.py` provides the key contract: score first, update second.
- `research/model_lab/models/registry.py` supports model specs such as `flow:hurdle_lognormal{"decay":0.99}` and `price:conditional_mechanical_mixture{"decay":0.995}`.
- `research/model_lab/notebooks/production_models_walkforward.ipynb` is the baseline notebook and should not be overwritten.

## New Notebook Scope

Create a new notebook, likely `research/model_lab/notebooks/experiments_01_online_model_search.ipynb`, with sections for:

- Dataset selection and data health checks for `nobitex`, `USDT_IRT`, `30000ms`.
- Baseline reproduction for production flow and price models.
- Hyperparameter sweep leaderboards.
- Regime-aware analysis: zero-volume, low-volume, high-volume, spread, hour-of-day, and cold-start vs warm performance.
- Paired NLL comparisons between candidate models on identical timestamps.
- Final shortlist of winners and failure modes.

## Objective Metrics

Primary metric:

- Mean prequential NLL over scored steps, compared only within the same family.

Secondary metrics:

- Median NLL and tail NLL percentiles.
- Cumulative NLL difference vs production baseline.
- Rolling mean NLL over time to see whether a model improves as it learns.
- NLL by regime: zero-volume, active volume, high-volume tail, wide-spread periods, and hour-of-day.
- Paired bootstrap confidence intervals or paired t-style summaries for NLL differences.

## Experiment Track A: Existing Flow Model Sweeps

Start with the cheapest, highest-signal experiments:

- Compare `hurdle_lognormal`, `hurdle_gamma`, `hurdle_weibull`, `gamma`, and `weibull`.
- Sweep decay values such as `0.80`, `0.90`, `0.95`, `0.975`, `0.99`, `0.995`.
- Sweep sparse-market priors such as `init_p0` around `0.3`, `0.5`, `0.6`, `0.7`.
- Report whether different decays win in cold-start vs warm windows.

Hypothesis: because the observed flow is sparse and bursty, zero-inflated/hurdle families should beat non-hurdle families, and optimal decay may be lower during regime shifts but higher during stable periods.

## Experiment Track B: New Flow Models

If the baseline sweeps are not enough, add lab-only models behind the existing `OnlineModel` protocol:

- Log-Student-t hurdle flow for heavier tails than lognormal.
- Bivariate hurdle model with explicit probabilities for `(0,0)`, `(>0,0)`, `(0,>0)`, and `(>0,>0)`.
- Conditional side model: first model buy volume, then model sell volume conditional on buy activity or recent imbalance.
- State-conditioned hurdle model where the zero probability depends on lagged spread, recent volume, and flow imbalance.
- Adaptive-forgetting flow model where decay changes after large prediction errors.

The first new candidate should likely be log-Student-t hurdle or bivariate hurdle because they preserve a tractable PMF and should be easy to compare objectively.

## Experiment Track C: Existing Price Model Sweeps

Compare production price models under controlled hyperparameter grids:

- `price:conditional_mechanical_mixture` with sweeps over `decay`, `reference_vwap_vol`, `shape_alpha`, `student_df`, and `initial_sigma`.
- `price:residual` with sweeps over `residual_decay`, `residual_sigma`, and `reference_vwap_vol` where the adapter supports it.
- `baseline:mid_random_walk` as a simple sanity check.

For price models, report NLL overall and by volume regime, because quiet periods and burst periods may prefer different models.

## Experiment Track D: New Price Models

Add models only after baseline price sweeps establish weaknesses:

- Student-t mid delta model on stationary targets such as `delta_mid`, log-return, or spread-normalized delta.
- Flow-conditioned mid model with online regression on lagged volume, spread, imbalance, and recent return.
- AR(1) or AR(p) model on residuals to capture mean reversion or momentum in book refresh.
- GARCH-style volatility model for volatility clustering after large flow events.
- Two-state online regime model for quiet vs active regimes, compared against the current instantaneous flow-gated mixture.
- Microprice delta model if full orderbook features are reliable enough.

The price model scoring target must stay explicit. Full book/VWAP models should be compared to full book/VWAP models; mid-only models should be labeled as narrower-price-target baselines unless their likelihood target matches the production model target.

## Experiment Track E: Stationarity And Neural Feasibility

Before trying neural sequence models, test whether transformations make the series stable enough:

- Log mid return instead of price level.
- Spread-normalized mid delta.
- Volume divided by an EWMA activity scale.
- Flow imbalance and signed flow ratio.
- Lag windows of 10, 30, and 60 intervals.

Only after these diagnostics should we try small neural models. With roughly a few thousand steps in the reference window, start with small GRU or tiny Transformer density heads only if longer history is available or stationary features show strong predictive structure.

## Comparison Discipline

Every candidate must be evaluated on the same ordered step sequence and same scoring timestamps. The notebook should make it easy to answer:

- Which model wins total mean NLL?
- Which model wins after warm-up?
- Which model wins in zero-volume vs high-volume regimes?
- Is the win statistically meaningful on paired per-step NLL differences?
- Does the model remain sampleable and interpretable enough for possible MPC use?

## Deliverables

- A new notebook under `research/model_lab/notebooks/`.
- No changes to production `mm/` code during the first experiment wave.
- Optional lab-only helper code only if needed for repeated analysis or new model classes.
- A compact result summary in `research/model_lab/outputs/` after experiments are run.

## Recommended First Coding Pass

First implement the notebook and run only existing model sweeps plus regime analysis. Then add one or two new lab-only flow models and one or two simple price models. Leave neural models for a later pass unless the stationarity diagnostics and available data length justify them.