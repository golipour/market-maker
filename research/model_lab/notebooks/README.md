# Notebooks

| Notebook | Purpose |
|----------|---------|
| [production_models_walkforward.ipynb](production_models_walkforward.ipynb) | **Start here** — hurdle lognormal flow + conditional mechanical mixture price, walk-forward NLL |
| [experiments_01_online_model_search.ipynb](experiments_01_online_model_search.ipynb) | Hyperparameter sweeps, regime diagnostics, and research-only candidate models |
| [experiments_02_comprehensive_plan.ipynb](experiments_02_comprehensive_plan.ipynb) | **Plan-driven** — compare model families across Tracks A/B/C (tails, joint flow, GARCH, HMM, stationarity) with plain-English verdict |

## Run

From repo root (Python 3.10+, `pip install -r requirements.txt`, Postgres with market data):

```bash
pip install pandas matplotlib  # optional for tables/plots
jupyter notebook research/model_lab/notebooks/production_models_walkforward.ipynb
```

Or open in VS Code / Cursor and run all cells.

## Customize

Edit `FLOW_HYPERPARAMS` / `PRICE_HYPERPARAMS` in §2, or use the experiment slot in §9.
