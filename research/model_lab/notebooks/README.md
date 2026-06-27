# Notebooks

| Notebook | Purpose |
|----------|---------|
| [production_models_walkforward.ipynb](production_models_walkforward.ipynb) | **Start here** — hurdle lognormal flow + conditional mechanical mixture price, walk-forward NLL |

## Run

From repo root (Python 3.10+, `pip install -r requirements.txt`, Postgres with market data):

```bash
pip install pandas matplotlib  # optional for tables/plots
jupyter notebook research/model_lab/notebooks/production_models_walkforward.ipynb
```

Or open in VS Code / Cursor and run all cells.

## Customize

Edit `FLOW_HYPERPARAMS` / `PRICE_HYPERPARAMS` in §2, or use the experiment slot in §9.
