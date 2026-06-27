# Model Lab

Walk-forward flow/price model research on persisted Postgres market data.

## Start here

| Doc | Contents |
|-----|----------|
| **[GUIDE.md](GUIDE.md) §1** | Problem domain, interval steps, walk-forward NLL metric |
| **[GUIDE.md](GUIDE.md) §2** | **Current production models** — formulas, configs, trained-parameter examples |
| **[GUIDE.md](GUIDE.md) §5** | Quick start commands |
| **[notebooks/production_models_walkforward.ipynb](notebooks/production_models_walkforward.ipynb)** | Interactive walk-forward NLL for production flow + price models |

## One-command benchmark (production models)

```bash
python -m research.model_lab.experiments.compare \
  --exchange nobitex --symbol USDT_IRT --interval-ms 30000 \
  --since 48h \
  --models flow:hurdle_lognormal,price:conditional_mechanical_mixture,price:residual,baseline:zero_flow
```

See `experiments/configs/default.json` for the full model list matching `mm/configs/market_maker.runtime.example.json`.
