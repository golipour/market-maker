# Order-book analysis: production model findings (2026-09-28)

## What was done

Extracted the raw L2 book from Postgres and ran the **real** production
price model on it.

- Source: `mm_events`, topic `market:events:nobitex:USDT_IRT:10000`,
  `event_type='MarketEvent'` — 68,217 buckets, 7.90 days, **zero gaps**
  (`dt == 10000` for every consecutive pair), 24 levels per side.
- File: `research/model_lab/dataset/nobitex_usdt_irt_10s_book.csv` (125 MB).
- Model: `TorchConditionalMechanicalMixturePriceModel`, imported directly
  from `mm/mpc/price_torch.py`, with the real production parameters from
  `mm/configs/market_maker.runtime.example.json:225`.
- Books normalized through the production
  `normalize_orderbook_from_raw`.

This finally removes the caveat carried in Q26/Q27, where the gate and RLS
mean had to be stubbed because the CSV only holds book aggregates.

## Tick size

`tick_size = 1.0` IRT, `lot_size = 0.01`, derived from the book itself
(every level price is a whole number; min observed qty is 0.01).
Bid/ask are integers, mid is a half-integer, so `dmid` lives on a 0.5 grid
and the lattice cell width is `h = 0.5`. This **confirms** the assumption used
throughout Q1–Q29.

## Finding 1 — the gate saturates and disables the Student-t branch

`market_resp` (the EM responsibility weight) pins at 1.0 for **97.4%** of
steps:

| window | mean `w` | max |
|---|---|---|
| first 5k | 0.826 | 1.000 |
| mid 5k | 0.993 | 1.000 |
| last 5k | 0.993 | 1.000 |

`gate_beta` fits to `[4.788, 0.044]`, i.e. `q_market ≈ 0.993` for all
realistic flow. The model therefore predicts

```
q(y) = 1.0 * Normal(0, market_var) + ~0.0 * StudentT(3, refresh_var)
```

The Student-t refresh branch — the entire fat-tail defence — is switched off
in practice. The model is, on this data, a **Gaussian**.

The gate is a *responsibility* estimate under a Gaussian market branch. A
Gaussian can always explain the data slightly better than a heavy-tailed
alternative at the point estimate, so the EM weight drifts to the branch that
fits the bulk and stops paying for the tail.

## Finding 2 — the Gaussian's scale is far too small for the moves

Fitted `market_var['mid']` gives `sigma = 88.7` ticks. The realised `dmid`
has `sigma = 61.2` ticks but a **maximum of 951 ticks = 100.9 sigma**.

Consequence: density underflows on extreme steps. Scoring with a
`1e-300` floor produced losses of exactly `-log(1e-300) = 690.78` for 10% of
steps, and a mean NLL of 73.6 nats.

**The mean NLL is not a usable statistic on this data.** The loss is entirely
tail-driven:

| loss bucket | share of total loss |
|---|---|
| below median (3.12) | ~2.0% |
| p90 (188.6) | 10.1% |
| top decile | **84.8%** |

Median NLL is a sane 3.12 nats. Any comparison on the mean is a comparison
of how many steps underflowed.

## Finding 3 — the zero mass is real in live data

`P(dmid = 0) = 0.3124` on the live book, versus 0.3507 in the CSV window.
The hurdle is a genuine feature of this market, not an artifact of one
sample.

## Status of the earlier recommendation

The proposal (add a zero atom `p0 * 1{y == 0}` to the predictive density)
is **unchanged and still supported** — Finding 3 confirms the atom exists in
live data at a similar rate.

However, the Q28 "add the zero atom" experiment was scored on the **CSV**,
whose `dmid` has `sigma = 52.2` and a much fatter empirical tail
(`max = 3563`). On the live book the incumbent collapses to a Gaussian with
a 100-sigma miss rate, so the ranking of fixes could differ. **That
comparison has not been redone on the book data** and should be before any
code change is proposed.

## What is NOT established

- No claim is made here that the model is "broken" in production; the
  saturation is a property measured on 7.9 days of one instrument.
- The `w` weight shown is `market_resp`, which the model uses as a *mixing*
  weight in `sample_dynamic_vwap`; whether the deployed sampler weights
  identically was not traced end-to-end.
- Downstream MPC effect (re-quote rate, inventory distribution, PnL) was not
  measured.
