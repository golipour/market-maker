# Algorithm & architecture walkthrough

This folder contains a **code-derived** end-to-end walkthrough of the
`mm/` Python package. It is meant to be read in order; each section builds
on the previous one. Citations use the `startLine:endLine:filepath` form so
they can be opened in the editor.

| # | Section | What it covers |
| --- | --- | --- |
| 00 | [Overview](00-overview.md) | Bird's-eye view, conventions, file map. |
| 01 | [Runtime & processes](01-runtime.md) | `RuntimeComponents`, the event bus, the control plane. |
| 02 | [Ingestion](02-ingestion.md) | From raw trades to per-interval `MarketEvent`s. |
| 03 | [Frame assembly](03-frame-assembly.md) | `StrategyInputFrame` and `JointStrategyFrame`. |
| 04 | [Strategy runner](04-strategy-runner.md) | The per-decision loop, cancellation, gates. |
| 05 | [MPC state & action](05-mpc-state.md) | `State`, `SigmoidMixtureAction`, L-param decoding. |
| 06 | [Stochastic models](06-mpc-models.md) | Price and order-flow models. |
| 07 | [The optimizer](07-mpc-optimizer.md) | Monte Carlo, Adam, warm-start, cancellation. |
| 08 | [Reward & objective](08-objective.md) | PnL decomposition, validity, viability. |
| 09 | [Target book](09-target-book.md) | Sigmoid → ladder → `JointTargetBook`. |
| 10 | [Risk](10-risk.md) | `NetDeltaRiskEngine`, validity/viability backstops. |
| 11 | [Edge cases](11-edge-cases.md) | Negative objective, finite checks, restart. |
| 12 | [End-to-end sample](12-end-to-end.md) | One `MarketEvent` traced from the wire to the venue. |
| 13 | [Glossary](13-glossary.md) | Definitions for the terminology used in the document. |

## How to read

1. Start with [00-overview.md](00-overview.md) for the bird's-eye view.
2. Read sections 01–04 to understand the *non-MPC* part of the system (the
   "data plane").
3. Read sections 05–08 to understand the MPC *itself* (the "decision plane").
4. Read sections 09–10 to understand the *non-MPC post-processing* (the
   "execution plane").
5. Read sections 11–12 for the failure modes and a worked example.
6. Use [13-glossary.md](13-glossary.md) as a reference.

## What this document is not

- It is not an API reference. The auto-generated API reference lives
  elsewhere; this document focuses on the *flow* and the *math*.
- It is not a config guide. The config schema and example values are
  documented separately in [`docs/configuration.md`](../configuration.md).
- It does not discuss historical experiments; those are in
  [`research/`](../research) and are explicitly *not* used as a source of
  truth for the algorithm.
