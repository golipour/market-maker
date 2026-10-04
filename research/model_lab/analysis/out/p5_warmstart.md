# Warm start x learning rate: the interaction that changes the lr recommendation

## Why this was re-run

Every earlier sweep in this investigation used the default
`WarmStartParams(enabled=False)`, i.e. a cold start on each decide. The
shipped preset (`mm/configs/market_maker.runtime.example.json`) is warm:

    warm_start = {enabled: true, alpha: 0.8, alpha_mode: "fixed",
                  preserve_optimizer_state: true}

Cold start is not a neutral baseline. It changes three things:

1. the epoch-0 point is a fixed symmetric default rather than a blend of the
   previous decision,
2. Adam's moments are reset each decide, so decisions are independent,
3. the corpus is replayed from a fixed point, not advanced through time.

So the cold-start optimum is not the production optimum.

## Method

`p5_warmstart_lr.py` replays a corpus as a **time-ordered decision
sequence** with a single optimizer instance, which is what production does.
The per-epoch trace is captured by wrapping `torch.optim.Adam.step` on the
class (instance wrapping does not work: the optimizer is rebuilt on entry to
`optimize_with_diagnostics`).

Cadence matters: production decides every 10 s. Striding the replay by 37 rows
(370 s) makes the previous book almost unrelated to the current one and
cripples warm start. All decisive runs use `stride=1`.

## Result: the lr optimum depends on warm start

100 consecutive real Nobitex states from row 40000 and, independently,
100 states from row 150000. 20-epoch production budget.

    arm         corpus A (row 40k)          corpus B (row 150k)
    cold lr=1.0   1.546129e-03  (100.00%)   1.355876e-03  (100.00%)
    warm lr=0.15  1.441746e-03  ( 93.25%)   1.275143e-03  ( 94.05%)
    warm lr=1.0   1.389181e-03  ( 89.86%)   1.199076e-03  ( 88.44%)
    cold lr=0.15  1.210189e-03  ( 78.25%)   1.067210e-03  ( 78.71%)

Paired per-decide comparison, same lr, same sequence:

    lr=1.00   warm - cold = -1.57e-04   warm wins  50/100  (corpus A)
    lr=1.00   warm - cold = -1.57e-04   warm wins  36/100  (corpus B)
    lr=0.15   warm - cold = +2.32e-04   warm wins  96/100  (corpus A)
    lr=0.15   warm - cold = +2.08e-04   warm wins  94/100  (corpus B)

## Reading

* With **cold** start, `lr=0.5` is the best arm and `lr=1.0` oscillates
  (18-26% of epochs in the heavily penalized negative region).
* With **warm** start, `lr=1.0` is clearly harmful (warm loses to cold on
  36-50% of decides) because the larger step throws the parameter away from
  the warm point that warm start just paid for, and the preserved Adam
  moments carry that mistake into the next decide.
* `lr=0.15` inverts the sign: warm start then beats cold on 92-96% of
  decides. The interaction replicates across two disjoint corpora and under
  two different book constructions (geometric-decay ladder and flat ladder),
  so it is not an artifact of the synthetic replay book.

## Caveat: the replay book is synthetic

The cached export carries top-of-book and aggregate depth, not a per-level
ladder, so `replay_states.py` synthesizes one. The absolute J values are
therefore not production-representative, and the identity of the single best
arm flips between corpora (cold+lr1.0 wins on A and B in mean J, but
warm+lr0.15 is the only arm where warm start reliably beats cold start).

What survives the caveat is the **interaction**: `lr=1.0` is unsafe with warm
start regardless of corpus, and a large reduction in lr converts warm start
from a liability into a win. The actionable, corpus-independent claim is
directional, not a specific number.

## Not recommended from this evidence

Do not set a specific `adam_lr` from these runs. The correct next step is a
shadow run against live float64 decisions with real ladders, comparing
`lr=1.0` (current) against a reduced lr under production warm start.
