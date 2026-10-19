I now have a complete picture. Here is the analysis.

---

# Failure-Mode Analysis: Viability & Validity Constraints vs. Pure Risk-Adjusted Return

## Current live configuration (from `mm_events`)

| Parameter | Value |
| :--- | :--- |
| `min_viable_quote` | **250,000,000 IRT (Toman)** |
| `shortfall_tolerance` (ε) | **0.10** |
| `lambda_learning_rate` (η) | **0.05** |
| `lambda_init` / `lambda_max` | **1.0** / **100.0** |
| `dual_ewma_decay` | **0.9** |
| `recovery_fraction` (ρ) | **0.0** (disabled) |
| `validity.penalty_scale` (μ) | **0.0** (disabled) |
| `validity.min_score` | 0.30 |
| `objective_normalize` | **unset / disabled** |
| `horizon` / `n_samples` / `adam_epochs` | **3 / 16 / 8** |

The composed objective is:

```1002:1086:mm/mpc/joint_optimizer.py
            J, components = combine_validity_term(   # −μ·max(0, B_min − B)
            ...
            J, components = combine_viability_term(  # −λ·S
```

where \(S = \mathbb{E}[d_{\text{bid}}^2 + d_{\text{ask}}^2]\) with \(d = [q_{\min} - \text{reserve}]_+ / q_{\min}\).

---

## Category 1 — Viability: Structural Failure Modes

### 1.1 The reward is *relative*, but the loss is *absolute* — scale mismatch

The base term `risk_adjusted_return` is a **dimensionless** return (ΔW/W₀), typically \(10^{-4}\)–\(10^{-6}\). The viability term is \(-λ \cdot S\), where \(S\) is O(1).

When `objective_normalize` is **disabled**, no EWMA scaling reconciles them. The consequence is that viability **always wins when λ > 0**, regardless of how small the reserve breach actually is. There is no regime where a genuine \(0.001\%\) shortfall gets traded off against a real PnL opportunity.

**Loss scenario:** During the night, `viability_penalty` reached **98.9% of total loss** at a bid deficit of only **0.40** (i.e. cash 40% below reserve). A pure risk-adjusted-return optimizer would have continued two-sided quoting and earned the spread. The viability term instead suppressed bids to **0.04 USDT** — effectively withdrawing the market-making business for4 hours. That is a guaranteed opportunity loss, and during a trending move it converts to *adverse selection*: quoting only asks into a rising market means you never get the up-tick inventory you need.

### 1.2 λ is a **global, persistent, non-convex** dual state — hysteresis and overshoot

λ lives on the MPC instance (`mpc._viability_lambda_local`) and is updated **once per frame**, outside the inner solve:

```1283:1292:mm/mpc/joint_optimizer.py
        lambda_applied_local = mpc._viability_lambda_local
        mpc._viability_lambda_local, mpc._viability_shortfall_ewma_local = (
            update_viability_lambda(
```

with ascent \(\lambda \leftarrow \text{clip}(\lambda + \eta(\bar{S} - \varepsilon), 0, \lambda_{\max})\).

Three distinct hazards:

1. **Asymmetric rise, slow decay.** λ rises at `η=0.05 × gap` per tick, but only falls when \(\bar{S} < \varepsilon\). After the night event λ peaked at **2.47** and took ~4 hours of clean recovery to return to 0. During that entire window, residual λ kept applying a penalty even as the breach was fully resolved (06:00 → 09:00 shows λ falling2.47 → 0.00 while `bid_deficit` was still 0.30 → 0.06). **You are trading with a stale penalty from a past regime.**

2. **The update uses the *best* epoch's components, not the realized next-state.** `update_viability_lambda` reads `best_metrics.components`, which is the objective evaluated *on simulated paths*. If the MC sample is optimistic about fills (only 16 samples, horizon 3), \(\bar{S}\) is biased low, λ under-rises, and the breach persists longer than the math says it should.

3. **λ is not reset on regime change.** `reset_viability_lambda` exists but is **never called from the optimizer** — it is only referenced in tests. So λ persists across a crash-restart boundary, a symbol switch, or a config reload that changes `min_viable_quote`.

**Loss scenario:** A transient 30-second dip in cash (e.g. one large manual withdrawal, or a single large fill) pushes λ from 0 → 2.5. The bot then suppresses bids for hours afterward — long after the cash has been restored by unrelated sells. You lose the entire up-tick.

### 1.3 The constraint uses **total** balance, not **free** balance — it under-counts committed capital

```200:222:mm/core/wallet.py
def reported_wallet_total(raw: Any) -> float:
```
`inventory_cash_from_snapshot` returns `reported_wallet_total(quote)` = **free + locked**.

This is the safe direction (total ≥ free, so the constraint is *looser* than a free-balance constraint would be). But it creates the opposite failure:

**Loss scenario — the reverse trap.** Because the constraint only watches the *total*, it is completely blind to *encumbrance*. If the bot has250M Toman of total IRT but 240M of it is locked in resting bids, `d_bid = 0` and the constraint is silent. The optimizer will happily quote more bids, all of which will be rejected by the exchange for insufficient free balance. You pay the `rules_min_notional` / insufficient-funds rejection cost on every order while believing you are safe. Last night this is visible as `avg_rejected` climbing to **6.8/tick** with `pending_place_count` at **~10** — the optimizer thought it was placing, and the venue was refusing.

### 1.4 Viability is a **level** constraint with **squared** hinge — it is blind to the *rate* of change

\(d^2\) penalizes *distance below reserve*, not *velocity toward it*. There is no term that rewards *accelerating* recovery. With `recovery_fraction = 0.0`, the explicit recovery objective is entirely off:

```1039:1044:mm/mpc/objective_terms.py
    if viability_params is not None and float(viability_params.recovery_fraction) > 0.0 and rec_shortfall is not None:
```

**Loss scenario:** During the 04:00–07:00 window the bot had **zero bid capacity** and sat on 222M Toman of asks. If the market had continued *down*, the bot would have been unable to buy the dip at all — it would have watched its reserve drain to zero with no mechanism to defend. The squared hinge gives no extra credit for acting early, so the optimizer has no reason to pre-emptively rest a *small* bid that would keep the reserve from breaching in the first place. The constraint is purely reactive.

### 1.5 The reserve is denominated in **quote**, but the breach is caused by **base** accumulation

\(d_{\text{ask}} = [q_{\min} - b_t p_t]_+ / q_{\min}\). Last night `d_ask` stayed at **0.0000** while the bot held3,335 USDT ≈ 878M Toman — because \(q_{\min} = 250\)M is far below the base-notional. So in a *downtrend* the bot accumulates base, which *inflates* base-notional and makes `d_ask` even more slack, while `d_bid` is the only binding side.

**Loss scenario:** In a **downtrend**, base-notional *falls* (price drops), so \(b_t p_t\) shrinks. If the price drops ~72%, `d_ask` would suddenly bind too — and at that point both sides are in deficit simultaneously, \(S\) roughly doubles, and the bot has *no* quoting capacity on either side. It becomes completely passive and cannot trade its way back. The reserve is a fixed *quote* number with no volatility adjustment, so it is procyclically tightest exactly when volatility is highest.

### 1.6 MC noise with n_samples=16 makes λ jitter, and the penalty is non-smooth

`viability_path_shortfall` reduces over a (16, 3) tensor. The hinge `clamp(-m, min=0)` has **kink at the boundary** — gradient is discontinuous. With 8 Adam epochs and 16 paths, the estimate of \(S\) is high-variance. This produces:
- **Gradient spikes** at the reserve boundary → the optimizer oscillates between quoting and not quoting.
- **λ feedback on noise**: a single unlucky frame raises \(\bar{S}\), raising λ, which suppresses bids, which *changes the fill distribution* next frame, which is the signal λ is supposed to be tracking. This is a **control loop on a noisy estimator** — a classic setup for limit-cycle oscillation.

---

## Category 2 — Validity (Orderbook Shape): Failure Modes

Currently `penalty_scale = 0.0`, so validity is **inert**. But it is the term most likely to cause large loss once enabled, because of *how* it is shaped.

### 2.1 Validity is a **hinge on a composite score** → it is satisfiable by *gaming* the score

```945:965:mm/mpc/objective_terms.py
    shortfall = torch.clamp(float(params.min_score) - book_score, min=0.0)
```

`B ∈ [0,1]` is a weighted mix of presence/tightness/balance/uniformity. The optimizer maximizes \(B\) to get above 0.30. Since \(B\) is a *soft* score, the cheapest way to raise it is **not** to quote well — it is to quote **wide and thin**. Raising `presence_reference_qty` or widening spreads can lift `tightness` while destroying actual market-making quality.

**Loss scenario:** With `penalty_scale=100` and `min_score=0.3`, a shortfall of 0.01 costs 1.0 in objective — versus a `risk_adjusted_return` of order \(10^{-4}\). The penalty is **~10,000× larger than the entire return signal**. The optimizer will do *anything* to satisfy validity, including quoting a single dust order far from mid to satisfy `presence`. Meanwhile it has abandoned all economically meaningful quoting. This is exactly the class of bug that produces "the bot is technically alive but makes no money."

### 2.2 `μ` is **fixed**, unlike viability's adaptive λ — no dual feedback

Viability learned its multiplier from observed shortfall. Validity uses a **hard-coded** `penalty_scale`. If μ is set too high, there is no mechanism to relax it. If set too low, it does nothing. There is no `min_score` adaptation and no reset.

### 2.3 Validity is evaluated **only at h=0** — it does not constrain future steps

```1002:1029:mm/mpc/joint_optimizer.py joint_h0 = x_torch[:joint_step_len]
            ... book_quality_score(local_h0, ...)
```

Only the **first** horizon step's decoded book is scored. With `horizon=3`, steps h=1,2 are unconstrained. The optimizer can satisfy validity at h=0 (a pretty, tight book) and then quote garbage in h=1,h=2. Since the controller typically executes the h=0 book, this is *mostly* benign — but it means validity provides **no protection against the optimizer using bad future steps to game the terminal reward**, and it provides zero guarantee about the h≥1 books at all.

### 2.4 Validity and viability can **conflict**, and there is no precedence rule

Both subtract from \(J\). Nothing in the code establishes an ordering. If validity pushes toward *thin, tight, high-score* books while viability pushes toward *zero-size* books when cash is short, the optimizer gets **conflicting gradients**. In the extreme (validity demanding a minimum depth, viability demanding near-zero bid size), the result is a book that is valid-by-score but economically useless — or the reverse.

### 2.5 `measure='combined'` couples validity to **market** depth, which the bot does not control

If `book_quality.measure = 'combined'`, `B` includes *external* market depth. When the market thins out (exactly when you most need your own quotes), `B` drops for reasons entirely outside the bot's control — and the penalty fires, shrinking quotes at the worst possible moment.

---

## Category 3 — Interaction & Systemic Risks

### 3.1 No **hard** reserve enforcement anywhereBoth constraints are **soft penalties** added to a scalar objective. There is no post-solve check, no order-size clamp, no assertion that the final book satisfies `cash_free ≥ threshold`. A sufficiently strong `risk_adjusted_return` signal — or a pathological MC sample — can produce an action that violates the reserve outright, and it will be dispatched to the exchange.

### 3.2 Optimizer **timeout** silently freezes λ in a bad state

```643:658:mm/application/strategy_runner.py
        except asyncio.TimeoutError as exc:
            timed_out = True
            self._decide_cancel_event.set()
```

When the optimizer times out (it did, at05:00 and again at 22:00 — `OptimizerDecisionTimeout: exceeded 8000ms`), the runner **holds current orders**. But `update_viability_lambda` is inside the optimizer's post-loop block — **if the loop is cancelled, λ is never updated**. λ is therefore frozen at whatever value the last *completed* frame set. Repeated timeouts mean λ can sit at a stale high value indefinitely, with no decay path.

### 3.3 The loss I saw last night was **not** primarily a viability *error* — it was viability **working as designed but with the wrong objective balance**

To be precise: the system did not lose money. It *stopped making money* for ~4 hours while the reserve was below threshold. The fills data shows it bought 670 USDT and sold 437 USDT at roughly flat prices (263,452 avg buy vs 263,451 avg sell). **P&L was ~zero.** The cost was **opportunity**: it was one-sided for a third of the night.

---

## Summary Table

| # | Failure Mode | Trigger | Loss Type | Severity |
| :--- | :--- | :--- | :--- | :--- |
| 1.1 | Scale mismatch (no CE normalization) | Any reserve breach | Opportunity loss (withdraws business) | **High** |
| 1.2a | λ asymmetric rise / slow decay | Transient breach | Hours of suppressed quoting post-recovery | **High** |
| 1.2b | λ updated from optimistic MC (n=16) | Noisy fills | Under-reacts, breach persists | Medium |
| 1.2c | `reset_viability_lambda` never called | Restart / config change | Stale λ across regimes | Medium |
| 1.3 | Total-vs-free balance blindness | High encumbrance | Orders rejected, false safety | Medium |
| 1.4 | No recovery term (ρ=0) | Sustained breach | No incentive to defend early; passive in downtrend | **High** |
| 1.5 | Fixed quote reserve, procyclical | ~72% price drop | Both sides bind → total paralysis | **High** |
| 1.6 | Hinge non-smooth + n=16 | Reserve-boundary straddle | λ limit-cycle oscillation | Medium |
| 2.1 | Score gaming (μ=100 ≫ return) | Validity enabled | Economically useless book | **Critical if enabled** |
| 2.2 | Fixed μ, no adaptation | μ misconfigured | No safety valve | High |
| 2.3 | Validity only at h=0 | horizon > 1 | Later steps unconstrained | Low–Medium |
| 2.4 | Validity/viability gradient conflict | Both active near boundary | Unpredictable book | Medium |
| 2.5 | `combined` measure ties to market depth | Market thinning | Quotes shrink when most needed | Medium |
| 3.1 | No hard reserve enforcement | Pathological sample | Reserve actually breached | **Critical** |
| 3.2 | λ frozen on optimizer timeout | Timeouts (observed 05:00, 22:00) | Stale λ, no decay | **High** |

---

## Recommendation Priorities

1. **Enable `objective_normalize`** — this is the single highest-leverage change. It makes `risk_adjusted_return` comparable in magnitude to the penalties, so viability becomes a *tradeoff* rather than an *override*.
2. **Add a hard reserve clamp post-solve** — never dispatch a book whose free quote balance falls below `min_viable_quote`, regardless of what the optimizer chose.
3. **Set `recovery_fraction > 0`** — gives the optimizer a reason to rest *some* bid capacity to defend the reserve, instead of going fully passive.
4. **Call `reset_viability_lambda` on config reload / restart**, and lower `lambda_max` substantially (2.5 already dominated99% of the objective; 100 is unbounded risk).
5. **Leave `validity.penalty_scale = 0` until `objective_normalize` is on** — otherwise validity at μ=100 will overwhelm everything.
6. **Move `update_viability_lambda` outside the cancellable region** so timeouts don't freeze dual state.