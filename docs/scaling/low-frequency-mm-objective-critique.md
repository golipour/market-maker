# Low-frequency market maker: objective and architecture critique

This document critiques the current market-making objective and structural constraints as implemented in the codebase, with a focus on what is wrong today and how a low-frequency market maker should be built instead.

The stated goal is a **profitable, balanced, less risky order book with good depth and spread** — with a tunable trade-off between **risk-adjusted return** and **book-quality** objectives.

---

## What the objective actually is today

The terminal objective in `mm/mpc/reward_tensor.py` is:

```python
J = risk_adjusted_return
```

where `risk_adjusted_return` is an entropic certainty equivalent over Monte Carlo path returns:

```
CE = min(R) - (1/beta) * ln(mean_i exp(-beta * (R_i - min(R))))
```

Everything else — min/max size per side, imbalance skew floors, capacity ratio, spread caps — is enforced as a **hard feasibility box** (`_quote_capacity_bounds_batched` in `mm/mpc/mpc_tensor_batched.py`), not as part of `J`.

That mismatch between the stated goal and what the optimizer actually maximizes is the root of most problems below.

---

## What is wrong

### 1. The "good orderbook" objective does not exist in the objective

The desired trade-off is between risk-adjusted return and book quality (depth, balance, spread). There is no book-quality term in `J`. Depth, balance, and spread are enforced only as constraints:

- `quote_size_min_per_side` (inflated by `quote_imbalance_skew_strength`)
- `quote_spread_max`
- `quote_capacity_ratio`
- hard caps on per-side size

Consequences:

- Whenever quoting is negative expected value under the models, the optimizer sits **exactly at the floor**. "Good depth" degenerates into minimum compliance, and the trade-off cannot be tuned — there is only a wall to move, not a weight to turn.
- Floors can force quoting even when it is adversely selected, and **the cost of that obligation is invisible**. A constraint has a shadow price; it is never computed, so it is unclear whether book-quality requirements cost 5% or 80% of P&L.

If book quality is a real obligation (for example, an exchange market-making program), model it as explicit reward terms: presence/uptime, two-sided spread tightness, quoted depth within a band — each with weights. Then the trade-off becomes tunable, and diagnostics can show what each obligation costs.

### 2. Risk is entirely model-mediated, and the models are the weakest link

The entropic certainty equivalent only sees risk that the price and flow models generate. Two specific failures:

**Adverse selection is structurally absent.** In the rollout, price innovations and flow consumption are sampled independently; a fill is not correlated with the subsequent price move. In reality, a bid gets filled disproportionately right before price trades through it. An optimizer blind to this systematically overestimates EV at tight spreads — it will quote too tight, and "risk-adjusted" does little because the risk is not in the simulated paths.

**The horizon is too short for inventory risk.** With `n_samples=32` and `horizon=4`, the CE estimate is noisy. Entropic utility is tail-sensitive; 32 paths cannot estimate tails reliably, so `beta` may fit sample noise rather than a true risk preference. The short horizon also truncates inventory carry risk with **no terminal inventory valuation** — classic MPC myopia: the rollout can end holding a position whose liquidation cost is never charged.

There is no explicit inventory carry penalty (for example, proportional to \(q^2 \sigma^2 \tau\)). Inventory price risk is delegated entirely to the entropic CE over simulated paths.

### 3. The fill model optimizes for differentiability, not realism

The soft-fill apparatus — sigmoid fill fractions, `back_of_queue` share approximation, remainder-canceled assumptions in the rollout — exists so Adam can backprop through many epochs. At low frequency:

- **Queue priority is one of the largest real edges, and the optimizer cannot see it.** Each frame re-optimizes from scratch. Keeping an existing resting order (and its queue position) has value that `J` never credits. The reconcile planner downstream reduces churn mechanically, but the optimizer has no cost for cancel/replace, so it freely proposes books that destroy priority.
- Non-finite guards scattered through the loop (`nan_to_num`, zeroed gradients, fallback actions) indicate a numerically fragile surrogate rather than a robust formulation.

### 4. Risk and execution constraints are applied after the decision, with no feedback

The net-delta risk engine vetoes or clips actions post-MPC. Venue order limits and quantization reshape the ladder at the execution edge. The optimizer's EV estimate is for a book that may never reach the exchange in that form.

Structurally, the feasible set should be **inside** the optimization (or the optimizer should be told what was clipped and why). Otherwise you get persistent propose→reject churn, and diagnostics measure a fiction.

### 5. "Balance" emerges from the wallet, not from the market

Capacity bounds derive bid size from cash and ask size from inventory. After a run of buys, the system structurally cannot quote the ask. Displayed book symmetry becomes a random walk driven by fill history — the opposite of a stable, balanced book.

### 6. Objective terms are not dimensionally comparable

The certainty-equivalent return is dimensionless per-frame return. Any future book-quality or churn terms must be scaled carefully. Tuned weights will not transfer across symbols, wallet sizes, or volatility regimes without recalibration.

---

## How a low-frequency market maker should be implemented

The uncomfortable claim: **at 1s+ intervals on thin venues, a fragile high-dimensional differentiable MPC is the wrong primary tool.** Most P&L variance comes from fair-value error and adverse selection, not from optimizing a continuous depth profile. Complexity is spent in the wrong place.

### 1. Fair value first

A robust anchor (local microprice blended with a global reference venue, staleness-aware) is the highest-leverage component. The plumbing exists (`mm/pricing/anchor.py`); investment should go into making this estimate good and measuring its error.

### 2. Measure, do not simulate, the two quantities that matter

Use the ledger and fill history empirically:

- **Fill intensity vs. distance from fair value** — fit from your own fills, per venue.
- **Markout per fill** (price drift 1–5 minutes after each fill) — this **is** adverse selection cost, measured, not assumed. Subtract it from spread capture in the objective.

### 3. Shrink the action space

Bid/ask skew around fair value, per-side size, and one or two ladder-shape parameters: roughly 3–5 numbers. Then a differentiable surrogate is less necessary — evaluate candidates on a coarse grid or with CEM against a **realistic, discrete** fill simulator (exact queue positions, exact quantization, actual venue limits). Same models, no soft-fill sigmoid, no Adam instability, and the optimizer evaluates the book that will actually be placed.

### 4. Price inventory explicitly

A carry penalty proportional to \(\sigma^2 q^2 \tau\) (Avellaneda–Stoikov style), plus a terminal liquidation cost if any horizon rollout is kept. Do not rely on entropic utility over a small number of paths to discover that holding inventory is risky.

### 5. Make book quality an explicit, weighted reward

If there are obligations — uptime, two-sided presence, quoted depth within X bps — add them to `J` with weights. Then the stated trade-off is one scalar (or a small set of weights), and you can plot P&L vs. book quality as you sweep them.

If there are no obligations, drop the floors entirely: a correctly priced MM produces a good book as a byproduct. Forced quoting at negative EV is just a donation.

### 6. Charge for churn inside the decision

Cancel/replace cost, queue-priority value, and hysteresis (only re-quote when the target moves more than a band) belong in the objective, not only in the reconcile planner.

### 7. Move risk limits into the feasible set of the optimizer

Keep the downstream risk engine only as a hard backstop (kill switch, fat-finger limits).

### 8. Close the loop

Model scoring exists (`mm/models/scoring.py`). Extend it to compare **predicted** fill rates and markouts vs. **realized**, per regime. On thin venues, regime shifts dominate any static calibration.

---

## Summary

The architecture optimizes a precise answer to the wrong question: a model-simulated return with no adverse selection, no queue value, weak inventory carry modeling, and book quality jammed in as unpriced hard constraints.

Invert the design:

1. **Measured** adverse selection and inventory cost in a small, honest objective.
2. **Book quality** as explicit weighted terms (or no floors if not required).
3. **Simple search** over a small action space against an exact execution simulator.

That is a better fit for a low-frequency market maker than tuning a high-dimensional differentiable MPC whose main failure modes are invisible to the objective.
