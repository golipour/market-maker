# Viability term (availability functional)

Availability measures whether the wallet can sustain two-sided quoting at a configured **minimum viable volume**, pathwise over the MPC horizon. It is **not** about order count or free-vs-locked splits.

## 1. State

\[
s_t = (p_t,\, b_t,\, q_t,\, \ell_t^b,\, \ell_t^q)
\]

| Symbol | Meaning |
|--------|---------|
| \(p_t\) | Mid price (quote per base) |
| \(b_t\) | **Total** base balance (free + locked) |
| \(q_t\) | **Total** quote balance (free + locked) |
| \(\ell_t^b, \ell_t^q\) | Locked in open orders |

Locked funds are not gone — they release or convert. Availability is defined on **total balances** \((b_t, q_t)\), not free-only snapshots.

## 2. Minimum viable volume (single threshold)

Define one operational floor in quote currency:

\[
q_{\min} \;=\; \texttt{min\_viable\_quote}
\]

This is the minimum **quote-side wallet volume** you require to remain viable on bids — a business parameter, not exchange min-notional per order.

Derive the price-consistent base floor:

\[
b_{\min}(p_t) = \frac{q_{\min}}{p_t}
\]

## 3. Hard feasibility

\[
\text{bid viable} \iff q_t \ge q_{\min}
\]
\[
\text{ask viable} \iff b_t \ge b_{\min}(p_t) = \frac{q_{\min}}{p_t}
\]

Two-sided availability (hard):

\[
A_t = \mathbf{1}(q_t \ge q_{\min}) \cdot \mathbf{1}\!\left(b_t \ge \frac{q_{\min}}{p_t}\right)
\]

Example at \(p_t = 100{,}000\) IRT/USDT and \(q_{\min} = 50{,}000\) IRT:

| Side | Total balance checked | Floor |
|------|----------------------|-------|
| Bid | \(q_t\) (IRT) | 50,000 IRT |
| Ask | \(b_t\) (USDT) | 0.5 USDT |

## 4. Soft availability (for MPC gradients)

Binary feasibility is not differentiable. Use smooth margins:

\[
m_t^{\text{bid}} = q_t - q_{\min}, \qquad
m_t^{\text{ask}} = b_t - \frac{q_{\min}}{p_t}
\]

\[
a_t^{\text{bid}} = \sigma\!\left(\frac{m_t^{\text{bid}}}{\delta}\right), \qquad
a_t^{\text{ask}} = \sigma\!\left(\frac{m_t^{\text{ask}}}{\delta / p_t}\right)
\]

\[
A_t = a_t^{\text{bid}} \cdot a_t^{\text{ask}}
\]

\(\delta = \texttt{margin\_buffer}\) is a smoothing scale in quote currency (fees, latency, rounding). Base-side smoothing uses \(\delta / p_t\) for price consistency.

## 5. Pathwise availability risk

For rollout path \(i\):

\[
A_{\min}^{(i)} = \min_{h=0..H} A_{t+h}^{(i)}
\]

Captures wallet drift, fill sequences, and asymmetric depletion along simulated paths. Rollout records **total** balances after each horizon step.

## 6. Shortfall functional

Required level \(A^\star \approx 0.99\) (`availability_target`).

Per path:

\[
S^{(i)} = \max(0,\, A^\star - A_{\min}^{(i)})
\]

\[
\mathcal{S} = \mathbb{E}[S]
\]

## 7. Objective coupling

Dual form (scalar objective):

\[
J = CE - \lambda_t \cdot \mathcal{S}
\]

\(\lambda_t\) adapts from shortfall vs `shortfall_tolerance` \(\epsilon\); it is not hand-tuned like a fixed \(w_{\text{inv}}\).

## 8. Config keys

| Key | Role |
|-----|------|
| `min_viable_quote` | \(q_{\min}\): minimum total quote-side volume for bid viability |
| `margin_buffer` | \(\delta\): sigmoid smoothing scale (quote currency) |
| `availability_target` | \(A^\star\) |
| `shortfall_tolerance` | \(\epsilon\) for \(\lambda\) update |
| `lambda_init`, `lambda_learning_rate` | Dual state |

## 9. What this is not

- Not a target for displayed order-book depth or quoted flow.
- Not sensitive to how many orders can be placed — only whether total wallet volume stays above the viable floor on both sides.
- Not free-only balances — locked order collateral counts toward \(q_t\) and \(b_t\).
