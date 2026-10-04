# Mathematical Formulation of the Price and Order-Flow Models

This document is the compact mathematical statement of the **models the live MPC
actually samples from** during rollout. It is the counterpart of
[`mpc-mathematical-formulation.md`](mpc-mathematical-formulation.md) for the
stochastic inputs $\varepsilon$ of the rollout operator $\Phi$. How those
volumes consume the relative-spread book (soft fill, queue share) is stated
in [`mpc-spread-grid-and-depth.md`](mpc-spread-grid-and-depth.md).

Source of truth is the current implementation, not research notes or unused
kinds. The runtime example and controller builder wire:

| Role | Config key | Kind | Class |
| --- | --- | --- | --- |
| Local / global flow | `controller.local_market_flow_model`, `.global_market_flow_model` | `hurdle_lognormal` | `TorchHurdleLogNormalModel` |
| Local / global price | `controller.local_price_model`, `.global_price_model` | `conditional_mechanical_mixture` | `TorchConditionalMechanicalMixturePriceModel` |

Primary references:

- `mm.mpc.market_order_flow_torch.TorchHurdleLogNormalModel`
- `mm.mpc.price_torch.TorchConditionalMechanicalMixturePriceModel`
- `mm.pricing.reference_prices_ticks` (mechanical book consume, fixed-depth VWAP)
- `mm.mpc.mpc_tensor_batched.MonteCarloMPCTensorBatched` (how samples enter $\Phi$)
- `mm.models.scoring` (prequential NLL diagnostics)

Kinds that exist in the builder but are **not** the current production models
(`residual`, `ewma_gaussian`, `hurdle_gamma`, `hurdle_weibull`, `gamma`,
`weibull`) are omitted.

---

## 1. Role in the MPC

At each decision $t$, the inner Adam loop evaluates $J(u;\xi_t,\theta_t)$
by Monte Carlo. For path $i=1,\ldots,N$ and horizon step $h=0,\ldots,H-1$:

1. Draw market-order volumes $(V^{+}_{i,h}, V^{-}_{i,h})$ from the
   **hurdle-lognormal flow model**.
2. Consume that flow against the public book (and the decoded quotes) via
   `soft_fill_batched`. This produces a **mechanical** next book and mid.
3. Draw a **conditional mechanical-mixture** price innovation and convert it
   to inventory-aware buy/sell VWAPs used to mark wealth.

Buy volume $V^{+}$ is aggressive buy flow (hits asks). Sell volume $V^{-}$
is aggressive sell flow (hits bids). The two sides are modelled as
**independent**. Local and global venues each keep their own parameter state
and are sampled independently.

With `optimizer.fixed_mc_samples = true`, the same draws are reused across all
Adam epochs of one `optimize()` call.

---

## 2. Market order-flow model (`hurdle_lognormal`)

### 2.1 Generative law

For each side $s \in \{+,-\}$ (code: buy / sell), the interval volume $V_s$
is a **hurdle** random variable: a point mass at zero, and a log-normal on the
positive part.

$$
\begin{aligned}
\mathbb{P}(V_s = 0) &= p_{0,s}, \\
V_s \mid V_s > 0 &\sim \mathrm{LogNormal}(\mu_s, \sigma_s^2).
\end{aligned}
$$

Equivalently, with $U_s \sim \mathrm{Unif}(0,1)$ independent of
$Z_s \sim \mathcal{N}(\mu_s,\sigma_s^2)$,

$$
V_s \;=\;
\begin{cases}
0, & U_s \le p_{0,s}, \\
\exp(Z_s), & U_s > p_{0,s}.
\end{cases}
$$

This is exactly `TorchHurdleLogNormalModel.sample()` (continuous, no rounding).
Buy and sell draws are independent, so

$$
\mathbb{P}(V^{+}, V^{-})
  = \mathbb{P}(V^{+})\,\mathbb{P}(V^{-}).
$$

The mean of one side is

$$
\mathbb{E}[V_s]
  = (1-p_{0,s})\,\exp\!\bigl(\mu_s + \tfrac12 \sigma_s^2\bigr).
$$

**Sampling vs scoring.** Diagnostics score *integer* lots
$k = \mathrm{round}(V_{\mathrm{obs}})$ with the discrete PMF

$$
\begin{aligned}
\mathbb{P}(X=0) &= p_0, \\
\mathbb{P}(X=k)
  &= (1-p_0)\,
     \frac{
       \Phi\!\bigl(\tfrac{\log(k+1/2)-\mu}{\sigma}\bigr)
       -\Phi\!\bigl(\tfrac{\log(k-1/2)-\mu}{\sigma}\bigr)
     }{
       1-\Phi\!\bigl(\tfrac{\log(1/2)-\mu}{\sigma}\bigr)
     },
     \quad k \ge 1,
\end{aligned}
$$

where $\Phi$ is the standard normal CDF. The MPC rollout uses the continuous
sampler above; it does not round.

### 2.2 Online parameter update (EWMA)

Let $\gamma \in (0,1)$ be `decay` (example: $0.95$) and
$\alpha = 1-\gamma$. On each closed interval the model sees integer lots
$x_s = \mathrm{round}(V_s^{\mathrm{obs}})$.

**Zero mass** (updated every step, including zeros):

$$
p_{0,s} \leftarrow
  \gamma\, p_{0,s} + \alpha\,\mathbf{1}_{\{x_s = 0\}},
  \qquad p_{0,s} \in [0,1].
$$

**Positive-part log moments** (updated only when $x_s > 0$):

$$
\begin{aligned}
S_s &\leftarrow \gamma S_s + \log x_s, \\
Q_s &\leftarrow \gamma Q_s + (\log x_s)^2, \\
W_s &\leftarrow \gamma W_s + 1, \\
\mu_s &= S_s / W_s, \\
\sigma_s^2 &= \max\bigl(\varepsilon^2,\; Q_s/W_s - \mu_s^2\bigr).
\end{aligned}
$$

Zeros freeze $(\mu_s,\sigma_s)$. That is the hurdle split: $p_{0,s}$ sees
all observations; $(\mu_s,\sigma_s)$ are EWMA moments of $\log x$ on the
positive subset only.

Initial values come from config (`init_p0`, `init_mu`, `init_sigma`). Until
the first positive observation, $(\mu_s,\sigma_s)$ stay at those initials.

### 2.3 What this model is *not*

- Sides are not jointly modelled (no copula, no imbalance factor).
- There is no intra-interval arrival process; one scalar volume per side per
  $\Delta t$.
- There is no dependence on the current book, mid, or inventory.

---

## 3. Price model (`conditional_mechanical_mixture`)

The price model does not sample a raw $\Delta \mathrm{mid}$ in isolation.
It predicts a **four-component book-shape transition**, then converts that
shape to **dynamic-depth buy/sell VWAPs** for marking.

### 3.1 Book features

Fix a reference depth $Q_{\mathrm{ref}} =$ `reference_vwap_vol` (example:
$1000$ local, $100$ global). On an order book $\mathcal{B}$ with mid
$m$ and best spread $S = a-b$,

$$
\begin{aligned}
P^{\mathrm{buy}}(\mathcal{B})
  &= \mathrm{VWAP}_{\mathrm{ask}}(\mathcal{B}; Q_{\mathrm{ref}}), \\
P^{\mathrm{sell}}(\mathcal{B})
  &= \mathrm{VWAP}_{\mathrm{bid}}(\mathcal{B}; Q_{\mathrm{ref}}), \\
\psi^{\mathrm{buy}}
  &= \max\bigl(\varepsilon,\; P^{\mathrm{buy}} - m - S/2\bigr), \\
\psi^{\mathrm{sell}}
  &= \max\bigl(\varepsilon,\; m - P^{\mathrm{sell}} - S/2\bigr).
\end{aligned}
$$

$\psi^{\mathrm{buy}}$ ($\psi^{\mathrm{sell}}$) is extra ask (bid) shape
beyond half-spread at the fixed reference depth.

The four **state components** are

$$
f(\mathcal{B})
  = \bigl(m,\; \log S,\; \log \psi^{\mathrm{buy}},\; \log \psi^{\mathrm{sell}}\bigr).
$$

A transition from book $\mathcal{B}$ to book $\mathcal{B}'$ is

$$
\Delta f = f(\mathcal{B}') - f(\mathcal{B}).
$$

Components are indexed $c \in \{\mathrm{mid},\; \log S,\; \log\psi^{+},\; \log\psi^{-}\}$.

### 3.2 Mechanical (market-order-flow-only) transition

Given observed or sampled flow $(V^{+}, V^{-})$ and a starting book
$\mathcal{B}$, the mechanical next book $\mathcal{B}^{\mathrm{mech}}$ is
obtained by consuming volume against standing depth, best-first:

- $V^{-}$ walks the bid side (highest price first),
- $V^{+}$ walks the ask side (lowest price first),
- no limit-order replenishment.

The mechanical feature move is $\Delta f^{\mathrm{mech}} = f(\mathcal{B}^{\mathrm{mech}}) - f(\mathcal{B})$.

On **live updates**, $\mathcal{B}^{\mathrm{mech}}$ is this MOF-only consume
of the previous public book; the *observed* move $\Delta f^{\mathrm{obs}}$
uses the actual next book (which also includes refresh / new limits).

On **MPC rollout**, $\mathcal{B}^{\mathrm{mech}}$ is the book after
`soft_fill_batched` (sampled flow plus the bot’s own fills). That is the
mechanical feature the sampler conditions on.

### 3.3 Two-regime mixture

Let $V_{\Sigma} = V^{+} + V^{-}$ and

$$
x_{\mathrm{gate}} = \bigl(1,\; \min(\log(1+V_{\Sigma}),\, 12)\bigr)^\top.
$$

The **market-event probability** (flow gate) is

$$
q
  = \mathrm{clip}\bigl(
      \sigma(\beta_{\mathrm{gate}}^\top x_{\mathrm{gate}}),\;
      0.001,\; 1 - w_{\mathrm{refresh}}^{\min}
    \bigr),
$$

with logistic $\sigma(z) = (1+e^{-z})^{-1}$ and default `min_refresh_weight` $w_{\mathrm{refresh}}^{\min} = 0.15$ ($q \le 0.85$). High total flow raises $q$
when $\beta_{\mathrm{gate},1} > 0$.

Enforcing the refresh floor $w_{\mathrm{refresh}}^{\min} > 0$ prevents **gate saturation**: under heavy, sustained flow where $\beta_{\mathrm{gate}}^\top x_{\mathrm{gate}} \gg 0$, an uncapped gate would push $q \to 1.0$. If a large quote jump or sudden order book shift occurs during that flow burst, the mechanical Gaussian likelihood severely underflows. Capping $q \le 0.85$ (leaving at least $15\%$ weight on the heavy-tailed Student-$t$ innovation branch) preserves tail defense at all times and eliminates likelihood blow-ups.

For each component $c$, the predictive law of the realized move
$\Delta f_c$ is the mixture

$$
\Delta f_c
  \;\sim\;
  q\cdot
    \mathcal{N}\!\bigl(
      \mu^{\mathrm{mkt}}_c,\;
      (\sigma^{\mathrm{mkt}}_c)^2
    \bigr)
  + (1-q)\cdot
    t_{\nu}\!\bigl(
      \mu^{\mathrm{ref}}_c,\;
      (\sigma^{\mathrm{ref}}_c)^2
    \bigr).
$$

**Market regime** (flow-driven, Gaussian), linear in the mechanical move:

$$
\mu^{\mathrm{mkt}}_c
  = \beta_{c,0} + \beta_{c,1}\,\Delta f^{\mathrm{mech}}_c.
$$

**Refresh regime** (quote revision with little or no flow, Student-$t$):

$$
\mu^{\mathrm{ref}}_c = m_c
  \qquad\text{(learned intercept, no mechanical feature)}.
$$

The Student-$t$ is parameterized so its variance equals
$(\sigma^{\mathrm{ref}}_c)^2$: a standard $t_{\nu}$ draw $T$ is scaled by

$$
\sigma^{\mathrm{ref}}_c \sqrt{(\nu-2)/\nu},
$$

with $\nu =$ `student_df` (example: $3$). This matches the stored
refresh variance used in the log-density.

Default `point_mode = "sample"` draws the mixture (Bernoulli gate, then
Gaussian or scaled-$t$). Alternatives: `"mean"` returns the mixture mean;
`"map"` returns the mean of the regime with $q \ge 1/2$.

The four component innovations are drawn independently given $q$ and
$\Delta f^{\mathrm{mech}}$. The same Bernoulli $q$ is shared across
components (one market/refresh choice per step).

### 3.4 Reconstructing prices and dynamic VWAP

Let $\Delta\hat f$ be the sampled (or mean) component vector. Then

$$
\begin{aligned}
\hat m &= m + \Delta\hat f_{\mathrm{mid}}, \\
\hat S &= S \cdot \exp(\Delta\hat f_{\log S}), \\
\hat\psi^{+} &= \psi^{+} \cdot \exp(\Delta\hat f_{\log\psi^{+}}), \\
\hat\psi^{-} &= \psi^{-} \cdot \exp(\Delta\hat f_{\log\psi^{-}}),
\end{aligned}
$$

with log-increments clamped to $[-5,5]$.

Marking uses **inventory-aware depth**, not $Q_{\mathrm{ref}}$:

$$
d^{+} = q_{\mathrm{cash}} / \hat m, \qquad d^{-} = |b_{\mathrm{inv}}|.
$$

A shape-depth ratio $r(d; Q_{\mathrm{ref}}, \alpha)$ rescales extra shape
from the reference depth to the mark depth (`shape_alpha` $\alpha$, example
$0.5$):

$$
r(d)
  =
  \mathrm{clip}\!
  \left(
    \frac{
      \bigl((d+\delta)/Q_{\mathrm{ref}}\bigr)^{\alpha}
      - (\delta/Q_{\mathrm{ref}})^{\alpha}
    }{
      \bigl((Q_{\mathrm{ref}}+\delta)/Q_{\mathrm{ref}}\bigr)^{\alpha}
      - (\delta/Q_{\mathrm{ref}})^{\alpha}
    },\; 0,\; \infty
  \right),
$$

with shift $\delta = \max(10^{-9}, 10^{-6} Q_{\mathrm{ref}})$ so the map is
$0$ at $d=0$ and has finite slope for autograd. If $\alpha = 0$,
$r = \mathbf{1}_{d>0}$.

Dynamic VWAPs returned by `sample_dynamic_vwap`:

$$
\begin{aligned}
\mathrm{VWAP}^{\mathrm{buy}}
  &= \hat m + \hat S/2 + \hat\psi^{+}\, r(d^{+}), \\
\mathrm{VWAP}^{\mathrm{sell}}
  &= \hat m - \hat S/2 - \hat\psi^{-}\, r(d^{-}).
\end{aligned}
$$

These are the marks that enter $\Delta W$ in the CE term of $J$.

### 3.5 Online parameter update

Updates run **prequentially**: score the new interval with parameters from
the previous interval, then `update(prev_step, curr_step)`.

Let $\gamma =$ `decay` (example: $0.99$), $\alpha = 1-\gamma$.
Compute $\Delta f^{\mathrm{obs}}$ from the actual next book and
$\Delta f^{\mathrm{mech}}$ from MOF-only consume of the previous book.

**Responsibilities.** For each component, market and refresh log-densities
are summed (components treated as independent given the regime), then

$$
\begin{aligned}
\ell_{\mathrm{mkt}}
  &= \log q + \sum_c \log \mathcal{N}(\Delta f^{\mathrm{obs}}_c; \mu^{\mathrm{mkt}}_c, (\sigma^{\mathrm{mkt}}_c)^2), \\
\ell_{\mathrm{ref}}
  &= \log(1-q) + \sum_c \log t_{\nu}(\Delta f^{\mathrm{obs}}_c; \mu^{\mathrm{ref}}_c, (\sigma^{\mathrm{ref}}_c)^2), \\
r_{\mathrm{mkt}}
  &= \frac{e^{\ell_{\mathrm{mkt}}}}{e^{\ell_{\mathrm{mkt}}}+e^{\ell_{\mathrm{ref}}}},
  \qquad
  r_{\mathrm{ref}} = 1 - r_{\mathrm{mkt}}.
\end{aligned}
$$

**Gate (logistic gradient step):**

$$
\beta_{\mathrm{gate}}
  \leftarrow
  \mathrm{clip}\bigl(
    \beta_{\mathrm{gate}} + \alpha\,(r_{\mathrm{mkt}} - q)\, x_{\mathrm{gate}},\;
    -20,\; 20
  \bigr).
$$

**Market means (weighted RLS with forgetting $\gamma$)** for each $c$,
with feature $x_c = (1,\; \Delta f^{\mathrm{mech}}_c)^\top$ and weight
$w = r_{\mathrm{mkt}}$:

$$
\begin{aligned}
x_w &= \sqrt{w}\, x_c, \quad
y_w = \sqrt{w}\, \Delta f^{\mathrm{obs}}_c, \\
k &= \frac{P_c x_w}{\gamma + x_w^\top P_c x_w}, \\
\beta_c &\leftarrow \beta_c + k\,(y_w - x_w^\top \beta_c), \\
P_c &\leftarrow \gamma^{-1}\bigl(P_c - k x_w^\top P_c\bigr).
\end{aligned}
$$

If $w \approx 0$, $\beta_c$ is left unchanged. $P_c$ is initialized as
$P_0 I$ with $P_0 =$ `rls_p0`.

**Refresh means (responsibility-weighted EWMA):**

$$
m_c \leftarrow m_c + \alpha\, r_{\mathrm{ref}}\,(\Delta f^{\mathrm{obs}}_c - m_c).
$$

**Variances:**

$$
\begin{aligned}
(\sigma^{\mathrm{mkt}}_c)^2
  &\leftarrow
  (1 - \alpha r_{\mathrm{mkt}})(\sigma^{\mathrm{mkt}}_c)^2
  + \alpha r_{\mathrm{mkt}}\,(\Delta f^{\mathrm{obs}}_c - \mu^{\mathrm{mkt}}_c)^2, \\
(\sigma^{\mathrm{ref}}_c)^2
  &\leftarrow
  (1 - \alpha r_{\mathrm{ref}})(\sigma^{\mathrm{ref}}_c)^2
  + \alpha r_{\mathrm{ref}}\,(\Delta f^{\mathrm{obs}}_c - m_c)^2,
\end{aligned}
$$

floored at $\sigma_{\min}^2$. Mid starts from `initial_sigma`; log
components from `initial_log_sigma`.

### 3.6 Predictive density used in diagnostics

Prequential NLL for the mid component (the logged score) is the mixture
density at the realized mid change $\Delta m$:

$$
p(\Delta m)
  = q\,\varphi\!\bigl(\Delta m; \mu^{\mathrm{mkt}}_{\mathrm{mid}}, \sigma^{\mathrm{mkt}}_{\mathrm{mid}}\bigr)
  + (1-q)\, t_{\nu}\!\bigl(\Delta m; \mu^{\mathrm{ref}}_{\mathrm{mid}}, \sigma^{\mathrm{ref}}_{\mathrm{mid}}\bigr),
$$

with $q$ from the flow gate and $\mu^{\mathrm{mkt}}_{\mathrm{mid}} = \beta_{\mathrm{mid},0} + \beta_{\mathrm{mid},1}\,\Delta m^{\mathrm{mech}}$.
This is `score_conditional_mechanical_mid_delta`.

---

## 4. How the two models compose in one rollout step

For path $i$, step $h$, with current book $\mathcal{B}_{i,h}$, wallet
$(b_{i,h}, q_{i,h})$, and decoded quotes $u^{(h)}$:

$$
\begin{aligned}
(V^{+}_{i,h}, V^{-}_{i,h})
  &\sim \text{hurdle-lognormal}, \\
(\mathcal{B}_{i,h+1}^{\mathrm{mech}},\; \Delta\mathrm{cash},\; \Delta\mathrm{inv})
  &= \mathrm{soft\_fill}(u^{(h)}, \mathcal{B}_{i,h}, V^{+}_{i,h}, V^{-}_{i,h}), \\
\Delta f^{\mathrm{mech}}_{i,h}
  &= f(\mathcal{B}_{i,h+1}^{\mathrm{mech}}) - f(\mathcal{B}_{i,h}), \\
q_{i,h}
  &= \sigma\!\bigl(\beta_{\mathrm{gate}}^\top (1, \log(1+V^{+}_{i,h}+V^{-}_{i,h}))\bigr), \\
\Delta \hat f_{i,h}
  &\sim q_{i,h}\,\mathcal{N}(\beta_{0}+\beta_{1}\odot\Delta f^{\mathrm{mech}}_{i,h}, \sigma^{\mathrm{mkt}2})
     + (1-q_{i,h})\, t_{\nu}(m, \sigma^{\mathrm{ref}2}), \\
(\mathrm{VWAP}^{+}_{i,h+1}, \mathrm{VWAP}^{-}_{i,h+1})
  &= \Gamma\!\bigl(\Delta\hat f_{i,h},\; d^{+}_{i,h+1},\; d^{-}_{i,h+1}\bigr).
\end{aligned}
$$

$\Gamma$ is the reconstruction in §3.4. Wealth marking uses those VWAPs
(bid VWAP when inventory is short, sell VWAP when long). The flow model
does not see the price model; the price model *does* see the sampled flow,
both through the gate $q$ and through $\Delta f^{\mathrm{mech}}$.

---

## 5. Example config (runtime example)

From `mm/configs/market_maker.runtime.example.json`:

```text
local/global_market_flow_model:
  kind: hurdle_lognormal
  decay γ = 0.95
  init_p0 = 0.3, init_μ = 0, init_σ = 1

local_price_model:
  kind: conditional_mechanical_mixture
  reference_vwap_vol Q_ref = 1000
  shape_alpha α = 0.5
  decay γ = 0.99
  initial_sigma = 50, initial_log_sigma = 0.25
  student_df ν = 3
  point_mode = sample

global_price_model: same kind, Q_ref = 100
```

---

## 6. Symbol map

| Symbol | Code / config |
| --- | --- |
| $V^{+}, V^{-}$ | `buy_vol`, `sell_vol` |
| $p_{0,s}, \mu_s, \sigma_s$ | `p0_buy/sell`, `mu_buy/sell`, `sigma_buy/sell` |
| $\gamma_{\mathrm{flow}}$ | `local_market_flow_model.decay` |
| $Q_{\mathrm{ref}}$ | `reference_vwap_vol` |
| $\psi^{+}, \psi^{-}$ | `buy_shape`, `sell_shape` |
| $\Delta f^{\mathrm{mech}}$ | mechanical component deltas |
| $q$ | `market_prob` (gate) |
| $r_{\mathrm{mkt}}$ | `market_resp` (posterior, update only) |
| $\beta_{\mathrm{gate}}$ | `gate_beta` |
| $\beta_c$ | `market_beta[c]` |
| $m_c$ | `refresh_mean[c]` |
| $\sigma^{\mathrm{mkt}}_c, \sigma^{\mathrm{ref}}_c$ | `market_sigma`, `refresh_sigma` |
| $\nu$ | `student_df` |
| $w_{\mathrm{refresh}}^{\min}$ | `min_refresh_weight` |
| $\alpha$ (shape) | `shape_alpha` |
| $\gamma_{\mathrm{price}}$ | `local_price_model.decay` |
