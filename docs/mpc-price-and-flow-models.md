# Mathematical Formulation of the Price and Order-Flow Models

This document is the compact mathematical statement of the **models the live MPC
actually samples from** during rollout. It is the counterpart of
[`mpc-mathematical-formulation.md`](mpc-mathematical-formulation.md) for the
stochastic inputs $\varepsilon$ of the rollout operator $\Phi$. How those
volumes consume the relative-spread book (soft fill, queue share) is stated
in [`mpc-spread-grid-and-depth.md`](mpc-spread-grid-and-depth.md).

Source of truth is the current implementation, not research notes or unused
kinds. The runtime example and controller builder wire:

| Role | Config key | Kind | Class | Typical asset / venue |
| --- | --- | --- | --- | --- |
| Local / global flow | `controller.local_market_flow_model`, `.global_market_flow_model` | `hurdle_lognormal` | `TorchHurdleLogNormalModel` | All pairs (with $\sigma_{\mathrm{floor}} \ge 0.35$) |
| Local / global price | `controller.local_price_model`, `.global_price_model` | `hurdle_student_t` | `TorchHurdleStudentTPriceModel` | USDT_IRT (Nobitex, Bitwana) |
| Local / global price | `controller.local_price_model`, `.global_price_model` | `hurdle_lognormal` | `TorchHurdleLogNormalPriceModel` | BTC_IRT (multi-magnitude jumps) |
| Local / global price (legacy) | `controller.local_price_model`, `.global_price_model` | `conditional_mechanical_mixture` | `TorchConditionalMechanicalMixturePriceModel` | Experimental 4-component mixture |

Primary references:

- `mm.mpc.market_order_flow_torch.TorchHurdleLogNormalModel`
- `mm.mpc.price_torch.TorchHurdleStudentTPriceModel`
- `mm.mpc.price_torch.TorchHurdleLogNormalPriceModel`
- `mm.mpc.price_torch.TorchConditionalMechanicalMixturePriceModel`
- `mm.pricing.reference_prices_ticks` (mechanical book consume, fixed-depth VWAP)
- `mm.mpc.mpc_tensor_batched.MonteCarloMPCTensorBatched` (how samples enter $\Phi$)
- `mm.models.scoring` (prequential NLL diagnostics)

Kinds that exist in the builder as legacy/baselines (`residual`, `ewma_gaussian`,
`hurdle_gamma`, `hurdle_weibull`, `gamma`, `weibull`) are omitted.

---

## 1. Role in the MPC

At each decision $t$, the inner Adam loop evaluates $J(u;\xi_t,\theta_t)$
by Monte Carlo. For path $i=1,\ldots,N$ and horizon step $h=0,\ldots,H-1$:

1. Draw market-order volumes $(V^{+}_{i,h}, V^{-}_{i,h})$ from the
   **regularized hurdle-lognormal flow model**.
2. Consume that flow against the public book (and the decoded quotes) via
   `soft_fill_batched`. This produces a **mechanical** next book and mid.
3. Draw price innovations:
   - In **discrete hurdle models** (`hurdle_student_t`, `hurdle_lognormal`),
     sample independent hurdle innovations $(\varepsilon^{\mathrm{buy}}, \varepsilon^{\mathrm{sell}})$,
     leaving book VWAPs unchanged with probability $p_0$ and perturbing them by
     heavy-tailed Student-$t$ (or signed log-normal) jumps with probability $1-p_0$.
   - In the **conditional mechanical mixture** (`conditional_mechanical_mixture`),
     draw a 4-component feature innovation conditional on mechanical flow, then
     reconstruct dynamic-depth buy/sell VWAPs.

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

This is `TorchHurdleLogNormalModel.sample()` (continuous, no rounding).
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

### 2.2 Online parameter update (EWMA) with variance floor

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
\sigma_s^2 &= \max\bigl(\sigma_{\mathrm{floor}}^2,\; Q_s/W_s - \mu_s^2\bigr).
\end{aligned}
$$

**Variance floor rationale:** The prior floor $\sigma_{\mathrm{floor}} = 0.35$ (`sigma_floor`)
prevents variance collapse. On low-liquidity books (such as Bitwana, where
$>98\%$ of intervals have zero trades, or where consecutive trades have
identical lot sizes), empirical sample variance $Q_s/W_s - \mu_s^2 \to 0$. Without
a floor, $\sigma_s \to 0$, causing standard score $z = (\log x - \mu)/\sigma$
to diverge into tens of thousands of standard deviations whenever an unexpected
fill occurs, producing infinite NLL penalties. The variance floor guarantees
numerical stability across all liquidity regimes.

Initial values come from config (`init_p0`, `init_mu`, `init_sigma`). Until
the first positive observation, $(\mu_s,\sigma_s)$ stay at those initials.

### 2.3 What this model is *not*

- Sides are not jointly modelled (no copula, no imbalance factor).
- There is no intra-interval arrival process; one scalar volume per side per
  $\Delta t$.
- There is no dependence on the current book, mid, or inventory.

---

## 3. Discrete Lattice Price Models

### 3.1 Empirical motivation

Real exchange mid-prices are quantized on discrete tick lattices with quantum
$h$ ($h = 0.5$ IRT for USDT pairs, $h = 100$ IRT for BTC pairs). Historical
data reveals two structural realities:

1. **High zero probability:** Between $60\%$ and $95\%$ of 10-second intervals
   have zero mid-price move ($\Delta\mathrm{mid} = 0$). Continuous models lack
   an atom at zero and misallocate probability mass to fractional ticks.
2. **Heavy-tailed jumps:** Realized jumps exhibit severe excess kurtosis that
   Gaussian distributions underestimate, leading to adverse selection underpricing.

Discrete hurdle models separate stationary book intervals from jump sizes and
evaluate exact discrete probability mass functions (PMFs) summing to $1.0$.

---

### 3.2 Hurdle Student-$t$ Price Model (`hurdle_student_t`)

Designed for stationary quote assets (USDT_IRT on Nobitex and Bitwana).

#### Discrete Generative Law

Let $h$ be the tick quantum (`tick_size`, e.g. $0.5$) and $\Delta\mathrm{mid} \in \{k \cdot h : k \in \mathbb{Z}\}$.
The transition probability mass function is:

$$
\mathbb{P}(\Delta\mathrm{mid} = 0) = p_0,
$$

and for non-zero tick jumps $k \in \mathbb{Z} \setminus \{0\}$:

$$
\mathbb{P}(\Delta\mathrm{mid} = k \cdot h)
  = (1 - p_0) \cdot
    \frac{
      T_{\nu}\!\left(\frac{k h + h/2}{s}\right)
      - T_{\nu}\!\left(\frac{k h - h/2}{s}\right)
    }{
      1 - \left[
        T_{\nu}\!\left(\frac{h/2}{s}\right)
        - T_{\nu}\!\left(-\frac{h/2}{s}\right)
      \right]
    },
$$

where $T_{\nu}$ is the standard Student-$t$ CDF with degrees of freedom
$\nu$ (`student_df`, empirically $\nu = 2.5$) and scale parameter $s$.
Because the denominator subtracts the zero-bin mass $[-h/2, h/2]$,
the discrete PMF analytically satisfies:

$$
\sum_{k=-\infty}^{\infty} \mathbb{P}(\Delta\mathrm{mid} = k \cdot h) \equiv 1.0.
$$

#### Online Parameter Updates

Updates track zero probability, conditional non-zero variance, and online
tail calibration factor $\alpha$:

1. **Zero mass update:**
   $$
   z_t = \mathbf{1}_{\{|\Delta\mathrm{mid}| < h/2\}}, \qquad
   p_0 \leftarrow \gamma_p\, p_0 + (1 - \gamma_p)\, z_t.
   $$
2. **Conditional second moment:** On non-zero steps ($z_t = 0$),
   $$
   m_2 \leftarrow \gamma_s\, m_2 + (1 - \gamma_s)\, (\Delta\mathrm{mid})^2.
   $$
   Raw scale:
   $$
   s_{\mathrm{raw}} = \sqrt{\max\!\left(m_2 \frac{\nu - 2}{\nu},\; \sigma_{\min}^2\right)}.
   $$
3. **Online tail calibration factor $\alpha$:**
   The theoretical fourth-moment Student-$t$ ratio is $u_{\mathrm{tgt}}^2 = \frac{\nu}{\nu - 2}$.
   Given observed normalized jump $u_{\mathrm{obs}} = \Delta\mathrm{mid} / s_{\mathrm{raw}}$:
   $$
   \alpha \leftarrow
     \gamma_{\mathrm{cal}}\,\alpha
     + (1 - \gamma_{\mathrm{cal}})\,\sqrt{\mathrm{clip}\!\left(\frac{u_{\mathrm{obs}}^2}{u_{\mathrm{tgt}}^2},\; 0.1,\; 10.0\right)},
   \qquad \alpha \in [0.2,\; 5.0].
   $$
   Effective scale:
   $$
   s = \max(\sigma_{\min},\; s_{\mathrm{raw}} \cdot \alpha).
   $$

#### Rollout Sampling

Inside the rollout operator $\Phi$, `sample_residual()` draws independent
buy and sell innovations:

$$
U \sim \mathrm{Unif}(0, 1), \quad T \sim t_{\nu},
\qquad
\varepsilon =
\begin{cases}
0, & U \le p_0, \\
T \cdot s, & U > p_0.
\end{cases}
$$

With probability $p_0$, $\varepsilon = 0$, simulating a stationary book with
no artificial drift; with probability $1 - p_0$, $\varepsilon$ injects a heavy-tailed
shock that penalizes inventory adverse selection.

---

### 3.3 Hurdle Signed-LogNormal Price Model (`hurdle_lognormal`)

Designed for high-volatility pairs (e.g. BTC_IRT) whose price movements span
several orders of magnitude (from 100 IRT to 5,000,000 IRT).

#### Discrete Generative Law

For tick size $h$ and $k \in \mathbb{Z} \setminus \{0\}$:

$$
\mathbb{P}(\Delta\mathrm{mid} = 0) = p_0,
$$

$$
\mathbb{P}(\Delta\mathrm{mid} = k \cdot h)
  = \frac{1}{2}\,(1 - p_0)\cdot
    \frac{
      \Phi\!\left(\frac{\log(|k| h + h/2) - \mu}{\sigma}\right)
      - \Phi\!\left(\frac{\log(\max(10^{-12},\, |k| h - h/2)) - \mu}{\sigma}\right)
    }{
      1 - \Phi\!\left(\frac{\log(h/2) - \mu}{\sigma}\right)
    }.
$$

The factor $1/2$ assigns symmetric probability to positive and negative jumps.
Online moments $(\mu, \sigma)$ are updated via EWMA on $\log|\Delta\mathrm{mid}|$
for non-zero steps, with $\sigma \ge \sigma_{\mathrm{floor}} = 0.35$.

---

## 4. Conditional Mechanical-Mixture Price Model (`conditional_mechanical_mixture`)

The mechanical mixture model predicts a **four-component book-shape transition**,
then converts that shape to **dynamic-depth buy/sell VWAPs** for marking.

### 4.1 Book features

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

A transition from book $\mathcal{B}$ to book $\mathcal{B}'$ is $\Delta f = f(\mathcal{B}') - f(\mathcal{B})$.
Components are indexed $c \in \{\mathrm{mid},\; \log S,\; \log\psi^{+},\; \log\psi^{-}\}$.

### 4.2 Mechanical (market-order-flow-only) transition

Given observed or sampled flow $(V^{+}, V^{-})$ and starting book $\mathcal{B}$,
the mechanical next book $\mathcal{B}^{\mathrm{mech}}$ is obtained by consuming
volume against standing depth without limit-order replenishment.
The mechanical feature move is $\Delta f^{\mathrm{mech}} = f(\mathcal{B}^{\mathrm{mech}}) - f(\mathcal{B})$.

### 4.3 Two-regime mixture

Let $V_{\Sigma} = V^{+} + V^{-}$ and $x_{\mathrm{gate}} = (1,\; \min(\log(1+V_{\Sigma}),\, 12))^\top$.
The **market-event probability** (flow gate) is

$$
q
  = \mathrm{clip}\bigl(
      \sigma(\beta_{\mathrm{gate}}^\top x_{\mathrm{gate}}),\;
      0.001,\; 1 - w_{\mathrm{refresh}}^{\min}
    \bigr),
$$

with default $w_{\mathrm{refresh}}^{\min} = 0.15$ ($q \le 0.85$). Enforcing $w_{\mathrm{refresh}}^{\min} > 0$
prevents **gate saturation** during heavy flow bursts.

For each component $c$, the predictive law is:

$$
\Delta f_c
  \;\sim\;
  q\cdot \mathcal{N}\!\bigl(\mu^{\mathrm{mkt}}_c,\; (\sigma^{\mathrm{mkt}}_c)^2\bigr)
  + (1-q)\cdot t_{\nu}\!\bigl(\mu^{\mathrm{ref}}_c,\; (\sigma^{\mathrm{ref}}_c)^2\bigr).
$$

- **Market regime** (Gaussian): $\mu^{\mathrm{mkt}}_c = \beta_{c,0} + \beta_{c,1}\,\Delta f^{\mathrm{mech}}_c$.
- **Refresh regime** (Student-$t$ with scale $\sigma^{\mathrm{ref}}_c \sqrt{(\nu-2)/\nu}$): $\mu^{\mathrm{ref}}_c = m_c$.

### 4.4 Reconstructing dynamic VWAP

Reconstructed state $\hat m, \hat S, \hat\psi^{+}, \hat\psi^{-}$ is scaled to
inventory depth $d^{+} = q_{\mathrm{cash}}/\hat m, d^{-} = |b_{\mathrm{inv}}|$ via shape ratio $r(d; Q_{\mathrm{ref}}, \alpha)$:

$$
\begin{aligned}
\mathrm{VWAP}^{\mathrm{buy}}
  &= \hat m + \hat S/2 + \hat\psi^{+}\, r(d^{+}), \\
\mathrm{VWAP}^{\mathrm{sell}}
  &= \hat m - \hat S/2 - \hat\psi^{-}\, r(d^{-}).
\end{aligned}
$$

---

## 5. Composition in One Rollout Step

For path $i$, horizon step $h$, with book $\mathcal{B}_{i,h}$ and wallet $(b_{i,h}, q_{i,h})$:

### Step 1: Flow and Soft Fill
$$
\begin{aligned}
(V^{+}_{i,h}, V^{-}_{i,h})
  &\sim \text{Hurdle-LogNormal}(\mathbf{p}_0, \boldsymbol{\mu}, \boldsymbol{\sigma}), \\
(\mathcal{B}_{i,h+1}^{\mathrm{mech}},\; \Delta\mathrm{cash},\; \Delta\mathrm{inv})
  &= \mathrm{soft\_fill}(u^{(h)}, \mathcal{B}_{i,h}, V^{+}_{i,h}, V^{-}_{i,h}).
\end{aligned}
$$

### Step 2: Price Innovation Injection

#### Branch A: Discrete Hurdle Residuals (`hurdle_student_t` or `hurdle_lognormal`)
$$
\begin{aligned}
\varepsilon^{\mathrm{buy}}_{i,h},\, \varepsilon^{\mathrm{sell}}_{i,h}
  &\sim \text{HurdleResidual}(p_0, s, \nu), \\
\mathrm{VWAP}^{\mathrm{buy}}_{i,h+1}
  &= \mathrm{VWAP}_{\mathrm{mof}}^{\mathrm{buy}} + \varepsilon^{\mathrm{buy}}_{i,h}, \\
\mathrm{VWAP}^{\mathrm{sell}}_{i,h+1}
  &= \mathrm{VWAP}_{\mathrm{mof}}^{\mathrm{sell}} + \varepsilon^{\mathrm{sell}}_{i,h}.
\end{aligned}
$$

#### Branch B: Conditional Mechanical Mixture (`conditional_mechanical_mixture`)
$$
\begin{aligned}
\Delta f^{\mathrm{mech}}_{i,h} &= f(\mathcal{B}_{i,h+1}^{\mathrm{mech}}) - f(\mathcal{B}_{i,h}), \\
q_{i,h} &= \sigma(\beta_{\mathrm{gate}}^\top x_{\mathrm{gate}}), \\
\Delta\hat f_{i,h} &\sim q_{i,h}\,\mathcal{N}(\mu^{\mathrm{mkt}}, \sigma^{\mathrm{mkt}2}) + (1-q_{i,h})\,t_{\nu}(\mu^{\mathrm{ref}}, \sigma^{\mathrm{ref}2}), \\
(\mathrm{VWAP}^{+}_{i,h+1}, \mathrm{VWAP}^{-}_{i,h+1}) &= \Gamma(\Delta\hat f_{i,h}, d^{+}, d^{-}).
\end{aligned}
$$

---

## 6. Example Configurations

### 6.1 Recommended Production Config (USDT_IRT)

```json
{
  "local_market_flow_model": {
    "kind": "hurdle_lognormal",
    "decay": 0.95,
    "init_p0": 0.3,
    "init_mu": 0.0,
    "init_sigma": 1.0,
    "sigma_floor": 0.35
  },
  "local_price_model": {
    "kind": "hurdle_student_t",
    "decay": 0.99,
    "student_df": 2.5,
    "tick_size": 0.5,
    "initial_sigma": 1.0,
    "calibrate": true
  }
}
```

### 6.2 High-Volatility Production Config (BTC_IRT)

```json
{
  "local_price_model": {
    "kind": "hurdle_lognormal",
    "decay": 0.99,
    "tick_size": 100.0,
    "init_p0": 0.5,
    "init_mu": 4.0,
    "init_sigma": 1.0,
    "sigma_min": 0.35
  }
}
```

### 6.3 Mechanical Mixture Config (Legacy / Experimental)

```json
{
  "local_price_model": {
    "kind": "conditional_mechanical_mixture",
    "reference_vwap_vol": 1000,
    "shape_alpha": 0.5,
    "decay": 0.99,
    "initial_sigma": 50,
    "initial_log_sigma": 0.25,
    "student_df": 3.0,
    "min_refresh_weight": 0.15,
    "point_mode": "sample"
  }
}
```

---

## 7. Symbol Map

| Symbol | Code / config | Meaning |
| --- | --- | --- |
| $V^{+}, V^{-}$ | `buy_vol`, `sell_vol` | Aggressive buy/sell flow per step |
| $p_{0,s}, \mu_s, \sigma_s$ | `p0_buy/sell`, `mu_buy/sell`, `sigma_buy/sell` | Hurdle flow parameters |
| $\sigma_{\mathrm{floor}}$ | `sigma_floor` | Prior floor on log-flow variance ($0.35$) |
| $h$ | `tick_size` | Discrete lattice quantum ($0.5$ USDT, $100$ BTC) |
| $p_0$ | `p0`, `p0_mid` | Probability of zero mid-price move |
| $s$ | `scale`, `sigma_mid` | Student-$t$ scale parameter on discrete lattice |
| $\nu$ | `student_df` | Student-$t$ degrees of freedom ($2.5$ USDT, $3.0$ mix) |
| $\alpha_{\mathrm{cal}}$ | `alpha_cal` | Online recursive tail calibration factor |
| $\Delta f^{\mathrm{mech}}$ | mechanical component deltas | MOF-only feature move |
| $q$ | `market_prob` | Mixture market-regime gate probability |
| $w_{\mathrm{refresh}}^{\min}$ | `min_refresh_weight` | Minimum weight floor for refresh tail regime ($0.15$) |
| $Q_{\mathrm{ref}}$ | `reference_vwap_vol` | Fixed reference volume for mixture shape ($1000$) |
| $\alpha$ (shape) | `shape_alpha` | Power-law depth elasticity exponent ($0.5$) |
