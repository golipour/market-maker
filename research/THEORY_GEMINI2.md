A First-Principles Mathematical Framework for Dynamic Market Modeling

To find the optimal predictive model for an online market maker, we must first mathematically define the environment, the objective, and the fundamental limits of predictability. We abandon static fitting and view the market as a continuous, non-stationary stochastic process observed at discrete intervals.

I. The Fundamental Problem Representation

Let the true state of the market (the full limit order book, latent order intentions, and macroeconomic context) be a continuous-time stochastic process $\mathcal{X}(t)$.

We do not observe $\mathcal{X}(t)$. We observe a discrete, low-dimensional projection at interval intervals $\Delta t$. Let this discrete observation sequence be $Y_t = (\Lambda_t, P_t)$, where $\Lambda_t$ is the market flow (volume) and $P_t$ represents price/shape metrics at time step $t$.

We define our information set (filtration) at time $t$ as $\mathcal{F}_t = \sigma(Y_0, Y_1, \dots, Y_t)$.

The theoretical objective of the predictive model is to output a sequence of conditional probability density functions $q_{\theta_t}$ that perfectly match the true, unknown conditional transition density $p$:

$$\text{Target:} \quad q(Y_{t+1} \mid \mathcal{F}_t; \theta_t) \approx p(Y_{t+1} \mid \mathcal{F}_t)$$

Why Forward NLL is the Theoretically Optimal Metric

You asked why we use forward NLL and if we should question it. From an information-theoretic standpoint, minimizing forward NLL is exactly equivalent to minimizing the Kullback-Leibler (KL) Divergence between the true data-generating process and our model, evaluated sequentially.

$$D_{KL}(p \parallel q) = \mathbb{E}_{p}\left[\log p(Y_{t+1} \mid \mathcal{F}_t)\right] - \mathbb{E}_{p}\left[\log q(Y_{t+1} \mid \mathcal{F}_t; \theta_t)\right]$$

The first term is the negative conditional entropy of the market, $-H(Y_{t+1} \mid \mathcal{F}_t)$. This is the irreducible noise (the fundamental limit of predictability). The second term is the cross-entropy, which is exactly the expected Negative Log-Likelihood (NLL).

Therefore:


$$\text{NLL}_t = H(Y_{t+1} \mid \mathcal{F}_t) + D_{KL}(p \parallel q)$$

Conclusion: NLL is the correct theoretical metric. We cannot beat the entropy $H$, but we can drive $D_{KL}$ to zero. If the data is fundamentally "unpredictable" (efficient market), $H$ is extremely high, and the best model is simply the marginal distribution $p(Y_{t+1})$.

II. Addressing Predictability and Non-Stationarity

1. The Limits of Prediction (Information Theory)

Is the market predictable at all? Predictability is mathematically defined by Mutual Information $I(Y_{t+1}; \mathcal{F}_t)$.
If $I(Y_{t+1}; \mathcal{F}_t) = 0$, the market is a pure Markov martingale, and the past tells us nothing about the future shape. However, in microstructure, while price returns often have near-zero mutual information with the past, volatility and order flow exhibit high mutual information (Hawkes-style self-excitation).

2. Modeling Non-Stationarity (The Online Filtering Problem)

Standard machine learning assumes data is independent and identically distributed (i.i.d.). The market is fundamentally non-stationary. The parameters governing the true distribution $p$ are themselves a stochastic process.

To model this, we must cast the problem as a State-Space Model. Let $\theta_t$ be the latent parameters of the market regime:

Parameter Evolution (Drift): $\theta_{t} = f(\theta_{t-1}) + \eta_t$ (where $\eta_t$ is regime-shift noise).

Observation Emission: $Y_t \sim q(Y_t \mid \theta_t)$

Your current system uses Exponential Weighted Moving Averages (EWMA) to update parameters (e.g., $\mu \leftarrow \gamma \mu + (1-\gamma)x$). EWMA is mathematically equivalent to a steady-state Kalman Filter applied to a random walk parameter model.
To do better, our roadmap must move beyond fixed-decay EWMA and allow the model to learn the rate of adaptation dynamically (e.g., using Online Natural Gradient Descent, where updates are scaled by the inverse Fisher Information Matrix).

III. Factoring the Joint Density (Dependence vs. Independence)

We must predict both flow $\Lambda$ and price $P$. The true predictive density is joint: $p(P_{t+1}, \Lambda_{t+1} \mid \mathcal{F}_t)$.
Currently, your baseline models assume conditional independence or a simple causal chain (Flow causes Mechanical Price move).

To model true dependence, we have two mathematical paths:

Factorization: $p(P_{t+1}, \Lambda_{t+1} \mid \mathcal{F}_t) = p(P_{t+1} \mid \Lambda_{t+1}, \mathcal{F}_t) \cdot p(\Lambda_{t+1} \mid \mathcal{F}_t)$. This justifies predicting flow first, then conditioning the price model on that flow.

Copula Modeling: Model the marginal distributions $u = F(P_{t+1})$ and $v = G(\Lambda_{t+1})$ separately, and link them using a Copula $C(u, v)$ to capture tail dependence (e.g., extreme flow correlating with extreme spread widening).

IV. The Roadmap to the Optimal Solution

Based on these first principles, here is the rigorous, step-by-step roadmap to finding the best solution space, strictly avoiding blind hyperparameter tuning.

Phase 1: Establishing the Entropy Floor (Epistemology)

Before fitting any complex models, we must measure the inherent predictability of the nobitex_USDT_IRT_30000ms_4d.csv data.

Action: Calculate the empirical Shannon Entropy of discrete bucketed returns and flow.

Action: Calculate the Mutual Information between interval $t$ and $t-k$.

Goal: Establish the theoretical lower bound for NLL. If a model's NLL approaches this empirical entropy, no further modeling will improve it.

Phase 2: Formulating Latent Regimes (Ontology)

Market data exhibits volatility clustering and distinct regimes (e.g., "informed toxic flow" vs. "noise trading").

Action: Formulate a Hidden Markov Model (HMM) or a Continuous-Time Markov Chain where the latent state $Z_t \in \{1, \dots, K\}$ dictates the emission distributions of flow and price.

Goal: Replace static continuous distributions with dynamic mixture models governed by a transition matrix, reducing the cross-entropy $D_{KL}$ by recognizing regime shifts instantly.

Phase 3: Optimizing the Online Update Rule (Learning Dynamics)

The fixed decay $\gamma$ is a massive theoretical limitation. When a regime shift occurs, $\gamma$ should temporarily drop to "forget" the old regime quickly, then rise to smooth the new regime.

Action: Implement Bayesian Online Change Point Detection (BOCPD) or Adaptive Filtering.

Goal: The model parameters $\theta_t$ update as a posterior distribution $p(\theta_t \mid Y_t, \theta_{t-1})$, allowing the system to quantify its own uncertainty about its parameters.

Phase 4: Probabilistic Deep Architectures (Methodology)

Only after defining the above do we reach for neural architectures. We do not want point-prediction ML. We want architectures that output probability distributions.

Action: Formulate an Online Mixture Density Network (MDN) or an Autoregressive Normalizing Flow. The inputs are the filtration $\mathcal{F}_t$, and the outputs are the parameters $\theta_t$ of a complex distribution (like a conditional mixture).

Action: Update the network weights online using streaming stochastic gradient descent (SGD), entirely prequential, respecting the strict $t-1 \to t$ walk-forward rule.

Summary of the Mathematical Path Forward

Instead of "trying models," we will sequentially answer:

What is the $H(Y_{t+1} \mid \mathcal{F}_t)$ of this asset?

What is the optimal Bayesian update rule for $\theta_t$ under non-stationarity?

What functional form (Copula, Mixture, AR) best minimizes $D_{KL}$ for the joint distribution of Flow and Price?