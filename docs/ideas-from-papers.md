Ranked by value per unit of work. Each idea comes with a test, so you can drop it quickly if it doesn't pay.

**Tier 1: cheap, mechanical, testable this week**

1. **PID on λ.** Add K_P on the instantaneous shortfall S_t − ε (not the EWMA), plus a K_D term on max(0, S_t − S_{t-1}). Keep your integral term. K_P = K_D = 0 recovers what you have now. Test: the cumulative violation Σ(S_t − ε)₊ and the λ-vs-S phase lag. If the λ roof is mainly there to stop oscillation, you may be able to loosen it.

2. **Gradient-norm balancing for λ.** Scale the viability gradient by ‖∇CE‖/‖∇S₁‖ so that λ = 1 means the same thing across assets and regimes. You run assets at very different Q_ref and depth scales, so this should cut per-asset retuning of λ_max and the learning rate.

3. **ESS monitor on the CE.** The CE gradient is a softmax-weighted average of path gradients, with weights ∝ e^{−γW}. Log ESS = 1/Σwᵢ². With heavy-tailed returns, a few paths can dominate, which would also produce the gradient collapse you saw at warm-start. This is a diagnostic I'm proposing, not something a paper establishes. If ESS is low, use more paths, antithetic sampling or a control variate, or lower β.

4. **Markout diagnostic before any modelling.** Compute signed mid-price markouts after your own fills (1s, 10s, 60s, per side, per regime, conditional on your inventory sign). If the average is negative beyond the spread you earned, you have adverse selection. If it isn't, skip tier 2.

**Tier 2: structural gaps, worth doing if the markouts say so**

5. **Couple fills to subsequent price in the rollout.** Your architecture separates price discovery from fill mechanics, which means fills are effectively independent of future price moves. That makes the CE overvalue fills, because adverse selection is exactly the dependence between "I got filled" and "price moved against me". The Feys paper explicitly scopes this out. The CE aggregates whatever path distribution you give it, so the fix has to live in the price/flow models. Concretely, shift the price drift conditional on a fill, with the shift estimated from the markouts in item 4.

6. **Toxicity-aware flow updates.** Your hurdle-lognormal EWMA treats every fill the same. Down-weight or reclassify the update when recent imbalance and your own skew direction align. This is the crude version of what Cartea et al.'s detection work does with online learning. A small online classifier on book state and recent flow could replace the heuristic later. Transfer caveat: those papers use proprietary FX client data. You can't tier anonymous book participants the way Barzykin et al. tier clients, so only the detection idea carries over, not the tiering.

7. **Trade-time clock for the price model.** Feys cites evidence that variance is roughly linear in trade count. Re-fit your price model per trade (or per volume bucket) instead of per wall-clock step. Your ARCH effects, and possibly some of that extreme ν ≈ 0.38 Student-t (stretches of zero-return ticks can inflate it), may weaken a lot. Test: ARCH-LM statistic and fitted ν in both clocks. This is a hypothesis from the subordination literature, and I haven't verified it on Nobitex or Wallex data.

**Tier 3: consistency checks on the objective**

8. **No double counting.** CE on marked terminal wealth already prices inventory variance. If any explicit inventory-risk term remains in J besides a terminal liquidation cost, it should equal γσ²/2 or be removed. If your rollouts have a terminal penalty, derive it from the walk-the-book cost to liquidate q and set α = ½L″(0) from the order book, not by tuning.

9. **Size scaling.** The entropic measure isn't positively homogeneous (Corollary 26): scaling the book by k is equivalent to scaling γ by k. So normalise W by Q_ref and keep β dimensionless, or your cross-asset β values aren't comparable.

10. **λ-vs-σ_t lag plot.** The clock-invariance corollary predicts any risk signal that isn't indexed to accumulated variance will lag regime changes.

**Evaluation harness (probably the most underrated idea)**

The FX repo reports before/after numbers on real ticks, P&L per unit variance, and a mid-day toxicity-shock stress test. Build the same for your system: replay on recorded data, inject a toxicity shock, and compare policies on loss during the shock and P&L per variance. Without this, none of the above can be ranked on evidence. The repo also cites a gym for model-based trading (mbt_gym); I only saw the citation, so check whether it's usable before relying on it.

**Two caveats on what I told you earlier**

- I said the CVaR result "proves" your separate viability constraint is the right call. That overstates it. Feys is a single-author preprint, and the result is conditional on his axioms. It supports keeping viability out of the objective, but doesn't prove it optimal.
- Most of this literature is FX-dealer or RFQ, bank-scale, and simulated or proprietary data. Transfer to anonymous lit books on thin venues is plausible but unproven.

I'd order the work 4 → 1 → 3 → 5, then the harness. The markouts decide whether 5 and 6 are worth building. If you want, I can draft the markout and ESS diagnostics against your existing code.



AS (2008) - GLFT (2013)
Use the AS/GLFT value function as an approximate terminal cost in the MPC. Standard MPC practice is to add a cost-to-go approximation at the end of the horizon, and the closed form is a natural candidate. That would shorten the horizon you need and reduce the H-dependence of your skew. It is only as good as the model’s validity near your actual dynamics, so test it against the frontier from step 2. Feys’ result also says any terminal penalty should come from the book’s liquidation cost, not be tuned, so derive it that way.