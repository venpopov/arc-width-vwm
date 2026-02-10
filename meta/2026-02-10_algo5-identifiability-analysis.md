# Algo5 identifiability analysis and reparametrization proposals

**Date:** 2026-02-10

## Problem statement

When fitting the noisy-gradient model (algo5) to real data via Bayesian inference, MCMC chains get stuck in one of two modes:

| Mode | τ    | δ₀   | σ_g  |
|------|------|------|------|
| A    | high | high | low  |
| B    | low  | low  | high |

Parameter recovery on simulated data did not detect this issue.

## Diagnosis: three-way multiplicative ridge

### Decomposing the arc-transition log-likelihood

Drop constants and write the arc component of the log-likelihood explicitly. Let $r_t = x_{t+1} - x_t$ be the **observed** logit-arc change and $u_t = r_t\,(t+1)^{1/2}$ (a known quantity, free of parameters). Then

$$
\ell_{\text{arc}}
= -(T{-}1)\log\delta_0
  -(T{-}1)\log\sigma_g
  -\frac{1}{2\sigma_g^{2}}
    \sum_{t}\!\left(\frac{u_t}{\delta_0} - g_t(\tau)\right)^{2}
  + \text{const}.
$$

This is a **normal regression** of $u_t/\delta_0$ on $g_t(\tau)$ with residual variance $\sigma_g^2$, plus a penalty from the normalizing constant. The pathology lives entirely in the $(τ, δ_0, σ_g)$ interaction inside this expression.

### How τ and δ₀ couple (the gradient-scale ridge)

The gradient is

$$
g_t = \underbrace{-s_t}_{O(1)}
     + \underbrace{(\pi-a_t)\,\frac{s_t(1-s_t)}{\tau}}_{O(1/\tau)}.
$$

The second term is the "spike" at the decision boundary $|y_t|\approx a_t$, and its magnitude is $\propto 1/\tau$ (the bump $s_t(1-s_t)$ has width $\sim\tau$ and height $1/(4\tau)$). So for the mean of the transition

$$
\delta_t\, g_t \;\approx\; \frac{\delta_0}{(t+1)^{1/2}}\;\frac{C}{\tau}
\;=\;\frac{\delta_0/\tau}{(t+1)^{1/2}}\;C,
$$

the product $\delta_0/\tau$ is the quantity the data can pin down; the individual factors are only weakly separated by the different shapes of $g_t$ across trials.

### How δ₀ and σ_g couple (the variance ridge)

The transition SD is $\delta_t\,\sigma_g$, so $\operatorname{Var} = \delta_0^2\,\sigma_g^2/(t+1)$. Only the *product* $\delta_0\,\sigma_g$ (= ω, the "effective innovation amplitude") is directly constrained by the observed residual scatter.

### The combined ridge

Putting both together, there is an approximate invariance surface

$$
(\tau,\delta_0,\sigma_g)
\;\to\;
(\lambda\tau,\;\lambda\delta_0,\;\sigma_g/\lambda),
$$

which preserves both the mean ($\delta_0/\tau = \text{const}$) and the variance ($\delta_0\sigma_g = \text{const}$). It is not an *exact* symmetry — the $O(1)$ piece $-s_t$ and the τ-dependent shape of $s_t$ do break it — but the approximate degeneracy creates a banana-shaped ridge that MCMC explores as two separated modes rather than a smooth continuum.

### Why parameter recovery didn't catch it

In recovery, the simulated data *is* generated from a single point on this ridge, and with 1000 trials the shape information in $g_t(\tau)$ is enough to pin down that point. In real data, model mis-specification or a flatter posterior landscape (fewer effective observations per parameter) turns the near-ridge into a genuine barrier.

### Contrast with algo4

Algo4's transition is $\text{logit}(p_{\text{det}}^{(t)}) + \sigma_e\,\varepsilon_t$ where the deterministic step depends on δ alone and $\sigma_e$ enters *only* in the variance. There is no τ and no multiplicative coupling — hence no ridge.

---

## Proposal 1 — Constant innovation noise ("execution noise")

The simplest fix is to remove the multiplicative coupling by making the noise scale independent of the learning rate:

$$
\boxed{x_{t+1} = x_t + \delta_t\,g_t(\tau) + \sigma_w\,\varepsilon_t,
\qquad \varepsilon_t\sim\mathcal{N}(0,1).}
$$

**Identifiability analysis.** Now $\sigma_w^2$ is pinned by the residual variance of $x_{t+1}-x_t-\delta_t g_t(\tau)$ — it enters *only* the variance. The τ–δ₀ coupling in the mean still exists but is far more benign: once σ_w is identified, the mean structure $\delta_0/\tau$ is constrained by a proper regression rather than by a three-way interaction.

**Interpretation.** The noise represents trial-to-trial *execution* variability in setting the arc (motor noise, attentional fluctuation), which is plausibly constant across trials rather than proportional to the learning rate.

**Downside.** Late-trial arcs will be as noisy as early-trial arcs. If the data show decreasing arc variability over time, this could be captured instead by letting the *systematic* step $\delta_t g_t$ shrink (which it does), while the residual noise σ_w stays constant — essentially the model predicts that late-trial variability converges to a floor of σ_w.

---

## Proposal 2 — Langevin noise scaling

If you want noise that decays but less aggressively, use the standard Langevin-dynamics scaling:

$$
\boxed{x_{t+1} = x_t + \delta_t\,g_t(\tau) + \sqrt{\delta_t}\;\sigma_L\,\varepsilon_t.}
$$

Now δ₀ enters the mean *linearly* and the variance as $\delta_0^{1/2}$ — a much weaker coupling. The approximate invariance $(\tau,\delta_0,\sigma_g)\to(\lambda\tau,\lambda\delta_0,\sigma_g/\lambda)$ is broken because scaling δ₀ by λ would require scaling σ_L by $\lambda^{-1/2}$, not $\lambda^{-1}$. The ridge curvature increases substantially.

**Interpretation.** This is the standard discretization of overdamped Langevin diffusion. The "temperature" $\sigma_L^2/2$ controls the stationary spread.

---

## Proposal 3 — Reparametrize the existing model

If you want to keep the original model equations *exactly*, reparametrize from (τ, δ₀, σ_g) to:

| New parameter | Definition          | Interpretation                          |
|---------------|--------------------|-----------------------------------------|
| φ = δ₀/τ     | effective learning rate | absorbs the 1/τ gradient scale        |
| ω = δ₀·σ_g   | effective noise amplitude | the observable innovation scale      |
| τ             | temperature         | controls gradient smoothness            |

Then δ₀ = φτ and σ_g = ω/(φτ). The transition becomes

$$
x_{t+1}\sim\mathcal{N}\!\left(x_t + \frac{\phi\tau}{(t{+}1)^{1/2}}\,g_t(\tau),\;\left(\frac{\omega}{(t{+}1)^{1/2}}\right)^{\!2}\right).
$$

The "ridge direction" $(\tau,\delta_0,\sigma_g)\to(\lambda\tau,\lambda\delta_0,\sigma_g/\lambda)$ now maps to (λτ, φ, ω) — it only moves τ, while φ and ω are invariant. So the sampler can explore τ freely without dragging the other two parameters along.

This doesn't eliminate the near-ridge entirely (because $g_t$ still has mild τ-dependence through the shape of $s_t$), but it concentrates the degeneracy into a single direction that is much easier for MCMC (especially NUTS) to traverse.

---

## Proposal 4 — Eliminate τ from estimation

The τ parameter's substantive role is minor — it controls the *softness* of the step-function approximation. As τ→0 the gradient converges to the true subgradient. For estimation we need τ>0 but its precise value has little scientific meaning.

**Fix τ at a data-driven value** — e.g., a small fraction of the median arc width — and estimate only (c, κ, δ₀, σ). This:

- eliminates one parameter
- removes the τ–δ₀ ridge entirely
- makes the model's geometry indistinguishable from algo4's (one rate parameter + one noise parameter, entering mean and variance separately)

You could also treat τ as a sensitivity analysis parameter, fitting the model at several fixed τ values and checking stability.

---

## Proposal 5 — A simpler stochastic arc model (the bigger picture)

The gradient computation in algo5 exists because the expected reward $R(a)=(\pi-a)P(|y|\le a)$ is not differentiable (the step function at $|y|=a$), so the smooth sigmoid with temperature τ was introduced to create a usable gradient. But this smoothing is an artifact of wanting a *gradient*-based update. A simpler stochastic optimization approach avoids it entirely.

### Robbins–Monro stochastic approximation

The first-order condition $R'(a^*)=0$ is equivalent to

$$
P(|y|\le a^*) = (\pi-a^*)\,f_{|y|}(a^*).
$$

A single-sample unbiased estimator of $-P(|y|\le a)$ is $-\mathbf{1}[|y_t|\le a_t]$, but the density term $(\pi-a)f(a)$ cannot be estimated from one sample. The algo5 gradient sidesteps this by using the smooth kernel $s_t(1-s_t)/\tau$ as a one-sample density estimate. That's clever, but it is the source of the τ parameter and all its estimation trouble.

**Alternative: use a sign-based or reward-based update.** Consider updating toward the observed error magnitude directly (analogous to algo4 but on the logit scale):

$$
x_{t+1} = x_t + \delta_t\!\left[\,\text{logit}(|y_t|/\pi) - x_t\,\right] + \sigma_w\,\varepsilon_t.
$$

This is an exponential smoother (on the logit scale) tracking $|y_t|$, plus execution noise. Parameters: $(c,\kappa,\delta_0,\sigma_w)$. No τ. The "gradient" is replaced by a simple error-correction rule. This tracks toward $E[\text{logit}(|y|/\pi)]$, which is related to but not identical to the reward-optimal arc — the relationship depends on the SDM distribution.

If you want to preserve the reward-optimization flavor more explicitly, a **bandit-style** update can be derived:

$$
x_{t+1} = x_t + \delta_t\!\bigl[p_t - \bar{p}_t\bigr]\,\text{sign}\!\left(x_t - x_{t-1}\right) + \sigma_w\varepsilon_t,
$$

where $p_t$ is the trial reward and $\bar{p}_t$ is its running average — the agent increases its arc step direction when reward exceeds baseline. This is a *reward-based* stochastic optimizer with no gradient computation at all.

These are more radical departures from the current model, but they trade mathematical elegance for practical estimability (no τ, no ridge).

---

## Recommended path forward

| Priority      | Action                                                                 | Rationale                                                                                          |
|---------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| **Now**       | Implement **Proposal 1** (constant noise σ_w) alongside current model | Minimal code change; breaks the three-way ridge; provides an anchor for comparison                |
| **Now**       | Also implement **Proposal 3** (reparametrize to φ, ω, τ)             | Tests whether the original model can be estimated with better geometry                            |
| **Short-term**| Try **Proposal 4** (fix τ) on real data                               | If results are stable across τ values, the parameter was never needed for fitting                 |
| **Longer-term**| Consider **Proposal 5** (simpler stochastic rule)                    | The gradient formulation may be over-engineered for the amount of information in participant data  |
