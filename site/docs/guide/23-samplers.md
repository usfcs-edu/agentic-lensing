# 23. HMC, SMC, and flows: three log-dets, one idea

Chapter 22 built the machine: a real, differentiable, 46-dimensional
log-posterior with the Occam term already inside it. This chapter is about
walking that surface and coming back with a number and an honest uncertainty
on it. Four technologies appear in this campaign's own numbers —
hand-built-metric HMC, a two-stage version of the same recipe, tempered SMC,
and a normalizing-flow "NeuTra" recipe — and the claim of this chapter is
that all four answer one question: *what change of coordinates turns this
specific posterior into something a local, gradient-following integrator can
explore*. Getting that question right is the difference between
$\hat R = 22$ and $\hat R = 1.003$ on this campaign's own real fits — two
different posteriors, not a before/after on one. By the end you will be
able to read an $\hat R$ or an ESS number and know whether to trust it, and
you will have *derived*, not been told, that the Occam term Chapter 8
introduced and the lensing magnification that opened the Log-Det Ledger in
Chapter 4 are the same computation this chapter closes.

!!! abstract "What you can skip"
    Metropolis–Hastings, Markov-chain convergence, and gradient-based
    optimization (autodiff, backprop) are yours already — none of that is
    re-derived here. A normalizing flow's bijection and log-det correction
    were already [Chapter 4](04-multivariable.md#change-of-variables)'s
    business; this chapter does not re-teach what a flow is. What *is* new:
    the specific, easy-to-get-backwards convention this repo's sampler
    libraries use for a "mass matrix"; Gelman–Rubin $\hat R$ and
    effective-sample-size in the exact multi-chain sense this book uses them;
    and the derivation that the Occam term, a flow's log-det, and the lensing
    magnification are, line for line, one computation.

## Why gradients change everything { #why-gradients }

A plain Metropolis proposal has no information about which direction the
target increases; it guesses, evaluates, and keeps the guess only if it got
lucky. A gradient removes the guessing: $\nabla\log p(\theta\mid D)$ points
exactly toward higher posterior density — the same quantity `jax.grad`
already computes for every one of this campaign's `@jax.jit`-compiled
renders ([Chapter 22](22-inference.md#the-forward-model)). Hamiltonian Monte
Carlo (HMC) does not use that gradient for one step; it integrates an entire
physically-motivated trajectory before proposing a single new point — the
next section's derivation.

!!! tip "You already know this"
    If you have used stochastic gradient Langevin dynamics (SGLD) or seen a
    diffusion model's score-based sampler, you have already used this exact
    primitive: a gradient of a log-density, biasing a random walk toward
    high-probability regions. HMC is what you get when you replace SGLD's one
    noisy step between gradient evaluations with many deterministic,
    momentum-conserving steps — trading a single kick for an entire coasting
    trajectory before the next random draw.

Gradients are necessary here, but this campaign's own benchmark is blunt
about them not being *sufficient*. At an identical, budget-matched number of
gradient evaluations, no gradient-based contender beat the plain multi-start
baseline on the harder real-lens targets — the single most efficient sampler
in the whole zoo used no gradient information at all: nested sampling
(`nautilus`) beat the baseline's ESS-per-gradient by $2.6$–$307\times$
<!-- check: ch23.nautilus_essgrad_lo = 2.6 ± 0.01 -->
<!-- check: ch23.nautilus_essgrad_hi = 307.0 ± 0.1 -->
across the zoo (`reproductions/claude-giga-lens/papers/main.tex:989-991`).
Only when the campaign stopped matching budgets and ran every contender
*until actually converged* did the ranking flip: parallel-tempered HMC
converged 5 of 6
<!-- check: ch23.pthmc_hard_converged = 5 ± 0 -->
<!-- check: ch23.pthmc_hard_total = 6 ± 0 -->
of the hardest real T1 systems, including one the baseline itself could not
close (`reproductions/claude-giga-lens/papers/main.tex:1019-1020`). This
book's recurring refrain, applied to sampling: a gradient is necessary but
not sufficient — it only pays off paired with the right local geometry,
which is this chapter's real subject, starting now.

## HMC and the metric { #hmc-and-the-metric }

Give the parameter vector $\theta$ a fictitious momentum $\mathbf p$ (bold,
to keep it visually apart from the posterior density $p(\theta\mid D)$) and
define a Hamiltonian $\mathcal H$ — total energy, potential plus kinetic,
calligraphic so it is never confused with the Hessian $H$ this section
reaches for two paragraphs from now:

$$
\mathcal H(\theta, \mathbf p) = \underbrace{-\log p(\theta \mid D)}_{\text{potential}}
\;+\; \underbrace{\tfrac12\, \mathbf p^\top M^{-1} \mathbf p}_{\text{kinetic}}.
\label{eq:hmc-ham}
$$

$M$, the **mass matrix**, is a free choice — any symmetric positive-definite
matrix. Hamilton's equations, $\dot\theta = \partial\mathcal H/\partial\mathbf p =
M^{-1}\mathbf p$ and $\dot{\mathbf p} = -\partial\mathcal H/\partial\theta =
\nabla\log p(\theta\mid D)$, conserve $\mathcal H$ exactly and are simulated by
**leapfrog** integration: half a momentum kick using $\nabla\log p$, a full
position drift using $M^{-1}\mathbf p$, another half kick. Every leapfrog
step costs exactly one gradient evaluation — the convention
`reproductions/claude-giga-lens/cgl/metrics.py:284-286`'s budget ledger
states outright ("`n_grad` counts logp gradient evaluations... 1 per leapfrog
step per chain"). Leapfrog is also *exactly* volume-preserving: its own
Jacobian has $|\det J|=1$ by construction — arguably a fourth, degenerate
ledger row, and the reason HMC's Metropolis correction needs no Jacobian
term at all, unlike a generic nonlinear proposal.

$M$ is not a free lunch; it decides whether leapfrog is efficient or
useless. Near the posterior's mode, $-\log p(\theta\mid D) \approx
\text{const} + \tfrac12(\theta-\hat\theta)^\top H(\theta-\hat\theta)$ (the
Laplace picture, [Chapter 8](08-probability.md#laplace)). Substitute
$u = M^{1/2}(\theta-\hat\theta)$ — a *linear* change of variables, exactly
[Chapter 4](04-multivariable.md#det-j-as-area-scaling)'s machinery — and the
local potential becomes $\tfrac12 u^\top M^{-1/2}HM^{-1/2}u$. Choose
$M = H$ and this collapses to $\tfrac12 u^\top u$: a perfectly isotropic
bowl, in $u$-coordinates, however stretched $H$ made $\theta$-space. That is
the whole argument for a mass matrix: it is a linear whitening transform,
built to cancel the target's own curvature, so that a fixed leapfrog step
size works equally well in every direction at once.

Since $\Sigma \equiv H^{-1}$ is the Laplace *covariance*, the rule is
$M = \Sigma^{-1} \approx H$ — and `reproductions/claude-giga-lens/cgl/e2.py:657-658`
says this in so many words: *"run\_staged/\_run\_phmc use momentum covariance
= Sigma^-1, the GIGA-Lens convention."* That single reciprocal is a genuine
convention trap: this campaign's two sampler libraries name the *same*
$\Sigma$ at *opposite* ends of it. TFP's `PreconditionedHamiltonianMonteCarlo`
wants its `momentum_distribution` handed $M$ directly, so
`reproductions/claude-giga-lens/cgl/samplers/remc_pt.py:125` inverts first
(`prec = np.linalg.solve(ginit.cov_reg, np.eye(dim))`). `blackjax`'s NUTS and
MCLMC adapters, by contrast, take a parameter literally called
`inverse_mass_matrix` — and `reproductions/claude-giga-lens/cgl/samplers/bj_smc.py:72`
passes $\Sigma$ to it *un-inverted*, commented `"inverse mass = covariance"`
(`reproductions/claude-giga-lens/cgl/samplers/bj_nuts.py:116` does the same).
Fill in the wrong library's parameter with
the other's convention and every direction gets the wrong momentum: the
tightly-constrained direction, which should move cautiously, gets a *light*
mass and leapfrog blows up in it; the loose direction gets a *heavy* mass
and crawls where it should roam freely.

!!! tip "You already know this"
    Building $M$ from the target's own curvature is quasi-Newton
    optimization, run for sampling instead of minimizing. Adam's per-parameter
    learning-rate scaling and this chapter's $M \approx H$ exist to fix the
    identical pathology: a surface whose curvature differs wildly by
    direction, where one fixed step size is either too timid everywhere or
    too reckless somewhere.

You judge whether a chosen $M$ worked with two multi-chain diagnostics.
$\hat R$ (Gelman & Rubin) compares within-chain variance $W$ to the pooled
between-chain variance: for $m$ chains of $n$ draws each,
$\widehat{\mathrm{var}} = \tfrac{n-1}{n}W + \tfrac{B}{n}$, and
$\hat R = \sqrt{\widehat{\mathrm{var}}/W}$. Near $1$, every chain agrees on
where the posterior's mass is; large means some chains are stuck somewhere
the others are not. (`cgl/metrics.py:23-68` runs a more careful
rank-normalized, *split* version — splitting each chain in half catches a
single chain drifting over time — but the arithmetic is the same idea.)
Effective sample size (ESS) asks the complementary question for *one*
chain: how many independent draws is a correlated sequence of $N$ actually
worth? For a stationary, autocorrelated sequence with integrated
autocorrelation time $\tau_{\mathrm{int}}$, $\mathrm{ESS}\approx N/\tau_{\mathrm{int}}$
— literally the same $N_{\mathrm{eff}}/N$ reduction
[Chapter 7](07-fourier.md#psd-and-autocorrelation) already applied to a
*stationary pixel sequence*; a Markov chain and a correlated noise field are,
arithmetically, the same object.

Ground both diagnostics in a real fit: the 46-dimensional posterior Chapter
5 measured at condition number $\sim10^{14}$
<!-- check: ch05.cond_ill_conditioned = 1e14 ± 1e10 -->
is *positive-definite* — a genuine, brutally stretched bowl — and a better
$M$ alone fixes it. Single-stage HMC, metric built from a floored SVI
covariance, reaches only
$\hat R = 3.10$
<!-- check: ch23.t2_rhat_singlestage = 3.10 ± 0.01 -->
(ESS $28$
<!-- check: ch23.t2_ess_singlestage = 28 ± 0.5 -->
at 300 kept draws); the honest "no hand-built mass matrix" contender,
auto-tuned MCLMC, does *worse* —
$\hat R = 5.9$
<!-- check: ch23.t2_rhat_automass_mclmc = 5.9 ± 0.01 -->
(ESS $9$
<!-- check: ch23.t2_ess_automass_mclmc = 9 ± 0.5 -->).
What converges this target is rebuilding $M$ from pooled draws of a first
HMC stage and running a second stage from there — the campaign's own
two-stage recipe
([§4.2, "The two-stage PHMC finding"](../current/claude-giga-lens/index.md#sec:twostage)) —
which takes the *same* posterior from $\hat R = 2.11$
<!-- check: ch23.t2_rhat_twostage_before = 2.11 ± 0.01 -->
to $1.003$
<!-- check: ch23.t2_rhat_twostage_after = 1.003 ± 0.001 -->
(`reproductions/claude-giga-lens/papers/main.tex:1140-1152`). A better
metric alone bought this section's whole argument in one recipe change. But
"rebuild a better $M$" presumes a valid local curvature exists to rebuild it
from — and the money number's own posterior, next, does not have one.

## Tempering and SMC { #tempering-and-smc }

Apply the exact recipe that just worked — rebuild $M$ from better and better
local curvature estimates — to the money number's own correlated,
binned-low posterior, and it fails outright. Two different
positive-definite metric repairs, a diagonal $|H_{ii}|$ metric and a
full-rank SVI covariance, both stall: $\hat R = 21$
<!-- check: ch23.saddle_rhat_diagraw = 21.0 ± 0.5 -->
for the first, $\hat R = 22.3$
<!-- check: ch25.rhat_saddle_metric = 22.3 ± 0.01 -->
for the second (`CAMPAIGN.md`, "P1c metric-fix attempts — 2026-07-10";
`reproductions/claude-giga-lens/papers/main.tex:909-914`) — both worse than
T2's already-broken single-stage attempt above. The reason is not a tuning
miss: this MAP is a **saddle**.
[Chapter 5](05-linear-algebra.md#definiteness-and-saddles) already showed
you its Laplace Hessian's minimum eigenvalue is $-14.85$
<!-- check: ch25.saddle_min_eigenvalue = -14.85 ± 0.01 -->
with $5$
<!-- check: ch25.n_negative_eigenvalues = 5 ± 0 -->
negative directions out of 46. There is no positive-definite local curvature
to build $M$ from at that point — both "repairs" are valid metrics for a
*different, imaginary* posterior. ([Chapter 26](26-the-saddle.md#the-map-is-a-saddle)
gives the saddle its full account; this chapter only needs the diagnosis.)

The campaign's own forensics on that $\hat R\approx 22$ are worth a moment,
because the culprit is not $\gamma$. Tracing the worst-mixing parameter finds
a Sérsic-versus-shapelet source-*centre* degeneracy split into two disjoint
clusters, at $-0.15$ and $+0.09$, with their own $\hat R$'s of $22.3$ and
$15.1$
<!-- check: ch23.source_centre_rhat_b = 15.1 ± 0.5 -->
(`reproductions/claude-giga-lens/papers/main.tex:912-914`) — decoupled from the slope entirely. No amount
of rebuilding $M$ helps here, for a structural reason: leapfrog is a *local,
deterministic* integrator. A trajectory started in one source-centre cluster
follows $\nabla\log p$ smoothly and has no mechanism for teleporting across
a valley of near-zero density to the other cluster, however well-scaled its
momentum is.

**Tempering** sidesteps the whole question of a single local metric. Define
a family of intermediate distributions bridging an easy reference $r$ (a
prior, or a Gaussian fit to one basin) to the true target,
$\pi_\lambda(\theta) \propto r(\theta)^{1-\lambda}\,p(\theta\mid D)^\lambda$,
$\lambda: 0\to1$.

!!! tip "You already know this"
    This is simulated annealing's temperature schedule, run in reverse: instead
    of cooling a hard optimization problem into an easy one, tempering *heats*
    an easy, unimodal reference up into the hard, possibly-multimodal target.

**Sequential Monte Carlo (SMC)** is the machinery that walks that family with
a *population* of particles rather than one chain: at each $\lambda$-step,
reweight every particle by the incremental likelihood ratio, resample
(discard low-weight particles, replicate high-weight ones), then mutate each
survivor with a few steps of an inner MCMC kernel (here, HMC) to diversify
without changing the marginal — exactly a particle filter, the machinery
behind robot localization, with $\lambda$ playing the role "time" plays
there. Because particles are resampled from the *whole population* every
step, one stuck in the wrong source-centre cluster is simply discarded in
favor of a copy from the other — the teleportation a single leapfrog
trajectory cannot do. And because the population's importance weights,
integrated across the whole anneal, give an unbiased estimate of the
normalizing constant, $\log Z$ falls out for free — the exact evidence
[Chapter 8](08-probability.md#evidence-and-nats) introduced. Both halves of
this book's headline swing were computed by *this same tool*: the diagonal
comparison used $300$
<!-- check: ch23.smc_n_particles_diagonal = 300 ± 1 -->
particles (`reproductions/claude-giga-lens/papers/main.tex`, Table `tab:basinflip`) and returned
$+162.2$ nats
<!-- check: ch25.dlogz_diagonal = 162.2 ± 0.1 -->
for the steep basin; the correlated comparison used $128$
<!-- check: ch25.n_smc_particles = 128 ± 0.5 -->
particles annealed over $28$
<!-- check: ch25.n_smc_lambda_steps = 28 ± 0.5 -->
steps, ESS between $77$
<!-- check: ch25.ess_smc_low = 77 ± 0.5 -->
and $118$
<!-- check: ch25.ess_smc_high = 118 ± 0.5 -->
<!-- check: ch25.dlogz_correlated = -28.9 ± 0.1 -->,
and returned $-28.9$ nats for the low basin — a $191.1$-nat swing
<!-- check: ch25.evidence_swing_nats = 191.1 ± 0.05 -->
[Chapter 25](25-money-number.md#the-evidence-flip) builds its verdict on.
One tool, run twice, produced both the basin weighting *and* a properly
marginalized $\gamma$ posterior a frozen HMC chain could never have reached
— and its outer loop needed no hand-built metric at all, only a modest one
for the inner mutation kernel, which only has to jiggle particles between
resamples, not carry the whole exploration alone.

## Flows and NeuTra { #flows-and-neutra }

A third strategy: instead of hand-building $M$ or replacing the sampler
outright, teach a neural network the whitening transform. A normalizing flow
$T$, fit to samples from *some* approximation of the posterior, bijects a
standard-normal $u$-space to $\theta$-space; run ordinary, identity-mass HMC
entirely in $u$-space, then push draws back through $T$.
`reproductions/claude-giga-lens/cgl/flows.py:17-21` states the pullback
exactly as [Chapter 4](04-multivariable.md#change-of-variables) already
derived it: $\log p_u(u) = \log p_{\mathrm{target}}(T(u)) +
\log\lvert\det J_T(u)\rvert$. If $T$ has genuinely learned the target's
correlations, $u$-space *is* the isotropic space the previous section built
$M$ for by hand — except a network built it, and its log-det correction is
doing the mass matrix's job.

The catch is the one that phrase hides: $T$ is only as good as what it was
trained on, and this campaign's NeuTra recipe fits its flow to samples drawn
from a floored *SVI Gaussian*
(`reproductions/claude-giga-lens/cgl/samplers/neutra.py:82-88`) — the same
local approximation that already failed to describe a saddle, above. A flow
trained on the wrong reference faithfully learns to whiten the wrong thing.
On the zoo's own cond-$10^{14}$ mock target, NeuTra's $\hat R$ reaches
$5.72$
<!-- check: ch23.neutra_rhat_illcond46 = 5.72 ± 0.01 -->
while parallel-tempered HMC's own tuned temperature ladder *blows up*
entirely, and
only the humble baseline and MCLMC stay healthy
(`reproductions/claude-giga-lens/papers/main.tex:1016-1025`). Worse,
NeuTra's *median* $\hat R$ across the whole benchmark looks almost fine,
$1.160$
<!-- check: ch23.neutra_eval_rhat_median = 1.16 ± 0.01 -->
(Table `tab:deveval`) — an aggregate that quietly buries one badly broken
system inside many easy ones, the same lesson tempered SMC taught from a
different angle: a sophisticated preconditioner is only as trustworthy as
what it was built or trained on.

Hand-built-metric HMC, SMC, and flow-space NeuTra are three answers to one
question — what change of coordinates makes *this specific* posterior
tractable for a local integrator — and this campaign needed pieces of all
three, because no single answer generalizes across a saddle, two
disconnected sub-modes, and a condition number of $10^{14}$ in the same 46
numbers.

## Closing the Log-Det Ledger { #closing-the-log-det-ledger }

Three rows, three costumes: lensing magnification
$\mu = 1/\lvert\det A\rvert$ ([Chapter 4](04-multivariable.md#the-log-det-ledger)),
a flow's pullback correction $\log\lvert\det J_T\rvert$ (same anchor), and
the Occam factor $-\tfrac12\log\det H$ this chapter's own mass matrix reused
([Chapter 8](08-probability.md#laplace)). Here is why they are one
computation, derived rather than asserted.

The Laplace evidence integral is a $d$-dimensional Gaussian integral over
the un-normalized posterior near its mode:

$$
\int_{\mathbb R^d} \exp\!\Big(\!-\tfrac12(\theta-\hat\theta)^\top
H(\theta-\hat\theta)\Big)\,d\theta.
\label{eq:gaussint}
$$

Substitute $u = H^{1/2}(\theta-\hat\theta)$ — the *identical* linear map this
chapter already used to build a mass matrix. By
[Chapter 4](04-multivariable.md#det-j-as-area-scaling)'s own
reciprocal-determinant law, $d\theta = du/\lvert\det H^{1/2}\rvert =
du/\sqrt{\det H}$, so $\eqref{eq:gaussint}$ becomes
$(2\pi)^{d/2}/\sqrt{\det H}$, and its logarithm is exactly
$\tfrac{d}{2}\log(2\pi) - \tfrac12\log\det H$ — the Occam factor, term for
term, out of the same area-scaling fact that turned a unit square into a
parallelogram back in Chapter 4. A flow's own pullback correction is the
*identical* substitution with the linear map $H^{1/2}$ replaced by a general
nonlinear $T$, applied *pointwise* to one draw instead of integrated over
the whole space — the only difference between rows 2 and 3.
`site/guide_src/lensing.py:139-146`, this
book's opening quote on the matter, is not a metaphor: *"This is the SAME
change-of-variables factor that a normalizing flow applies as $-\log|\det
J|$, and a cousin of the $-\tfrac12\log\det A$ Occam term in
`cgl/marg.py`. Three log-dets, one idea."*

Not a demonstration on toy numbers alone: row 1's own worked value is
$\det A = 0.485$
<!-- check: ch04.det_A = 0.485 ± 1e-9 -->,
$\mu = 2.0619$
<!-- check: ch04.mu = 2.0619 ± 1e-4 -->
([Chapter 4](04-multivariable.md#det-j-as-area-scaling)); row 3 — where, as
[Chapter 4](04-multivariable.md#the-log-det-ledger) already flagged, $A$
means `marg.py`'s ridge normal matrix, not the lensing Jacobian — measures,
at the campaign's own audited production point, $\log\det A = 323.229$
<!-- check: ch08.occam_logdetA_parity = 323.229 ± 0.001 -->,
a $161.6145$-nat
<!-- check: ch22.occam_correction_production_nats = 161.6145 ± 0.001 -->
correction — not a rounding footnote, a number on the same order as the
evidence swing this book's spine turns on. Three costumes, one determinant,
audited from a $2\times2$ toy to a 46-dimensional real fit.

!!! note "Log-Det Ledger — closed"
    | # | Costume | Formula | Where |
    |---|---|---|---|
    | 1 | Lensing magnification | $\mu = 1/\lvert\det A\rvert$ | [Ch. 4](04-multivariable.md#the-log-det-ledger) |
    | 2 | Normalizing-flow pullback density | $\log p_u(u) = \log p(T(u)) + \log\lvert\det J_T(u)\rvert$ | [Ch. 4](04-multivariable.md#the-log-det-ledger); [flows-and-neutra](#flows-and-neutra) above |
    | 3 | Gaussian-evidence Occam term | $-\tfrac12\log\det H$ | [Ch. 8](08-probability.md#laplace); [Ch. 22](22-inference.md#the-occam-term) |

    All three are the change-of-variables theorem
    ([Ch. 4](04-multivariable.md#change-of-variables)) applied to a linear
    lensing map, a nonlinear flow, and the linear whitening map implicit in a
    Gaussian integral. Rows 1 and 2 read the SAME theorem in opposite
    directions; row 3 integrates it instead of evaluating it pointwise. No
    fourth costume is left to find — every $\log|\det|$ this book computes is
    one of these three, or (leapfrog, above) a degenerate case where the
    answer is exactly zero.

!!! note "γ Ledger"
    **What this chapter rules in or out about $\gamma = 1.103$:** no digit of
    it — this chapter is the *instrument*, not the reading. What it fixes is
    the standard a reading has to clear before it counts: $\hat R$ close to
    $1$, ESS large enough that the number you report is not three noisy
    draws wearing a large sample's clothing. A saddle MAP, an
    $\hat R\approx22$ from two failed metric repairs, and a 128-particle
    tempered SMC run that actually converged, is the money number's own
    instance of exactly this audit.
    [Chapter 25](25-money-number.md#the-money-number) tells you whether it
    clears the bar.

## Connect to the repo { #connect }

- [`reproductions/claude-giga-lens/cgl/e2.py:531-721`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L531) —
  `laplace_evidence` and `build_metric_cov`: the Hessian eigendecomposition,
  the PD-vs-indefinite branch, and the exact sentence this chapter quotes on
  the mass-matrix convention.
- `reproductions/claude-giga-lens/cgl/samplers/bj_smc.py:69-97`,
  `bj_nuts.py:106-165`, `remc_pt.py:115-160`,
  `baseline_gigalens.py:106-135` — four libraries' momentum-metric plumbing
  side by side; the `inverse_mass_matrix`-vs-`covariance_matrix` trap is
  visible directly in the diffs between them.
- [`reproductions/claude-giga-lens/cgl/e2.py:756-836`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L756)
  (`run_correlated_smc`) and `cgl/samplers/common.py:297-360`
  (`run_adaptive_tempered_smc`) — the actual 128-particle basin-local SMC
  and its adaptive $\lambda$-schedule / resampling machinery.
- `reproductions/claude-giga-lens/cgl/samplers/neutra.py` and `cgl/flows.py` —
  the NeuTra recipe end to end: flow fit on floored-SVI samples, ChEES-HMC
  in the pullback space.
- `reproductions/claude-giga-lens/cgl/metrics.py:23-68` — `rank_diagnostics`,
  the real rank-normalized split-$\hat R$/ESS this chapter's hand-worked
  formula approximates.
- `reproductions/claude-giga-lens/papers/main.tex` —
  [§4.2, two-stage PHMC](../current/claude-giga-lens/index.md#sec:twostage),
  [§6.3, the sampler saga](../current/claude-giga-lens/index.md#sec:samplersaga),
  [§7.6, the A100 phase](../current/claude-giga-lens/index.md#sec:p2c), and
  Table `tab:basinflip`.
- `reproductions/claude-giga-lens/CAMPAIGN.md`, "P1c metric-fix attempts —
  2026-07-10" — the saddle diagnosis in the campaign's own words, both
  failed repairs included.

## Exercises { #exercises }

??? question "Exercise 23.1 — the convention trap, by hand"
    A toy posterior covariance $\Sigma = \mathrm{diag}(100, 0.01)$ — one
    loose direction, one tight one, condition number $10^4$. (a) Compute the
    correct mass matrix $M = \Sigma^{-1}$. Which direction gets the heavier
    mass, and why is that the right physical choice? (b) Suppose you mistake
    `blackjax`'s `inverse_mass_matrix` convention for TFP's
    `covariance_matrix` one and hand $\Sigma$ *directly* to TFP's momentum
    distribution instead of inverting it first. What (wrong) mass does the
    tight direction get, and what goes wrong in leapfrog because of it?

    ??? success "Solution"
        (a) $M = \Sigma^{-1} = \mathrm{diag}(0.01, 100)$
        <!-- check: ch23.toy_mass_loose = 0.01 ± 1e-9 -->
        <!-- check: ch23.toy_mass_tight = 100.0 ± 1e-9 -->:
        the *tight* direction (variance $0.01$) gets the *heavy* mass
        ($100$), the loose direction (variance $100$) gets the light mass
        ($0.01$) — correct, since a heavy mass moves slowly per momentum
        kick (small, careful steps where the posterior is narrow) and a
        light mass moves fast (covering the wide direction quickly).

        (b) Handing $\Sigma$ directly to a parameter that wants $M$ gives
        the tight direction mass $0.01$ — light instead of heavy. Leapfrog
        now accelerates fast exactly where the posterior is narrowest,
        overshoots every step, and either diverges outright or is rejected
        almost every proposal; the loose direction, meanwhile, gets the
        wrongly heavy mass $100$ and crawls where it should roam freely —
        exactly backwards, in both directions at once.

??? question "Exercise 23.2 — $\hat R$ on two chains that never meet"
    Two toy chains, four draws each: chain 1 $= (1,2,3,4)$, chain 2 $=
    (5,6,7,8)$ — deliberately non-overlapping. Using
    $\widehat{\mathrm{var}} = \tfrac{n-1}{n}W + \tfrac{B}{n}$ and
    $\hat R = \sqrt{\widehat{\mathrm{var}}/W}$, compute $W$, $B$, and
    $\hat R$. Is $\hat R$ bounded above? How does this toy's value compare to
    the money-number posterior's own $\hat R\approx22$?

    ??? success "Solution"
        Each chain's own mean is $2.5$ and $6.5$; each chain's within-chain
        variance (ddof$=1$) is $\tfrac{2.25+0.25+0.25+2.25}{3} = 5/3$, so
        $W = 5/3 \approx 1.667$
        <!-- check: ch23.toy_rhat_W = 1.6667 ± 0.001 -->.
        The grand mean is $4.5$, so
        $B = \tfrac{n}{m-1}\sum_j(\bar\theta_j-\bar\theta)^2 =
        4\times\big[(2.5-4.5)^2+(6.5-4.5)^2\big] = 32$
        <!-- check: ch23.toy_rhat_B = 32.0 ± 0.01 -->.
        Then $\widehat{\mathrm{var}} = \tfrac34(5/3) + 32/4 = 9.25$
        <!-- check: ch23.toy_rhat_varhat = 9.25 ± 0.01 -->
        and $\hat R = \sqrt{9.25/1.667} \approx 2.356$
        <!-- check: ch23.toy_rhat = 2.3558 ± 0.001 -->.
        $\hat R$ has no upper bound in principle — the more disjoint the
        chains or the tinier their own internal spread, the larger it grows.
        The real posterior's $\hat R\approx22$ is roughly nine times worse
        than this already-broken toy: chains not merely in different ranges,
        but far more confident in their own (wrong) location than these
        four-draw chains are.

??? question "Exercise 23.3 — the Occam term is half of a Gaussian integral"
    Redo this chapter's substitution for $d=1$: show
    $\int\exp\!\big(\!-\tfrac12 A(a-\hat a)^2\big)\,da = \sqrt{2\pi/A}$ by
    substituting $u=\sqrt A\,(a-\hat a)$. Chapter 8's toy ridge fit
    (`worked_examples.py --show ch08`) has $\log\det A = 4.025$
    <!-- check: ch08.toy_logdetA = 4.0254 ± 0.001 -->
    and reports `toy_logL_marg_style` $= -4.102$ before restoring the
    $\log Z$ normalization, and `toy_logZ_closed` $= -3.183$
    <!-- check: ch08.toy_logZ_closed = -3.183 ± 0.001 -->
    after. Which *half* of your Gaussian integral is already inside
    `marg.py`'s own $-\tfrac12\log\det A$, and which half is the separate
    constant Chapter 8 adds afterward?

    ??? success "Solution"
        With $u=\sqrt A(a-\hat a)$, $da=du/\sqrt A$, the integral becomes
        $\tfrac1{\sqrt A}\int\exp(-u^2/2)\,du = \sqrt{2\pi/A}$, whose log
        splits into two additive pieces: $-\tfrac12\log A$ (depends on $A$)
        and $\tfrac12\log(2\pi)$ (does not). `marg.py`'s own $\log L$
        (`cgl/marg.py:52-55`) already contains the *first* half — its
        $-\tfrac12\log\det A$ term is literally this integral's $1/\sqrt A$
        factor, in log form. The *second* half is exactly the normalization
        `marg.py`'s docstring drops as `"+ const"`
        (`reproductions/claude-giga-lens/papers/main.tex:471`), which
        Chapter 8 restores by hand: $-4.102 + \tfrac12\log(2\pi) = -4.102 +
        0.919 = -3.183$, matching `toy_logZ_closed` exactly. Both halves are
        one substitution; the code just never needed the $A$-independent
        half, since it cancels in any difference between two evaluations of
        the same model.

??? question "Exercise 23.4 — ESS from an autocorrelation time"
    [Chapter 7](07-fourier.md#psd-and-autocorrelation) measured an integrated
    autocorrelation time $\tau_{\mathrm{int}} = 7.453$
    <!-- check: ch07.tau_int_v3_11lag = 7.453 ± 0.01 -->
    for the fine drizzle product's *pixel* noise. Using
    $\mathrm{ESS} \approx N/\tau_{\mathrm{int}}$ — the same reduction, now applied
    to a Markov chain instead of a stationary noise field — what is the
    effective sample size of a single HMC chain with $N=5000$ post-warmup
    draws and that same autocorrelation time? Is that a healthy ESS by this
    book's own $\mathrm{ESS}_{\min}\ge1000$ convergence gate
    (`reproductions/claude-giga-lens/papers/main.tex:569`)?

    ??? success "Solution"
        $\mathrm{ESS} \approx 5000/7.453 \approx 670.9$
        <!-- check: ch23.ess_from_tauint = 670.87 ± 0.1 -->.
        That is *below* the campaign's own $\mathrm{ESS}_{\min}\ge1000$
        until-converged gate — a single chain this correlated, even with
        5000 raw draws, would not by itself clear the bar the book's own
        Track-B protocol sets. It would need either more raw draws, several
        more chains pooled together, or a shorter autocorrelation time (a
        better-mixing sampler, or a better mass matrix) to reach 1000
        independent-equivalent draws — exactly the same lever this chapter's
        "hmc-and-the-metric" section pulled on the real T2 posterior.
