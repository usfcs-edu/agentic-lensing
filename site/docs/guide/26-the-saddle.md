# Spine 2: the saddle, the invalid metric, and the SMC rescue

[Ch. 25](25-money-number.md#the-chain) walks the chain that produces
$\gamma_{\mathrm{binned}}(\mathrm{corr,low}) = 1.103$ and asks whether the
number is right. This chapter asks the prior question: could the campaign
extract that number at all? The posterior it needed to sample sits at the
object [Ch. 5](05-linear-algebra.md#definiteness-and-saddles) already taught
you to recognize — a saddle, not a mode — and the standard fix, a
Hamiltonian-Monte-Carlo momentum built from the local curvature
([Ch. 23](23-samplers.md#hmc-and-the-metric)), turns out to be the identical
formula Ch. 5 showed you working correctly one section earlier, for a
completely different purpose. Extracting the money number cost the campaign a
real pivot — from a bank of separately-seeded HMC chains, each carrying its
own local metric, to a single tempered population that carries none — and
this chapter is why nothing short of that pivot could have worked.

## The MAP is a saddle { #the-map-is-a-saddle }

Ch. 5 gave you both halves of the vocabulary this section needs: a symmetric
matrix is indefinite when it carries eigenvalues of both signs, and an
indefinite Hessian at a zero-gradient point is what a saddle *is*, in
coordinates. It also quoted, on this exact real fit, the numbers this section
explains the origin of: the campaign's correlated-likelihood posterior for
the binned-low product — forty-six dimensional, once
[Ch. 22](22-inference.md#marginalising-linear-amplitudes) analytically
profiles out the source's linear shapelet amplitudes — has a Laplace Hessian
with minimum eigenvalue $-14.85$ and five negative directions.
<!-- check: ch05.saddle_min_eig = -14.85 ± 0.01 -->
<!-- check: ch05.saddle_n_negative = 5 ± 0 -->
<!-- check: ch05.saddle_ndim = 46 ± 0 -->

Here is how it got missed. `map_polish`
(`reproductions/claude-giga-lens/cgl/e2.py:485-499`) is plain L-BFGS on
$f(\mathbf z) = -\log p(\mathbf z)$, where $\mathbf z$ is the model's full
(unconstrained, bijector-mapped) parameter vector: quasi-Newton steps built
from gradients alone, stopped the instant $\lVert\nabla f\rVert$ is small.
That stopping rule cannot distinguish a genuine minimum of $f$ — a bowl in
every direction, meaning a genuine peak of $\log p$ — from a saddle, a bowl
in most directions and a ridge in a few; both have zero gradient, and only a
second-order check tells them apart ([Ch. 2](02-derivatives.md#why-second-order)
made this point on a toy function; this is the same statement on real data).
The campaign's own second-order check, `laplace_evidence`
(`reproductions/claude-giga-lens/cgl/e2.py:531-572`), builds

$$
H \;\equiv\; -\nabla^2_{\mathbf z}\log p(\mathbf z)\big|_{\mathbf z^\star}
$$

— the Hessian of the *negative* log-posterior, so that $H$ positive definite
is the correct diagnosis of a genuine peak — and eigendecomposes it. At the
point `map_polish` reported, it is not positive definite. The consequence:
the saddle-consistent point $\gamma=1.27$ scores $\log p=-4757$, while
$\gamma=1.10$ — a direction `map_polish` never explored, because its
gradient was already near zero at the saddle — scores $74$ nats higher.
<!-- check: ch05.gamma_at_saddle_map = 1.27 ± 0.01 -->
<!-- check: ch05.logp_at_saddle_map = -4757 ± 1 -->
<!-- check: ch05.gamma_at_true_peak = 1.10 ± 0.01 -->
<!-- check: ch05.logp_at_true_peak = -4683 ± 1 -->
<!-- check: ch05.saddle_logp_gain = 74 ± 1 -->

This was not one optimizer's bad day. Run the same Laplace check on every
real-data basin the campaign fit under the correlated likelihood, and exactly
one comes back positive definite: the fine-scale, $3.2\times$-upsampled steep
basin, minimum eigenvalue $+0.108$, zero of forty-six eigenvalues needing to
be floored.
<!-- check: ch26.fine_steep_min_eig = 0.108 ± 0.005 -->
<!-- check: ch26.fine_steep_n_floored = 0 ± 0 -->
<!-- check: ch26.fine_steep_ndim = 46 ± 0 -->
That is the one basin that mixed cleanly under a standard preconditioned HMC
chain, $\hat{R} = 1.03$; every indefinite basin — the money product
included — froze.
<!-- check: ch26.fine_steep_rhat = 1.03 ± 0.01 -->
Across the whole campaign, convergence tracks Hessian *definiteness*, not
which product, not how many particles, not how carefully the chain was
tuned. The rest of this chapter is why.

## The same matrix twice { #the-same-matrix-twice }

[Ch. 8](08-probability.md#laplace) derives the Laplace approximation in
general and previews exactly this failure mode; here it is specialized, with
$H$ from the section above. At a genuine mode — $H$ positive definite — the
second-order Taylor expansion of $\log p$ *is* a Gaussian approximation:

$$
\log p(\mathbf z) \approx \log p(\mathbf z^\star) - \tfrac12(\mathbf z-\mathbf z^\star)^{\mathsf T} H (\mathbf z - \mathbf z^\star)
\ \Longrightarrow\
p(\mathbf z)\approx\mathcal N(\mathbf z^\star,\Sigma),\quad \Sigma = H^{-1}
$$

($\Sigma$ denotes the Laplace covariance here, unrelated to
$\Sigma_{\mathrm{cr}}$.) [Ch. 5](05-linear-algebra.md#symmetric-2x2)
eigendecomposes any symmetric matrix as $H=V\,\mathrm{diag}(\lambda_i)\,V^{\mathsf T}$
with $V$ orthogonal, so

$$
\begin{equation}
\label{eq:laplace-cov}
\Sigma = V\,\mathrm{diag}\!\left(\frac1{\lambda_i}\right) V^{\mathsf T}.
\end{equation}
$$

This is exactly what a Hamiltonian-Monte-Carlo sampler wants for its
momentum: a covariance built from the local curvature, so that a fixed-size
leapfrog step covers a comparable number of posterior standard deviations in
every direction, however stretched the posterior is
([Ch. 23](23-samplers.md#hmc-and-the-metric) already flags which of $\Sigma$
or $M=\Sigma^{-1}$ a given library's own "mass" argument wants — a real but
separate convention trap from the one below). `build_metric_cov`
(`reproductions/claude-giga-lens/cgl/e2.py:654-676`) uses
$\eqref{eq:laplace-cov}$ exactly whenever $H$ is positive definite, and says
so in its own docstring: it "reproduces the pre-fix fine-steep run
bit-for-bit."

At the saddle, five of the $\lambda_i$ in $\eqref{eq:laplace-cov}$ are
negative, so five diagonal entries of $\Sigma$, in its own eigenbasis, are
negative. A covariance cannot have a negative eigenvalue — variance is a mean
squared deviation — so $\eqref{eq:laplace-cov}$ has stopped meaning anything.
The fix the campaign's own legacy code tried first replaces every $\lambda_i$
by its absolute value:

$$
\begin{equation}
\label{eq:naive-cov}
\Sigma_{\text{legacy}} = V\,\mathrm{diag}\!\left(\frac1{|\lambda_i|}\right) V^{\mathsf T}.
\end{equation}
$$

(`reproductions/claude-giga-lens/cgl/e2.py:566-567`, literally
`cov = (V * (1.0 / w_abs)) @ V.T`.) $\eqref{eq:naive-cov}$ is real, symmetric,
strictly positive-eigenvalued — syntactically a legitimate covariance. It is
also, index for index, the *same* formula
[Ch. 5's tip box](05-linear-algebra.md#definiteness-and-saddles) already
showed you working correctly: the saddle-free-Newton step an earlier stage of
this pipeline takes to escape a different saddle is
$\mathbf{step} = V\,\mathrm{diag}(1/(|\lambda_i|+\text{damping}))\,V^{\mathsf T}\,\mathbf g$
(`reproductions/foundry-i/32_saddlefree_newton.py`) — the identical matrix,
applied to a gradient instead of used as a bilinear form.

Why is it fine as a step and broken as a covariance? Because the two uses are
checked differently. A Newton-style step's only job is to point somewhere
that decreases $f$; the saddle-free-Newton script's own backtracking line
search evaluates $f$ at the proposed point and rejects the step outright if
it fails to improve
(`reproductions/foundry-i/32_saddlefree_newton.py:207-222`) — the formula's
claim on truth is falsified or confirmed, every single iteration, against the
real function. $\eqref{eq:naive-cov}$ makes no such per-step claim. It
defines a distribution once, up front, and HMC's *correctness* — its
acceptance step preserves the posterior under any fixed momentum
covariance — does not depend on that distribution being accurate; only the
sampler's *efficiency* does, and inefficiency does not raise an error. It
fails to mix, silently: both principled repairs the campaign tried next,
guaranteed positive definite by construction unlike $\eqref{eq:naive-cov}$'s
sign-flip, left $\hat{R}$ at $21$ (a floored-diagonal metric) and $22.3$ (a
full-rank SVI covariance) — worse than doing nothing.
<!-- check: ch23.saddle_rhat_diagraw = 21.0 ± 0.1 -->
<!-- check: ch25.rhat_saddle_metric = 22.3 ± 0.05 -->

The deeper reason $\eqref{eq:naive-cov}$ misleads rather than approximates: a
negative $\lambda_i$ does not mean "this direction is broad." It means the
quadratic model has no well there at all — $\log p$ keeps *rising* as you
move away from $\mathbf z^\star$, to second order — so there is no local
Gaussian width to estimate. Flipping the sign manufactures a well of the
wrong shape, at a width set by whatever the wrong-signed curvature's
*magnitude* happens to be. Here that magnitude, $14.85$, is one of the
largest in the whole forty-six-dimensional spectrum, which manufactures a
narrow, *confident* direction exactly where the truth is an unconfined
escape route. `laplace_evidence`'s own docstring names this precisely: the
legacy metric "is INVALID on indefinite Hessians... so the stage-1 leapfrog
gets the geometry wrong and never mixes"
(`reproductions/claude-giga-lens/cgl/e2.py:541-546`).

!!! tip "You already know this"
    Adam and RMSProp scale each parameter's gradient step by roughly the
    inverse square root of a running average of its squared gradient — a
    cheap, per-parameter, curvature-flavored heuristic that makes descent
    converge faster. Nobody treats that same quantity, $1/\sqrt{v_t}$, as the
    standard deviation of a posterior over the weights, because it was built
    to answer "how far can I safely step here," not "how much does the data
    constrain this parameter" — and the two questions have different answers
    whenever the loss surface is not a clean bowl. $\Sigma_{\text{legacy}}$
    makes exactly that substitution.

Getting the sign right, it turns out, is necessary and not sufficient: even
the two positive-definite-by-construction repairs above still left $\hat{R}$
elevated. The next section is why.

## The nuisance { #the-nuisance }

$\hat{R}$, as usually reported, is one number standing in for forty-six —
typically the worst of them. That convention is doing real work here: the
headline $\hat{R} = 22.3$ describes one specific parameter pair, not
$\gamma$. The campaign traces it to the same source galaxy's *centre*,
described two ways inside one model: once by the smooth Sérsic light profile
([Ch. 10](10-galaxies.md#the-sersic-profile)), once by the flexible shapelet
basis [Ch. 22](22-inference.md#marginalising-linear-amplitudes) profiles out
of the *linear* part of the fit — a residual, nonlinear version of the same
light-assignment ambiguity survives in the remaining nonlinear parameters.
Chains split into two clusters, centred roughly $0.15$ and $0.09$ arcsec
either side of the pooled mean, and that *between*-cluster spread is what
inflates $\hat{R}$ to $22.3$ on one component and $15.1$ on the other.
<!-- check: ch21.rhat_source_centre_hi = 22.3 ± 0.05 -->
<!-- check: ch21.rhat_source_centre_lo = 15.1 ± 0.05 -->
<!-- check: ch21.source_centre_cluster_neg = -0.15 ± 0.01 -->
<!-- check: ch21.source_centre_cluster_pos = 0.09 ± 0.01 -->
[Ch. 21](21-degeneracies.md#degeneracy-is-gauge-symmetry) names this pattern
precisely: a direction the data barely distinguishes is a near-flat
direction of $H$ — the same object that just produced a saddle one section
ago.

$\hat{R}$'s own arithmetic makes it obvious why one badly-mixed pair can
dominate the headline while leaving $\gamma$ untouched.
[Ch. 23](23-samplers.md#hmc-and-the-metric) works a toy case by hand: two
chains that never overlap at all, $[1,2,3,4]$ and $[5,6,7,8]$, give a
between-chain variance $B=32$ against a within-chain variance $W\approx1.67$,
and $\hat{R}=\sqrt{\left(\frac{n-1}{n}W+\frac{B}{n}\right)/W}\approx2.36$.
<!-- check: ch23.toy_rhat_B = 32.0 ± 0.01 -->
<!-- check: ch23.toy_rhat_W = 1.6667 ± 0.001 -->
<!-- check: ch23.toy_rhat = 2.3558 ± 0.001 -->
$\hat{R}$ is computed *per parameter*: nothing in that formula couples one
parameter's between-chain split to a second parameter whose chains happen to
agree closely. A source-centre component whose chains straddle two clusters,
$2.36$-style, sits in the *same* draw as a $\gamma$ whose chains all agree —
and reporting only the worst parameter's $\hat{R}$ as "the fit's $\hat{R}$"
erases that distinction. CAMPAIGN.md's own diagnosis makes the disaggregation
explicit: of the forty-six correlated-likelihood parameters, $\gamma$'s own
mixing is unremarkable, nowhere near the two source-centre components driving
the headline number. That is what "decoupled" means operationally:
$\gamma_{\mathrm{best}}\approx1.10$
<!-- check: ch05.gamma_at_true_peak = 1.10 ± 0.01 -->
is stable across chains regardless of which source-centre cluster a given
chain is stuck in. What is *not* trustworthy until that source-centre
direction actually mixes is $\gamma$'s *width* — an unmixed chain reports a
confident-looking $\sigma$ that is really just the spread of one un-escaped
local basin.

One more way this compounds: the campaign's usual two-stage HMC recipe
re-estimates its momentum metric from stage-1 draws before running stage 2
([Ch. 23](23-samplers.md#hmc-and-the-metric)). When stage 1 is itself stuck
in one source-centre cluster, the re-estimated metric inherits that same
blind spot, and stage 2 mixes *worse* than stage 1 — the re-preconditioning
move that reliably rescues a merely slow chain actively poisons a
multimodal one.

## Why SMC rescued it { #why-smc-rescued-it }

[Ch. 23](23-samplers.md#tempering-and-smc) introduces tempered SMC in
general: a population of particles carried along an annealing path from an
easy reference distribution to the true posterior, resampled at every step
in proportion to how well each particle's importance weight held up. The
campaign's own correlated-posterior sampler names exactly what it is:
`run_correlated_smc` (`reproductions/claude-giga-lens/cgl/e2.py:756-769`)
calls itself, in its own docstring, "metric-free" — tempering plus
systematic resampling crossing the source-centre sub-modes that froze the
fixed-leapfrog HMC chain.

The reference distribution it anneals *from* is neither the model's prior
nor a single local Hessian. `fit_gaussian_from_draws`
(`reproductions/claude-giga-lens/cgl/e2.py:732-744`) builds it empirically,
by pooling the draws of the earlier, individually unconverged HMC chains —
the same chains whose $\hat{R}=22.3$ looked like pure failure in the section
above. Those chains never mixed *within* themselves, but collectively,
across many random seeds, they visited both source-centre clusters; a
pooled sample covariance across a population straddling two clusters is, by
construction, wide in exactly the direction a single local metric gets
wrong. The campaign inflates that pooled covariance by a further factor of
three,
<!-- check: ch26.smc_cov_inflate = 3.0 ± 0 -->
"fattens the tails for safe annealing" in its own comment
(`reproductions/claude-giga-lens/cgl/e2.py:738`), seeds $128$ particles from
it,
<!-- check: ch25.n_smc_particles = 128 ± 0 -->
and anneals the tempering parameter $\lambda$ — never $\beta$ in this book;
[Ch. 8](08-probability.md#evidence-and-nats) reserves $\beta$ for the source
position — from the reference at $\lambda=0$ to the true correlated
posterior at $\lambda=1$ in $28$ adaptively-chosen steps, effective sample
size $77$ to $118$ at each one.
<!-- check: ch25.n_smc_lambda_steps = 28 ± 0 -->
<!-- check: ch25.ess_smc_low = 77 ± 0 -->
<!-- check: ch25.ess_smc_high = 118 ± 0 -->

The mechanism that actually crosses the nuisance direction is resampling,
not the local HMC mutation kernel run at each temperature. Systematic
resampling reweights and duplicates particles by the *true* incremental
likelihood ratio between one temperature and the next — never by a local
quadratic model — so weight sitting in a wrongly-placed source-centre
cluster is thinned and weight near the better basin is duplicated, at every
step, regardless of what any one particle's own local proposal thought the
geometry looked like. A single HMC chain has one current position and one
fixed metric; it cannot do this. A population of $128$, corrected against
the true likelihood at every temperature, can.

One accounting identity falls out for free.
[Ch. 8](08-probability.md#evidence-and-nats) defines the evidence $Z$ as an
integral over the whole posterior; the SMC driver both this path and
[Ch. 25](25-money-number.md#the-evidence-flip)'s diagonal-likelihood run
share estimates $\log Z$ as the running sum of each step's average
incremental log-likelihood weight
(`reproductions/claude-giga-lens/cgl/samplers/common.py:334-340`) — no
separate calculation, no second sampler. The *same* $128$-particle run that
untangled the source-centre nuisance to extract
$\gamma_{\mathrm{binned}}(\mathrm{corr,low})$ also weighed the low basin
against the steep one, in the pass that produces
[Ch. 25](25-money-number.md#the-evidence-flip)'s $191$-nat flip.

The chapter's own title formula now has three instances, not two, and none
of them contradicts either of the others. $V\,\mathrm{diag}(1/|\lambda_i|)\,V^{\mathsf T}$
is a fine Newton *direction* and a broken *covariance*. $\hat{R}=22.3$ is a
fair verdict on the source-centre pair's own marginal and an irrelevant one
on $\gamma$'s. An HMC chain that never converged by the $\hat{R}$ criterion
was worthless as a posterior sample and exactly the right input for
estimating an SMC reference covariance's scale. A formula, a diagnostic, a
discarded chain — none of these is a fact standing alone. Each is a fact
*about a purpose*.

!!! note "γ Ledger"
    **What this chapter rules in or out about $\gamma = 1.103$:** it does not
    move the number — that arithmetic is
    [Ch. 25](25-money-number.md#the-money-number)'s. What it establishes is
    the *right* to trust the sampler that produced it. The saddle at the
    money product's MAP does not put $\gamma$ itself in doubt: the
    high-density point $\gamma_{\mathrm{best}}\approx1.10$ sits $74$ nats
    above the saddle-consistent value and is stable regardless of which
    source-centre cluster a given chain is stuck in. The alarming
    $\hat{R}=22.3$ a naive read would apply to the whole draw belongs to a
    different, decoupled parameter pair. What this chapter does *not*
    license is skipping
    [Ch. 25](25-money-number.md#the-sigma-arithmetic)'s uncertainty
    bookkeeping — only the metric-free SMC run, not the frozen HMC chains,
    produced a $\sigma$ worth trusting.

## Connect to the repo { #connect }

- [`reproductions/claude-giga-lens/cgl/e2.py:485-499`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L485)
  (`map_polish`) — the L-BFGS stage that stops on gradient alone and cannot
  see a saddle.
- [`reproductions/claude-giga-lens/cgl/e2.py:531-572`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L531)
  (`laplace_evidence`) — the second-order check that caught it, and the
  legacy $\Sigma_{\text{legacy}}$ formula's own docstring diagnosis.
- [`reproductions/claude-giga-lens/cgl/e2.py:654-676`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L654)
  (`build_metric_cov`) — the three metric choices (`laplace`, `diagraw`,
  `svi_cov`) this chapter's second section compares.
- [`reproductions/claude-giga-lens/cgl/e2.py:732-769`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L732)
  (`fit_gaussian_from_draws`, `run_correlated_smc`) — the metric-free path:
  pooled-chain reference covariance, inflation, tempered anneal.
- [`reproductions/claude-giga-lens/cgl/samplers/common.py:297-357`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/samplers/common.py#L297)
  (`run_adaptive_tempered_smc`) — the shared blackjax driver; `:334-340` is
  where $\log Z$ accumulates.
- [`reproductions/foundry-i/32_saddlefree_newton.py`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/foundry-i/32_saddlefree_newton.py)
  — the earlier saddle in this same pipeline, and the line search that makes
  the *same* formula valid there.
- `reproductions/claude-giga-lens/CAMPAIGN.md`, "P1c metric-fix attempts —
  2026-07-10" and "P1c MONEY NUMBER — 2026-07-14" — the diagnosis and the
  rescue, in the campaign's own words.
- main.tex, [§6.3, "Why sampling this posterior required tempered SMC"](../current/claude-giga-lens/index.md#sec:samplersaga) —
  the published record this chapter walks through.

## Exercises { #exercises }

??? question "Exercise 26.1 — build both matrices by hand"
    A toy Hessian (of $-\log p$, this chapter's sign convention) at a
    candidate point is

    $$
    H = \begin{pmatrix} 2 & 0 \\ 0 & -8 \end{pmatrix}.
    $$

    (a) Classify the point using
    [Ch. 5](05-linear-algebra.md#definiteness-and-saddles)'s trace/determinant
    test. (b) Write down $\Sigma = V\,\mathrm{diag}(1/\lambda_i)\,V^{\mathsf T}$
    from $\eqref{eq:laplace-cov}$ directly ($H$ is already diagonal, so
    $V=I$), and say precisely what is wrong with it as a covariance.
    (c) Write down $\Sigma_{\text{legacy}}$ from $\eqref{eq:naive-cov}$, and
    say which of its two diagonal entries is a genuine uncertainty estimate
    and which is fiction — and why the fictional one looks *more* confident
    (smaller) than the genuine one.

    ??? success "Solution"
        (a) $\det H = 2\times(-8) = -16 < 0$: indefinite — a saddle, not a
        mode. (b) $\Sigma = \mathrm{diag}(1/2,\,-1/8) = \mathrm{diag}(0.5,\,-0.125)$.
        The second entry is negative — not a variance. $\Sigma$ is not a
        covariance matrix. (c) $\Sigma_{\text{legacy}} = \mathrm{diag}(1/2,\,1/8)
        = \mathrm{diag}(0.5,\,0.125)$. The first entry, $0.5$, is genuine:
        $\lambda_1=2>0$ really is a bowl there, and $\Sigma_{11}=1/\lambda_1$
        really is that bowl's width. The second entry, $0.125$, is fiction:
        $\lambda_2=-8$ means $-\log p$ curves *downward* along that axis — a
        ridge, not a well — and there is no local Gaussian width to report
        at all. Because $|-8|>|2|$, the fictional entry ($0.125$) is
        *smaller* — more confident-looking — than the genuine one ($0.5$):
        the sign-flip does not just invent an answer, it invents an
        overconfident one exactly where the model has the least business
        being confident.

??? question "Exercise 26.2 — the second parameter the headline $\hat R$ hides"
    [Ch. 23](23-samplers.md#hmc-and-the-metric) computes $\hat{R}\approx2.36$
    by hand for two never-overlapping chains, $[1,2,3,4]$ and $[5,6,7,8]$.
    Suppose a second parameter, sampled by the *same two chains* at the
    *same four steps*, happens to draw $[10.0,10.1,9.9,10.0]$ from chain 1
    and $[10.0,9.9,10.1,10.0]$ from chain 2 — no cluster split at all. Using
    the same $\hat{R}=\sqrt{\left(\frac{n-1}{n}W+\frac{B}{n}\right)/W}$
    formula, is this second parameter's $\hat{R}$ closer to $1$ or to
    $2.36$? What does a report that quotes only "$\hat{R}=2.36$" for "the
    fit" actually tell you about this second parameter, and what would you
    need to check before trusting it?

    ??? success "Solution"
        The two chains for the second parameter have essentially the same
        mean ($\approx10.0$), and only chain-internal noise separates
        them, so the between-chain variance $B\approx0$ while the
        within-chain variance $W$ is small but nonzero — driving $\hat{R}$
        to very nearly $1$, nowhere near $2.36$. A report that quotes a
        single "$\hat{R}=2.36$ for the fit" tells you *nothing at all*
        about this second parameter — $\hat{R}$ is computed per parameter,
        and the headline number is, by the usual convention, the worst
        offender across all of them, not an average. Before trusting the
        second parameter you would need its *own*, disaggregated $\hat{R}$ —
        exactly the check CAMPAIGN.md ran to clear $\gamma$ of the
        source-centre pair's pathology.

??? question "Exercise 26.3 — the 74 nats, as a probability ratio"
    [The MAP is a saddle](#the-map-is-a-saddle) reports a $74$-nat gap
    between $\log p$ at $\gamma=1.27$ and $\log p$ at $\gamma=1.10$.
    [Ch. 8](08-probability.md#evidence-and-nats) establishes that
    log-probabilities in nats convert to a probability *ratio* by
    exponentiating. Compute $e^{74}$ (an order of magnitude is enough), and
    say in one sentence what it would mean for an optimizer to report the
    *lower*-probability point as "the" answer when a point $e^{74}$ times
    more probable sits nearby.

    ??? success "Solution"
        $e^{74} = 10^{74/\ln 10} \approx 10^{32.1}$ — on the order of
        $10^{32}$, a ratio with no everyday analogy (larger than the number
        of atoms in a human body). Reporting $\gamma=1.27$ as "the" MAP when
        $\gamma=1.10$ scores $\log p$ higher by this much is not a rounding
        error or a matter of taste between two comparably-good fits — it is
        the optimizer stopping at a point the model itself considers
        *astronomically* less probable than one nearby, because the
        stopping rule (a small gradient) cannot see the difference.
