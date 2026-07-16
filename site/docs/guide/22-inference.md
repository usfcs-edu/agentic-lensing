# Lens modelling as inference

Chapters 17 through 21 built the mathematics of a single ray: how one point
on the sky maps to one point in the source plane, and how the Jacobian of
that map turns a source's true brightness into an observed image's
brightness. This chapter assembles those rays into the actual machine
`claude-giga-lens` runs against real pixels — a parameterized generative
model of an entire HST cutout, differentiable end to end, evaluated by one
JAX function every time a sampler needs a number. You will count, by hand,
the exact parameter budget that machine carries: 74 numbers wide, until you
notice that 28 of them are linear-regression coefficients that never need to
be sampled at all, and that the Gaussian integral removing them carries the
same Occam term [Chapter 8](08-probability.md#laplace) derived on a
five-point toy. Every $\gamma$ this book has quoted so far — the money number
included — is a byproduct of exactly this function.

!!! abstract "What you can skip"
    You already own differentiable rendering, autodiff through convolutions,
    and ridge regression as a closed-form linear solve; none of that is
    re-derived here. What is new is the wiring: which physical model this
    repo actually samples, why 28 of its 74 parameters vanish before a
    sampler ever sees them, and where $-\tfrac12\log\det A$ shows up as a
    running number instead of a textbook term.

## The forward model { #the-forward-model }

A lens model is not one equation; it is a small graphics pipeline with a
physics constraint bolted onto its first step. `cgl/likelihood.py`'s
`build_marg_model` (`reproductions/claude-giga-lens/cgl/likelihood.py:208`)
builds exactly this pipeline for one real system,
DESI-165.4754−06.0423, at whichever drizzle scale you hand it. Four
ingredients go in:

- **The mass model.** An EPL (elliptical power law, [Chapter
  20](20-profiles.md#the-epl-and-gamma)) plus external shear
  ([Chapter 20](20-profiles.md#external-shear)) bend light. Six numbers
  <!-- check: ch22.n_mass = 6 ± 0 -->
  parameterize the EPL — $\theta_{\mathrm{E}}, \gamma, e_1, e_2$, and a
  center — and two more parameterize the shear, $\gamma_1$ and $\gamma_2$.
  <!-- check: ch22.n_shear = 2 ± 0 -->
  $\gamma$ here is unambiguous: it is the EPL density slope of
  [Chapter 20](20-profiles.md#the-epl-and-gamma), never the shear. The two
  shear components are always written with their subscripts, never a bare
  $\gamma$.
- **The lens light.** Four Sérsic profiles
  <!-- check: ch22.n_lens_light_components = 4 ± 0 -->
  ([Chapter 10](10-galaxies.md#the-sersic-profile)) model the light of the
  deflecting galaxy and a nearby companion, unlensed — they sit in the image
  plane directly, at $\boldsymbol\theta$, with no ray tracing involved. Each
  carries seven parameters: `R_sersic, n_sersic, e1, e2, center_x, center_y,
  Ie`
  <!-- check: ch22.n_lens_light_each = 7 ± 0 -->
  (`reproductions/claude-giga-lens/cgl/likelihood.py:299-303`).
- **The source, Sérsic part.** One more seven-parameter Sérsic profile, this
  one lensed: evaluated not at $\boldsymbol\theta$ but at the source-plane
  position $\boldsymbol\beta$ the lens equation
  ([Chapter 17](17-lens-equation.md#the-lens-equation)) sends it to.
- **The source, shapelet part.** A basis of $28$ shapelet functions at order
  $n_{\max}=6$
  <!-- check: ch20.shapelet_nmax = 6 ± 0 -->
  — a triangular-number count,
  $(n_{\max}{+}1)(n_{\max}{+}2)/2$, already computed in
  [Chapter 20](20-profiles.md#source-models)
  <!-- check: ch20.shapelet_depth_nmax6 = 28 ± 1e-9 -->
  — plus three more parameters that set the basis's pose: a scale radius the
  code calls `beta`, and a center $(x,y)$. That code identifier is *not*
  this guide's $\boldsymbol\beta$: it is the shapelet basis's characteristic
  size (the Refregier 2003 convention), a genuinely different quantity that
  happens to share a name. This guide keeps them apart on purpose — the
  contract that reserves $\boldsymbol\beta$ for source-plane position exists
  precisely so a collision like this one cannot leak into the notation.

Rendering composes these into one image. On a grid oversampled by a factor
$S$ relative to the data's own pixel scale (`supersample` in the code — for
the binned $0.08''$
<!-- check: ch22.delta_pix_v3b = 0.08 ± 0 -->
product this is a factor of $S=2$
<!-- check: ch22.supersample_v3b = 2 ± 0 -->,
turning its $130\times130$
<!-- check: ch22.n_pix_v3b = 130 ± 0 -->
pixel grid into a $260\times260$
<!-- check: ch22.n_pix_supersampled_v3b = 260 ± 0 -->
ray-tracing grid), every image-plane point $\boldsymbol\theta$ solves the
lens equation, every light profile is evaluated, the sum is convolved with
the PSF kernel at that *same* oversampled resolution
(`jax.lax.conv_general_dilated`,
`reproductions/claude-giga-lens/cgl/likelihood.py:342-345`), and only then
is the result average-pooled back down by the supersample factor to the
data's own pixel grid (`average_pool_2d`,
`reproductions/claude-giga-lens/cgl/likelihood.py:346-349`):

$$
\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta),
\qquad
I(\boldsymbol\theta) = \sum_{k=1}^{4} L_k(\boldsymbol\theta) \;+\; L_{\mathrm{src}}(\boldsymbol\beta),
\qquad
M = \mathrm{pool}_S\!\big(\mathrm{PSF} * I\big).
\label{eq:render}
$$

The order in Equation $\eqref{eq:render}$ is not a stylistic choice: PSF
convolution happens *before* the average-pool, on the oversampled grid,
because a real point-spread function has structure at scales finer than the
data pixel and pooling first would throw that structure away. It also means
the PSF kernel handed to the convolution must itself be sampled at the
oversampled scale, not the native one — get that wrong and the effective PSF
silently broadens, which is exactly the failure
[Chapter 8](08-probability.md#chi-squared) already showed you the cost of:
$\chi^2_\nu$ floored at $3.4$
<!-- check: ch08.chi2_nu_psf_broadened = 3.4 ± 0.05 -->
until `assert_psf_sampling`
(`reproductions/claude-giga-lens/cgl/guards.py:74-91`) made this exact
ordering mistake impossible to repeat silently, dropping the floor to
$1.05$
<!-- check: ch08.chi2_nu_psf_fixed = 1.05 ± 0.05 -->.
Finally, $M$ is compared against the observed data pixel by pixel through a
residual $R = Y - M$, weighted either by a diagonal noise model
([Chapter 8](08-probability.md#chi-squared)) or a whitening operator
([Chapter 24](24-correlated-noise.md#convolutional-whitening)) — the
comparison this whole render exists to feed.

!!! tip "You already know this"
    Strip away the astronomy vocabulary and `build_marg_model` is a
    differentiable renderer of exactly the kind you have written for
    graphics or vision: a coordinate transform (the lens equation) feeds a
    scene description (five Sérsic profiles plus a shapelet basis) evaluated
    on an oversampled grid, blurred by a known point-spread function,
    downsampled to the sensor's pixel grid, and compared to a real
    photograph pixel by pixel. The whole pipeline is one `@jax.jit`-compiled
    function, and every step in it is differentiable — the only reason
    gradient-based inference ([Chapter 23](23-samplers.md#why-gradients)) is
    on the table at all.

Add every component's parameter count and you get the model's total width:

$$
\underbrace{8}_{\text{mass + shear}} + \underbrace{28}_{4\times\text{lens light}}
+ \underbrace{7}_{\text{source Sérsic}} + \underbrace{3}_{\text{shapelet pose}}
+ \underbrace{28}_{\text{shapelet amplitudes}} \;=\; 74.
$$

<!-- check: ch22.n_mass_shear = 8 ± 0 -->
<!-- check: ch22.n_lens_light = 28 ± 0 -->
<!-- check: ch22.n_source_sersic = 7 ± 0 -->
<!-- check: ch22.n_shapelet_pose = 3 ± 0 -->
<!-- check: ch22.n_shapelet_amps_n6 = 28 ± 0 -->
<!-- check: ch22.n_params_full = 74 ± 0 -->

That total is not a guide invention: it is the model `foundry-i`'s
`_hmc_lib.py` samples in full
(`reproductions/foundry-i/README.md:217`), and it is exactly what
`reproductions/claude-giga-lens/cgl/e2.py:12-13` calls "74-dim PAPER-scale
z-vectors (EPL+Shear, 4 lens-light Sérsic, Sérsic + Shapelets($n_{\max}=6$)
source with 28 EXPLICIT amps)."

## Marginalising linear amplitudes { #marginalising-linear-amplitudes }

Twenty-eight of those 74 numbers are not like the other 46. Every shapelet
amplitude enters the render in Equation $\eqref{eq:render}$ *linearly*: hold
the mass, shear, lens-light, and source-pose parameters fixed, and the
rendered image is an exact linear combination of 28 fixed "basis" images,
one per shapelet function, each carried through the identical lens-equation
$\to$ PSF-convolve $\to$ pool pipeline as everything else
(`_design_ret`, `reproductions/claude-giga-lens/cgl/likelihood.py:329-352`).
That is the entire justification for treating them differently: a parameter
that appears linearly, under a Gaussian likelihood and a Gaussian prior, has
an *exactly* Gaussian conditional posterior given everything else — which is
precisely [Chapter 8](08-probability.md#laplace)'s claim that the Laplace
approximation is exact for a linear-Gaussian model, now invoked for a real
28-dimensional linear subspace instead of a one-parameter toy.

Collect the 28 unit-amplitude basis images into a design matrix $X_w$ (one
whitened column per shapelet function, `reproductions/claude-giga-lens/cgl/likelihood.py:356-367`)
and the whitened residual into $R_w$. [Chapter 8](08-probability.md#ridge-is-a-prior)
already derived the normal equations for exactly this setup:

$$
A = X_w^\top X_w + \Lambda, \qquad b = X_w^\top R_w, \qquad
\hat a = A^{-1}b \ \ (\text{Cholesky, never a literal inverse}).
\label{eq:normaleq}
$$

`marg.py`'s ridge precision $\Lambda_{ii} = (i+1)/25$
(`reproductions/claude-giga-lens/cgl/marg.py:38-39`) is a smoothness prior:
higher-order (wigglier) shapelet functions are penalized harder, so $A$ stays
positive-definite and the Cholesky solve — never `pinv` — is well posed. $A$
here is a $28\times28$ matrix: exactly one row and column per marginalized
amplitude, no larger. It does *not* grow with the other 46 parameters; those
enter only by reshaping $X_w$ and $R_w$ each time $A$ is recomputed, since
changing the mass model changes which pixels the source light lands on.

Not every linear parameter in this model gets this treatment, and which ones
do is a code-level choice inherited from `gigalens`, not a mathematical
necessity. The five Sérsic amplitudes — one per lens-light component plus
one for the source — are exactly as linear in Equation $\eqref{eq:render}$
as the 28 shapelet amplitudes are. But `gigalens`'s own flag,
`use_lstsq=False` on every Sérsic component versus `use_lstsq=True` on the
shapelets alone (`reproductions/claude-giga-lens/cgl/likelihood.py:301-302`
against `:314`), marginalizes only the second group. The five Sérsic `Ie`
parameters keep an explicit `LogNormal` prior and are sampled directly, like
any nonlinear parameter — which is why the marginalized model has $46 = 41$
sampled nonlinear parameters $+\ 5$ sampled `Ie`'s
(`reproductions/claude-giga-lens/cgl/likelihood.py:35`;
`reproductions/foundry-i/README.md:101-102`), not $46 = 74 - 28$ read
as "everything not a shapelet amplitude minus the shapelets." Both
bookkeepings land on the same 46
<!-- check: ch22.n_params_marg = 46 ± 0 -->,
because the arithmetic is the same subtraction either way:

$$
74 - 28 = 46.
$$

<!-- check: ch22.param_count_check = 0 ± 0 -->
<!-- check: ch22.marg_count_check = 0 ± 0 -->

Marginalizing these 28 amplitudes is not only an elegant derivation; it is a
numerical rescue. [Chapter 5](05-linear-algebra.md#conditioning) already
flagged the full posterior's own Hessian at condition number
$\sim\!10^{14}$
<!-- check: ch05.cond_ill_conditioned = 1e14 ± 1e10 -->,
leaving barely $1.65$
<!-- check: ch05.stable_digits_remaining = 1.65 ± 0.01 -->
stable float64 digits — nowhere near enough to trust a naive solve. The
$28\times28$ ridge normal matrix $A$ this section builds, by contrast,
measures at condition number $1.37\times10^4$
<!-- check: ch05.cond_marg_normal_matrix = 13700 ± 1 -->
at the campaign's own parity point
(`reproductions/claude-giga-lens/CAMPAIGN.md:702`) — an improvement of
roughly $7.3\times10^9$
<!-- check: ch05.cond_improvement_factor = 7.3e9 ± 0.1e9 -->,
restoring $11.5$
<!-- check: ch05.stable_digits_remaining_marg = 11.52 ± 0.01 -->
usable digits. Marginalizing the linear amplitudes does not merely remove 28
dimensions from a sampler's job; it removes the *worst-conditioned* 28
dimensions, handing them instead to a Cholesky solve that a ridge prior has
made numerically boring.

## The Occam term { #the-occam-term }

Every ridge solve carries the term `gigalens`'s own least-squares path
omits. Equation $\eqref{eq:normaleq}$'s log-likelihood is

$$
\log L = -\tfrac12\|R_w\|^2 + \tfrac12\, b^\top\hat a - \tfrac12\log\det A + \text{const},
\label{eq:occam}
$$

the same $-\tfrac12\log\det A$ Occam factor
[Chapter 8](08-probability.md#ridge-is-a-prior) introduced — Log-Det Ledger
row 3, opened there on a one-parameter toy and audited here in production, on
this repo's own 28-column shapelet design. `gigalens`'s stock linear solver
calls `pinv(X^\top WX)` with no ridge and no Occam correction at all
(`reproductions/foundry-i/README.md:93-95`); on the near-rank-deficient
shapelet basis that solve is not merely missing a constant, it is
non-smooth, floors the log-posterior's gradient norm, and never finds a
positive-definite mode
(`reproductions/foundry-i/README.md:95-99`).

The audited number is not a demonstration value — it is the parity harness's
own measurement, checked against `numpy.linalg.slogdet` to $10^{-10}$ at a
stored validation point
(`reproductions/claude-giga-lens/papers/main.tex`, Table `tab:parity`, Gate
E; `reproductions/claude-giga-lens/CAMPAIGN.md:306-311`):

$$
\log\det A = 323.229,
\qquad
\mathrm{cond}(A) \approx 1.4\times10^4.
$$

<!-- check: ch08.occam_logdetA_parity = 323.229 ± 0.001 -->
<!-- check: ch08.occam_condA_parity = 14000 ± 1 -->

At that exact point, dropping the Occam term the way `gigalens`'s plain
`lstsq` path does would over-report the log-likelihood by half of
$\log\det A$ — $161.61$ nats
<!-- check: ch22.occam_correction_production_nats = 161.6145 ± 0.001 -->
— not a rounding error, a number on the same order as the $191.1$-nat
<!-- check: ch25.evidence_swing_nats = 191.1 ± 0.05 -->
evidence swing that decides which basin the data prefer
([Chapter 25](25-money-number.md#the-evidence-flip)). That swing is only
trustworthy once every log-likelihood evaluation being compared — steep
basin and low basin alike — carries this same correction, consistently; a
comparison where one side's Occam term is audited and the other's is silently
dropped is not a Bayes factor at all, whatever units it is reported in. This
is the reason this campaign's evidence numbers exist as a genuine
contribution rather than a rerun of `gigalens` with more compute.

Two of the parity harness's other four gates matter directly here too. Gate
B (the log-posterior itself) and Gate D (the *diagonal limit*: setting the
correlated whitener to the identity must reproduce the plain diagonal
likelihood *exactly*) both pass at machine precision
(`reproductions/claude-giga-lens/papers/main.tex`, Table `tab:parity`) —
[Chapter 24](24-correlated-noise.md#the-diagonal-limit) is where that gate
becomes a methodology in its own right, not merely a unit test.

## The GIGA-Lens recipe { #the-gigalens-recipe }

Having a differentiable log-posterior is necessary but not sufficient; you
still have to sample it. GIGA-Lens's own answer to that problem (Gu et al.
2022) — and the baseline every sampler in this campaign is measured against
(`reproductions/claude-giga-lens/papers/main.tex:549`, contender "S0") — is a
three-step recipe: **multi-start MAP $\to$ SVI $\to$ preconditioned HMC**
(`reproductions/claude-giga-lens/papers/main.tex:189`).

**MAP** is mode-finding: run a gradient-based optimizer from several starting
points on the exact `_logpost` function this chapter built, and keep the
best local maximum found. It is cheap, and it is only a point — the
second-order Taylor picture of [Chapter 2](02-derivatives.md#taylor) applies
here directly, and so does its warning: a zero gradient is not evidence of a
mode. [Chapter 5](05-linear-algebra.md#definiteness-and-saddles) already
named the failure mode; [Chapter 26](26-the-saddle.md#the-map-is-a-saddle)
shows it happening on this exact posterior.

**SVI** (stochastic variational inference) fits a cheap Gaussian
approximation to the posterior around that mode — not by sampling, but by
optimizing the Gaussian's own mean and covariance to minimize a KL
divergence to the true posterior. It is not meant to be trusted as the
answer; it is meant to produce a covariance estimate fast, which becomes the
next step's preconditioner.

**HMC** then samples the real posterior, using gradients of `_logpost` and a
momentum "mass matrix" built from the SVI covariance to avoid the absurdly
small steps a naive sampler would need in the model's stiff directions
(recall $\mathrm{cond}\sim10^{14}$ above) or the wasted ones it would take in
its flat, ridge-tamed shapelet directions. [Chapter
23](23-samplers.md#hmc-and-the-metric) derives exactly what a mass matrix is
and why its direction convention is easy to get backwards.

Whether three steps in this order actually converge on this repo's own
46-dimensional, condition-$10^{14}$, occasionally-saddle-shaped posterior —
and what changes when they do not — is [Chapter 23](23-samplers.md#tempering-and-smc)
and [Chapter 26](26-the-saddle.md#why-smc-rescued-it) in full. This chapter's
job ends at the log-posterior the recipe is handed: 46 numbers wide, one
Occam term deep, and identical every time it is called.

!!! note "γ Ledger"
    **What this chapter rules in or out about $\gamma = 1.103$:** nothing
    directly — this chapter builds the machine, not the verdict. But every
    $\gamma$ number in [Chapter 25](25-money-number.md#the-chain)'s chain,
    money number included, is the output of exactly this 46-dimensional
    marginalized log-posterior, with this Occam term inside it, sampled by
    some version of the recipe above. Get the parameter count wrong, marginalize
    the wrong 28 numbers, or drop $-\tfrac12\log\det A$ the way `gigalens`'s
    stock solver does, and every $\gamma$ downstream is measuring a different
    quantity than the one this book has been careful to name.

## Connect to the repo { #connect }

- `reproductions/claude-giga-lens/cgl/likelihood.py:208-386` — the whole
  chapter, as running code: the prior and 46-dim bijector (`:294`'s
  `assert ndim == 46`), the two `PhysicalModel` instantiations (`:299-303`
  the deterministic render, `:311-315` the shapelet design), `_design_ret`'s
  render $\to$ convolve $\to$ pool pipeline (`:329-352`), and `_logpost`
  wiring `marg_loglik` into the full posterior (`:369-386`).
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/likelihood.py#L208)
- `reproductions/claude-giga-lens/cgl/marg.py` — the 55-line ridge core
  [Chapter 8](08-probability.md#ridge-is-a-prior) introduced; this chapter
  runs it at 28 real design columns instead of one toy amplitude.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/marg.py)
- `reproductions/claude-giga-lens/cgl/e2.py:11-34` — the campaign's own
  docstring stating the 74-to-46 reduction in one paragraph, plus the six
  measured basin-$\gamma$ seeds this exact model produced.
- `reproductions/claude-giga-lens/cgl/guards.py:74-91` — `assert_psf_sampling`,
  the guard that makes this chapter's render-order failure mode impossible to
  repeat silently.
- `reproductions/claude-giga-lens/papers/main.tex` — Table `tab:parity`
  (Gate E, the audited Occam term) and Table `tab:thresholds` (all five
  pre-registered gates, A through E).
- `reproductions/foundry-i/README.md:83-121` — the campaign's own
  retrospective on why marginalization was necessary: the `pinv`-based
  linear solve's non-smoothness, in the group's own words.

## Exercises { #exercises }

??? question "Exercise 22.1 — Recount the 74, and the 46"
    Without looking back at the running total, list this model's five
    component groups (mass, shear, lens light, source Sérsic, source
    shapelets) with a parameter count for each, sum them, and then say how
    many parameters remain once the shapelet amplitudes are marginalized
    away rather than sampled.

    ??? success "Solution"
        Mass (EPL): $\theta_{\mathrm{E}}, \gamma, e_1, e_2$, center $(x,y)$ —
        $6$. External shear: $\gamma_1, \gamma_2$ — $2$. Lens light: four
        Sérsic profiles at $7$ parameters each — $28$. Source Sérsic: $7$.
        Source shapelets: a $3$-parameter pose (scale, center) plus $28$
        amplitudes at $n_{\max}=6$. Summing: $6+2+28+7+3+28 = 74$
        <!-- check: ch22.n_params_full = 74 ± 0 -->.
        Marginalizing the $28$ amplitudes analytically removes them from the
        sampled space entirely (they are solved by a Cholesky step inside
        every log-posterior evaluation, not sampled by a Markov chain), so
        $74-28=46$
        <!-- check: ch22.n_params_marg = 46 ± 0 -->
        remain.

??? question "Exercise 22.2 — Why is $A$ only $28\times28$?"
    Equation $\eqref{eq:normaleq}$'s normal matrix $A = X_w^\top X_w +
    \Lambda$ is $28\times28$: the marginalized model has $46$ sampled
    parameters, yet $A$ never grows past the shapelet count. Where do the
    other $45$ parameters (everything except $\gamma$, say) actually show up
    in a single evaluation of the log-posterior?

    ??? success "Solution"
        $A$'s dimension is the number of *linear* parameters being
        marginalized at this evaluation, not the number of parameters the
        sampler explores overall. The $46$ nonlinear-plus-$I_e$ parameters
        never become rows or columns of $A$; instead, every one of them
        reshapes the *inputs* to $A$ each time the log-posterior is called.
        Changing $\theta_{\mathrm{E}}$ or $\gamma$ changes $\boldsymbol\beta$
        via the lens equation, which changes where each shapelet basis
        function lands on the sky, which changes every column of the design
        matrix $X_w$ — and therefore $A = X_w^\top X_w + \Lambda$ itself,
        recomputed from scratch at the new point. The $28\times28$ solve is
        exact and cheap; the $46$-dimensional exploration around it is what
        a sampler still has to do the hard way.

??? question "Exercise 22.3 — What the Occam term costs, in production"
    Using the audited production value $\log\det A = 323.229$
    <!-- check: ch08.occam_logdetA_parity = 323.229 ± 0.001 -->
    at the campaign's own parity point, compute how many nats `gigalens`'s
    plain `lstsq` path (no Occam correction) would over-report relative to
    the correct ridge-marginalized log-likelihood at that exact point.

    ??? success "Solution"
        Dropping $-\tfrac12\log\det A$ from Equation $\eqref{eq:occam}$
        changes the reported log-likelihood by exactly
        $+\tfrac12\log\det A = 0.5 \times 323.229 = 161.61$ nats
        <!-- check: ch22.occam_correction_production_nats = 161.6145 ± 0.001 -->.
        That is not a small correction relative to the evidence differences
        this campaign reports ([Chapter 25](25-money-number.md#the-evidence-flip)):
        it is on the same order as the swing that decides which basin the
        data prefer. A code path that always omits this term is not merely
        slightly optimistic — it is systematically biased toward whichever
        model has the *sharper* (larger-$\det A$) linear posterior, which is
        exactly backwards from what a genuine Bayesian comparison should
        reward.

??? question "Exercise 22.4 — Why PSF convolution comes before pooling"
    Equation $\eqref{eq:render}$ convolves with the PSF *before*
    average-pooling down to the data grid, on the oversampled grid, not
    after. Explain why doing it in the other order — pool first, then
    convolve at native resolution — would be wrong, and name the real repo
    failure this chapter connects it to.

    ??? success "Solution"
        A point-spread function has real structure — its core, its
        diffraction spikes — at angular scales finer than one data pixel;
        that is the entire reason images look blurred rather than merely
        pixelated. Pooling down to the native grid *before* convolving would
        throw away exactly the sub-pixel structure the PSF needs to act on,
        producing a systematically wrong (typically too-broad, since pooling
        is itself a smoothing operation) effective blur. The real-repo
        version of this mistake was not the order of operations in the code
        — that has always been convolve-then-pool — but a *sampling-scale*
        version of the same error: an empirical PSF kernel sampled at its
        own native pixel scale, fed into a convolution that assumed it was
        already sampled at the oversampled `delta_pix` scale, silently
        double-refined and broadened the effective PSF by a factor of two.
        Same noise, same data, same code otherwise — fixing only that
        sampling-convention mismatch dropped $\chi^2_\nu$ from $3.4$
        <!-- check: ch08.chi2_nu_psf_broadened = 3.4 ± 0.05 -->
        to $1.05$
        <!-- check: ch08.chi2_nu_psf_fixed = 1.05 ± 0.05 -->.
