# 25. Spine 1: the 191-nat flip and the verdict on 1.103

Chapter 1 handed you the destination and asked you to write down a guess
before you knew the argument. This chapter is the argument. It walks the full
chain from a drizzled *HST* exposure to $\gamma = 1.103 \pm 0.008$, link by
link, with a real repository artifact at every link; it derives the 191-nat
evidence swing that flips the binned bimodality verdict; and it puts the
report's own headline sigma claim — "$\sim17\sigma$ below the anchor," stated
in the abstract, in the `README.md` status line, and in a footnote that offers
its own arithmetic — through the same four numbers you already have, to see
whether it survives contact with a calculator. It does not. Neither of the
two numbers that do survive is close to 17, and one of them is the exact
statistic the campaign's own pre-registered gate defines. By the end of this
chapter the $\gamma$ Ledger closes, and you vote.

## The chain, link by link { #the-chain }

Every step below has a name, a file, and a number. None of it is new
machinery — it is [Ch. 11](11-observation.md#drizzle)'s drizzle correlation,
[Ch. 22](22-inference.md#marginalising-linear-amplitudes)'s marginalization,
[Ch. 23](23-samplers.md#tempering-and-smc)'s tempered SMC, and
[Ch. 24](24-correlated-noise.md#the-covariance-model)'s whitened likelihood,
run once, on one real system, twice — with a diagonal likelihood and with a
correlated one — so that the difference between the two runs *is* the
result.

| # | Link | The number it produces | Derived at |
|---|---|---|---|
| 1 | *HST* drizzles DESI−165.4754−06.0423 to three pixel scales | lag-1 correlation $\rho(1) = 0.815$ fine, $0.615$ binned, $0.305$ native | [Ch. 11](11-observation.md#why-drizzle-correlates-noise) |
| 2 | A diagonal likelihood assumes independent pixels anyway | posterior widths $\sim 7.7\times$ too tight on the fine product | [Ch. 11](11-observation.md#why-drizzle-correlates-noise) |
| 3 | Build $C = D^{1/2}KD^{1/2}$, whiten it, gate the whitener | operator-norm gate $e_{\mathrm{op}} \le 0.02$ (one whitener only under a pre-registered $\le0.05$ relaxation), all four whiteners pass | [Ch. 24](24-correlated-noise.md#convolutional-whitening) |
| 4 | Marginalize the 28 linear shapelet amplitudes analytically | the $-\tfrac12\log\det A$ Occam term | [Ch. 22](22-inference.md#the-occam-term) |
| 5 | Fit both basins of the binned product under each likelihood | steep $\gamma =$ <!-- check: ch25.gamma_diag_steep = 2.423 ± 0.027 --> $2.423$, low $\gamma =$ <!-- check: ch25.gamma_diag_low = 1.293 ± 0.012 --> $1.293$ (diagonal) | below, [The evidence flip](#the-evidence-flip) |
| 6 | Weigh the two basins by evidence, not by chain count | diagonal favours steep by $+162.2$ nats | below, [The evidence flip](#the-evidence-flip) |
| 7 | Repeat under the correlated likelihood | correlated favours low by $-28.9$ nats — a $191.1$-nat swing | below, [The evidence flip](#the-evidence-flip) |
| 8 | The correlated binned-low MAP turns out not to be a mode | Laplace Hessian minimum eigenvalue $-14.85$, five negative directions | [Ch. 26](26-the-saddle.md#the-map-is-a-saddle) |
| 9 | Sample it with tempered SMC instead of HMC | $128$ particles, $\lambda \to 1$ in $28$ steps | below, [The money number](#the-money-number) |
| 10 | Read off the converged slope | $\gamma = 1.103 \pm 0.008$ | below, [The money number](#the-money-number) |

Steps 1–4 are Chapters 11, 22, and 24's job, and this chapter takes them as
given: a validated, whitened, marginalized correlated-noise likelihood that
reduces *exactly* to the diagonal one when the covariance actually is
diagonal (`cgl/e2.py`'s regression test for that identity is
[Ch. 24](24-correlated-noise.md#the-diagonal-limit)'s closing gate). What this
chapter owns is steps 5 through 10: what happens when you point that
machinery at a real, disconnected, bimodal posterior and ask it which basin
is *actually* more probable, and what value of $\gamma$ that basin actually
contains.

!!! tip "You already know this"
    Step 5's basin fit alone is not the answer to step 6's question, and the
    gap between them is a mistake you have almost certainly already made with
    a badly-mixed sampler or a poorly-seeded ensemble. Counting how many of 48
    parallel chains *ended up* in each basin is counting local optima a
    multi-start optimizer happened to land near — it says something about
    where you initialized, not about the relative probability mass of each
    basin. The only way to answer "how much mass is actually here" is to
    integrate, which is exactly what an evidence estimate does and a chain
    occupancy count does not.

## The evidence flip { #the-evidence-flip }

On the binned ($0.08''$) product, both basins are real fits: a steep basin
near $\gamma \approx 2.4$ and a low basin near $\gamma \approx 1.3$, with
zero chains ever crossing between them in the stored HMC runs. Under the
*diagonal* likelihood, $45$ of $48$ chains land in the low basin and $3$ in
the steep one
<!-- check: ch25.n_chains_low = 45 ± 0.5 -->
<!-- check: ch25.n_chains_steep = 3 ± 0.5 -->
<!-- check: ch25.n_chains_total = 48 ± 0.5 -->
— a naive low-basin occupancy of
$93.75\%$ <!-- check: ch25.naive_w_low_occupancy = 0.9375 ± 0.0001 -->,
with exactly
$0$ <!-- check: ch25.n_inter_basin_migrations = 0 ± 0.5 -->
migrations between them. Read at face value, that occupancy says the low
basin dominates.

It does not. Per-basin tempered SMC — an independent evidence estimate for
*each* basin, not a chain count — gives $\log Z_{\mathrm{steep}} - \log Z_{\mathrm{low}} = +162.2$ nats <!-- check: ch25.dlogz_diagonal = 162.2 ± 0.1 --> under
the *same* diagonal likelihood the occupancy count above came from. The
steep basin, which held only $3$ of $48$ chains, carries essentially *all*
of the posterior mass. The $93.75\%$ occupancy split was a start artifact:
each chain reports where it began, and because none of them mix across the
gap between basins, occupancy measures the multi-start initializer, not the
posterior. This is [Ch. 8](08-probability.md#evidence-and-nats)'s evidence,
concretely: a chain count is a histogram of starting points; $\log Z$ is the
only one of the two that actually integrates the density.

Apply the *correlated* likelihood to the identical binned target and the
sign of that comparison reverses: $\log Z_{\mathrm{steep}} - \log Z_{\mathrm{low}} =
-28.9$ nats <!-- check: ch25.dlogz_correlated = -28.9 ± 0.1 -->. The low
basin now dominates. The total swing in the basin-evidence gap, steep-minus-low
under each likelihood, is

$$
\bigl|\, (+162.2) - (-28.9) \,\bigr| = 191.1 \text{ nats}
$$

<!-- check: ch25.evidence_swing_nats = 191.1 ± 0.05 -->

a Bayes factor of $e^{191.1}$, or in a base you can say out loud,
$10^{82.99}$ <!-- check: ch25.bayes_factor_log10 = 82.99 ± 0.01 -->
($191.1/\ln 10$). Jeffreys' scale calls anything past $5$ nats "decisive"; this
is $38$ times that bar. Only the *within*-likelihood difference is meaningful
here — the two likelihoods carry different, fixed whitening log-determinant
constants folded into their respective absolute $\log Z$'s (the Szegő-vs-exact
constant this campaign tracks separately;
[Ch. 24](24-correlated-noise.md#the-diagonal-limit)), so it is $\Delta\log Z$
within each likelihood, not $\log Z$ across them, that is comparable. That
within-likelihood swing is what the figure below plots.

<figure markdown="span">
  ![The 191-nat evidence flip: diagonal favours steep by 162.2 nats, correlated favours low by 28.9 nats](figures/ch25-basin-flip-light.svg#only-light){ width="90%" }
  ![The 191-nat evidence flip: diagonal favours steep by 162.2 nats, correlated favours low by 28.9 nats](figures/ch25-basin-flip-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 25.1.** Per-basin SMC evidence,
  $\Delta\log Z_{\mathrm{steep-low}}$, on the binned product under each likelihood.
  Diagonal: $+162.2$ nats, decisively favouring steep. Correlated: $-28.9$
  nats, decisively favouring low. The $191.1$-nat swing is a Bayes factor of
  roughly $10^{83}$ in the opposite direction. Source: `24_basin_evidence_v3b.py`
  and `CAMPAIGN.md` (P1c MONEY NUMBER, 2026-07-14).</figcaption>
</figure>

The pre-registered gate for this hypothesis (`README.md:95`) reads: the
steep basin's posterior mass must fall under $10\%$, *or* the corrected
log-likelihood gap between basins at their MAPs must be $\le 0$, for the
diagonal likelihood's apparent bimodality to be called a likelihood
artifact. $-28.9 \le 0$ passes that gate outright.
**Hypothesis 1 — bimodality is a noise-covariance artifact — is confirmed.**
The steep basin was never a second physical solution; it was what a diagonal
likelihood sees when it is handed noise that is not actually diagonal, on the
upsampled product where that mismatch is worst.

## The money number { #the-money-number }

Confirming H1 tells you which basin to trust. It does not yet tell you what
$\gamma$ is inside that basin — and extracting that value turned out to be
the harder half of this campaign, for a reason
[Ch. 26](26-the-saddle.md#the-map-is-a-saddle) tells in full. The short
version: the correlated-likelihood binned-low posterior's
MAP is not a mode. Its Laplace Hessian has minimum eigenvalue
$-14.85$ <!-- check: ch25.saddle_min_eigenvalue = -14.85 ± 0.01 --> with
$5$ <!-- check: ch25.n_negative_eigenvalues = 5 ± 0.5 --> negative directions
— a saddle, not a hilltop. Two different positive-definite metric repairs
(a diagonal Hessian metric and a full-rank SVI covariance), both built from
exactly the two-stage preconditioned-HMC recipe that calibrates cleanly on
mock data, both fail on this basin, but not identically: the diagonal-metric
repair never finishes on the binned-low target at all — it times out, stuck
(`CAMPAIGN.md`, "P1c metric-fix attempts," 2026-07-10) — while the
SVI-covariance repair does run to completion and only gets worse, its
$\hat{R}$ climbing to
$22.3$ <!-- check: ch25.rhat_saddle_metric = 22.3 ± 0.01 -->. The MAP itself
sits at $\gamma_{\mathrm{map}} = 1.27$ <!-- check: ch25.gamma_map_saddle = 1.27 ± 0.001 -->,
while the actual highest-density point in the chains that do move is nearer
$\gamma_{\mathrm{best}} \approx 1.10$ <!-- check: ch25.gamma_best_density_point = 1.10 ± 0.005 -->
— a polished optimum and the true peak of the posterior disagreeing about
where $\gamma$ even is, the unmistakable signature of an indefinite Hessian.
HMC, of any flavor, needs a momentum metric built from *some* notion of local
curvature; there is none to build here. What worked instead is tempered SMC
(same idea as [Ch. 23](23-samplers.md#tempering-and-smc), applied per-basin):
$128$ particles annealed from prior to posterior in $28$ steps
<!-- check: ch25.n_smc_particles = 128 ± 0.5 -->
<!-- check: ch25.n_smc_lambda_steps = 28 ± 0.5 -->,
with an effective sample size between
$77$ <!-- check: ch25.ess_smc_low = 77 ± 0.5 --> and
$118$ <!-- check: ch25.ess_smc_high = 118 ± 0.5 --> particles at the end of
the run — resampling, not local gradients, is what crosses a landscape a
single momentum metric cannot describe.

That run converges to the money number:

$$
\gamma_{\mathrm{binned}}(\text{corr, low}) = 1.103 \pm 0.008 \qquad
\theta_{\mathrm{E}} = 2.624 \pm 0.005
$$

<!-- check: ch25.gamma_money = 1.103 ± 0.008 -->
<!-- check: ch25.theta_e_money = 2.624 ± 0.005 -->

Set beside the diagonal-native anchor — the pre-registered reference value,
chosen because the native ($0.128''$) product is the least resampled and
therefore where a *diagonal* likelihood is closest to correct —

$$
\gamma_{\mathrm{anchor}} = 1.433 \pm 0.034
$$

<!-- check: ch25.gamma_anchor = 1.433 ± 0.034 -->

the correlated likelihood has clearly *moved* the slope, and in the
predicted direction: away from the diagonal likelihood's upsampling
artifacts and toward the anchor. But it does not land on it. Two more
points complete the picture. On the fine ($0.04''$) product, the
correlated likelihood deflates the diagonal's outright artifact
$\gamma = 2.585$ <!-- check: ch25.gamma_fine_diag_artifact = 2.585 ± 0.001 -->
(a MAP-only number — the fine product's chains never converged under the
diagonal likelihood at all) down to
$\gamma = 1.816 \pm 0.117$ <!-- check: ch25.gamma_fine_steep = 1.816 ± 0.117 -->,
still $3.1\sigma$ *above* the anchor (derived properly in
[The sigma arithmetic](#the-sigma-arithmetic) below). On the native product,
the correlated likelihood — starved of pixels by the *relaxed* native
whitener, which keeps only $1{,}466$ of $5{,}865$ pixels, a $75\%$ loss
([Ch. 24](24-correlated-noise.md#the-covariance-model)) — drifts to
$\gamma = 2.353 \pm 0.096$ <!-- check: ch25.gamma_native_corr = 2.353 ± 0.096 -->,
prior-pulled toward the slope prior's mean of $2.0$
<!-- check: ch25.gamma_prior_mean = 2.0 ± 0.001 -->
rather than data-driven; it is excluded from the decision for exactly that
reason (the pre-registered amendment that made diagonal-native, not
correlated-native, the anchor). And the fine product's *low* basin does not
appear in the bracket at all: two independently-seeded, physically sane
starting points both rail toward $\gamma \approx 1.0$ — the slope prior's
hard floor,
<!-- check: ch25.gamma_prior_low_wall = 1.0 ± 0.001 -->
one of the bounds of its `TruncatedNormal(2.0, 0.25, low=1.0, high=2.7)`
support (`cgl/e2.py:110`) — while the *real-space* image the whitened
likelihood is fitting turns to visible garbage. The fine whitener can be
gamed at low $\gamma$; the fine low basin is a characterized limitation, not
a fourth data point.

<figure markdown="span">
  ![Nine gamma values for one galaxy: the correlated likelihood brackets the diagonal-native anchor but does not unify onto it](figures/ch25-gamma-bracket-light.svg#only-light){ width="90%" }
  ![Nine gamma values for one galaxy: the correlated likelihood brackets the diagonal-native anchor but does not unify onto it](figures/ch25-gamma-bracket-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 25.2.** Every $\gamma$ this campaign
  measured on DESI−165.4754−06.0423, one real galaxy, nine numbers. The
  green line is the diagonal-native anchor, $1.433\,[1.400,1.469]$. The
  correlated likelihood (fine-steep $1.816$, binned-low $1.103$ — the money
  number, native $2.353$) moves every diagonal slope toward the anchor but
  brackets it from both sides rather than converging onto it. Source:
  `reproductions/claude-giga-lens/papers/main.tex` Table `tab:crossscale`;
  `worked_values`: anchor $1.433$, money $1.103$.</figcaption>
</figure>

## The sigma arithmetic { #the-sigma-arithmetic }

You now have every number the report's headline claim depends on:
$\gamma_{\mathrm{money}} = 1.103 \pm 0.008$ and $\gamma_{\mathrm{anchor}} = 1.433 \pm
0.034$. The campaign's own pre-registered gate for cross-scale unification,
frozen at P0 before any real-data run (`README.md:99`), is written down as
a formula:

$$
|\Delta\gamma_{\mathrm{EPL}}| < 2\,\sigma_{\mathrm{comb}}, \qquad
\sigma_{\mathrm{comb}} = \sqrt{\sigma_{\gamma,\,{\mathrm{one\ scale}}}^2 +
\sigma_{\gamma,\,{\mathrm{anchor}}}^2}
\label{eq:sigmacomb}
$$

That is one specific, pre-registered definition of "how many sigma," not a
free choice — and it is the same convention the report itself uses, in the
same paragraph as the money number, for the *other* bracket point: applied
to the fine-steep slope $1.816 \pm 0.117$ against the anchor, it gives

$$
\frac{|1.816 - 1.433|}{\sqrt{0.117^2 + 0.034^2}} \approx 3.14\sigma
$$

<!-- check: ch25.sigma_fine_steep_vs_anchor = 3.14 ± 0.01 -->

— matching the report's own quoted "$3.1\sigma$ high" (`main.tex:772`,
`:828`) to the rounding. Apply $\eqref{eq:sigmacomb}$, the pre-registered
formula, *consistently*, to the money number itself:

$$
\frac{|1.103 - 1.433|}{\sqrt{0.008^2 + 0.034^2}}
= \frac{0.330}{0.0349}
$$

<!-- check: ch25.sigma_vs_anchor_combined = 9.45 ± 0.01 -->

That comes to $9.45\sigma$. Not $17$. Two other readings of "how many
sigma" are at least defensible, and neither reaches $17$ either. Divide by
the anchor's own uncertainty alone, ignoring the money number's much
smaller error bar entirely:

$$
\frac{0.330}{0.034} \approx 9.71\sigma
$$

<!-- check: ch25.sigma_vs_anchor_own_error = 9.71 ± 0.01 -->

— close to the report's own footnote, which states "even against the
anchor's own $\sigma_\gamma \approx 0.034$ the discrepancy is
$\sim 9.5\sigma$" (`main.tex:763`). Or divide by the money number's error
bar alone, treating the anchor as if it carried no uncertainty at all —
indefensible statistically, since the anchor plainly has an error bar too,
but it is the only arithmetic that gets anywhere near double digits:

$$
\frac{0.330}{0.008} \approx 41.25\sigma
$$

<!-- check: ch25.sigma_vs_money_error_only = 41.25 ± 0.05 -->

Three ways to divide the same difference by some uncertainty: $9.71$,
$9.45$, $41.25$. The report's own footnote (`main.tex:763`) computes one of
these — $\approx 9.5\sigma$ — and immediately beside it, in the same
sentence, states "The $\sim17\sigma$ figure is the ledger value
(`CAMPAIGN.md`, P1c MONEY 2026-07-14), combining the SMC width $0.008$ with
the anchor uncertainty," which is a description of *exactly* the
$\sigma_{\mathrm{comb}}$ computation above — the one that gives $9.45$, not $17$.
`CAMPAIGN.md:138` states the $17\sigma$ figure directly, undated arithmetic
attached; the number is real (it appears in the abstract, the `README.md`
status line, and the pre-registered-threshold table's own status column,
`main.tex:1476`), but nowhere in the source does the division that would
produce it from the two numbers this chapter — and the report itself, in the
very same paragraph — already has.

Hold two things at once here, because both are true. First: the discrepancy
between the money number and the anchor is real and large by any of the
three defensible measures — even the loosest, $9.45\sigma$, clears the
pre-registered $2\sigma_{\mathrm{comb}}$ gate by more than four times over, so
**Hypothesis 2 fails decisively regardless of which sigma you pick.** The
exact multiple does not change the verdict. Second: a guide whose entire
premise is that this repository's numbers reproduce cannot let a
$\sim17\sigma$ claim stand merely because the conclusion it supports happens
to be correct on other grounds. This is precisely the discipline
[Ch. 1](01-orientation.md#how-to-read)'s closing paragraph promises: run the
division yourself, and a script — not a better adjective — is what catches
the gap.

## The verdict { #the-verdict }

Three hypotheses, three verdicts, all traced above:

- **H1 (bimodality), confirmed.** The steep basin's diagonal dominance was a
  noise-covariance artifact of the upsampled product, exposed by a
  191.1-nat evidence swing and a pre-registered gate ($\Delta\log\ell \le 0$)
  that the correlated result clears outright.
- **H2 (cross-scale unification), rejected.** The correlated likelihood
  moves every diagonal slope substantially toward the anchor — but
  brackets it, fine-steep $3.14\sigma$ above, binned-low $9.45\sigma$
  below, rather than converging onto it. The residual scale-dependence
  survives every honest sigma convention.
- **H3 (honesty), passed.** The fine posterior's width, $\sigma_\gamma =
  0.117$, is not artificially narrow: it exceeds the pre-registered floor of
  $\sigma_\gamma({\mathrm{native}})/1.5 \approx 0.023$ by a wide margin
  (`README.md:101`). The correlated likelihood did not manufacture
  confidence it had not earned.

So: is $\gamma = 1.103 \pm 0.008$ a trustworthy measurement of this galaxy's
density slope? The honest answer this chapter earns is *precise, not yet
accurate*. The number is real in the sense that matters most to a
statistician — it is the converged output of a validated likelihood
(exact to the diagonal case, exact to a dense-covariance Cholesky reference,
calibrated on drizzle mocks with known truth) and a sampler that actually
crossed the saddle rather than freezing on it. It is not yet trustworthy as
*the* slope of this galaxy, because the same campaign that produced it also
shows, with its own numbers, that changing nothing but the pixel scale of
the same exposures moves the answer by multiples of its quoted uncertainty
in both directions. "Necessary but not sufficient" is the report's own
phrase for this, and it is the right one: fixing the noise model was a real,
large, validated correction — and it was not, on its own, enough to make
three measurements of one number agree. What remains unaccounted for —
most plausibly the shapelet source model and the stationary-kernel PSF,
the two ingredients this campaign's likelihood fix left untouched — is
exactly the kind of systematic the code-comparison literature this report
cites (`main.tex:178`) already warns rivals the statistical error bars on
individual systems.

Go back to the sentence you wrote in
[Exercise 1.1](01-orientation.md#exercises). Whichever way you guessed, the
chain above is the argument for it, in full, with every number checked. That
is the only grading this exercise ever had.

!!! note "γ Ledger — closed"
    **What this chapter rules in or out about $\gamma = 1.103 \pm 0.008$:**
    it is the correctly-sampled mode of a validated, whitened likelihood's
    low basin on the binned product — not a sampling artifact, not a
    stuck-chain illusion, not prior-pulled (it sits $12.9$ of its *own*
    sigmas above the prior's hard floor at $1.0$
    <!-- check: ch25.distance_above_wall_in_own_sigma = 12.87 ± 0.05 -->,
    and $3.59$ prior-sigmas *below* the prior's mean of $2.0$
    <!-- check: ch25.prior_sigmas_below_prior_mean = 3.588 ± 0.01 -->, so the
    data pulled it down, the prior did not push it there). What it is not:
    unified with the same campaign's own native-diagonal anchor, at
    $9.45\sigma$ by the campaign's own pre-registered metric — and that gap
    is real physics left over (source model, PSF), not sampling error. The
    Ledger closes here; Chapter 29 collects it alongside Ch. 26's saddle for
    the guide's final synthesis.

## Connect to the repo { #connect }

- `reproductions/claude-giga-lens/papers/main.tex:746`–`799` (Table
  `tab:crossscale`) — the complete cross-scale bracket table this chapter's
  Figure 25.2 plots, and `:840`–`895` (§Bimodality verdict) — the 191-nat
  flip derivation and Table `tab:basinflip`.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/papers/main.tex#L746)
- `reproductions/claude-giga-lens/papers/main.tex:897`–`924` (§Why sampling
  this posterior required tempered SMC) — the saddle diagnosis this chapter
  only summarizes; `main.tex:1476` — the pre-registered threshold table's own
  H2 row, `FAIL (over-corr.; 1.103, ~17σ)`; `:763` — the footnote with the
  report's own, incompletely shown, sigma arithmetic.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/papers/main.tex#L897)
- `reproductions/claude-giga-lens/README.md:95`–`101` — the three
  pre-registered hypotheses (H1/H2/H3), verbatim, frozen before any real-data
  run; `:30`–`33` — the same $\sim17\sigma$ claim in the project's own status
  summary, not only in the paper.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/README.md#L95)
- `reproductions/claude-giga-lens/CAMPAIGN.md:133`–`163` ("P1c MONEY NUMBER
  — 2026-07-14") — the dated gate record every number in this chapter traces
  to, including the basin-evidence linchpin and the un-derived $17\sigma$
  line this chapter's [sigma arithmetic](#the-sigma-arithmetic) audits.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/CAMPAIGN.md#L133)
- `cgl/e2.py:110` — the slope prior, `TruncatedNormal(2.0, 0.25, low=1.0,
  high=2.7)`, whose floor and mean anchor the closing γ Ledger box.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L110)
- `24_basin_evidence_v3b.py:13`–`24` — the per-basin SMC evidence estimator
  (`logZ_k = log integral_{basin k} ...`), run once per likelihood to
  produce Figure 25.1.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/24_basin_evidence_v3b.py#L13)
- `site/guide_src/worked_examples.py` (`ch25_money_number`) — every number
  tagged in this chapter, computed or pinned to its cited artifact, one
  function, runnable with `--show ch25`.

## Exercises { #exercises }

??? question "Exercise 25.1 — check the H1 gate yourself"
    The pre-registered H1 gate (`README.md:95`) passes if the steep basin's
    posterior mass is under $10\%$ under the correlated likelihood, *or* if
    the corrected log-likelihood gap $\Delta\log\ell(\text{steep} -
    \text{low}) \le 0$ at the basin MAPs. You have $\log Z_{\mathrm{steep}} -
    \log Z_{\mathrm{low}} = -28.9$ nats under the correlated likelihood. Which
    branch of the gate does this satisfy, and does it need the first branch
    at all?

    ??? success "Solution"
        $\Delta\log Z(\text{steep}-\text{low}) = -28.9 \le 0$ satisfies the
        second branch outright — the steep basin's evidence is lower than the
        low basin's by $28.9$ nats, an enormous margin past a bare $\le 0$.
        You do not need the mass-fraction branch at all; the log-evidence gap
        alone clears the gate by any reasonable margin. (For reference,
        converting the gap to a mass fraction via the two-outcome softmax
        $w_{\mathrm{steep}} = 1/(1+e^{-\Delta\log Z})$ gives a steep-basin weight
        of order $e^{-28.9}$, comfortably under the $10\%$ first-branch bar
        too — both branches agree, which is what "decisive" is supposed to
        look like.)

??? question "Exercise 25.2 — the sigma arithmetic, from scratch"
    Using only $\gamma_{\mathrm{money}} = 1.103 \pm 0.008$ and $\gamma_{\mathrm{anchor}} = 1.433 \pm 0.034$, compute the discrepancy three ways: (a)
    divided by the anchor's uncertainty alone, (b) divided by the two
    uncertainties combined in quadrature (Eq. $\eqref{eq:sigmacomb}$), (c)
    divided by the money number's uncertainty alone. Which of the three is
    the campaign's own pre-registered test statistic?

    ??? success "Solution"
        The difference is $|1.433 - 1.103| = 0.330$.
        (a) $0.330 / 0.034 \approx 9.71\sigma$
        <!-- check: ch25.sigma_vs_anchor_own_error = 9.71 ± 0.01 -->.
        (b) $0.330 / \sqrt{0.034^2 + 0.008^2} \approx 9.45\sigma$
        <!-- check: ch25.sigma_vs_anchor_combined = 9.45 ± 0.01 -->.
        (c) $0.330 / 0.008 \approx 41.25\sigma$
        <!-- check: ch25.sigma_vs_money_error_only = 41.25 ± 0.05 -->.
        (b) is the pre-registered statistic — `README.md:99` defines the H2
        gate as $|\Delta\gamma| < 2\sigma_{\mathrm{comb}}$ with $\sigma_{\mathrm{comb}}$
        the quadrature sum, and it is the convention the report itself uses
        for the fine-steep row. None of the three is anywhere near $17$.

??? question "Exercise 25.3 — apply the report's own convention consistently"
    The report states the fine-steep slope, $\gamma = 1.816 \pm 0.117$, is
    "$3.1\sigma$ high" relative to the anchor. Reproduce that number using
    Eq. $\eqref{eq:sigmacomb}$, then explain why finding the *same*
    convention gives $9.45\sigma$ for the money number is more damaging to
    the report's $17\sigma$ claim than merely noting the two numbers
    disagree.

    ??? success "Solution"
        $|1.816 - 1.433| / \sqrt{0.117^2 + 0.034^2} = 0.383/0.1218 \approx
        3.14\sigma$ <!-- check: ch25.sigma_fine_steep_vs_anchor = 3.14 ± 0.01 -->
        — matching the quoted $3.1\sigma$. This is more damaging than a bare
        mismatch because it rules out the charitable reading that "$17\sigma$"
        and "$9.5\sigma$" are two different, individually reasonable
        conventions the report used in different places: the report uses
        $\sigma_{\mathrm{comb}}$ correctly, in the very same section, for a
        neighboring number. The convention that produces $17$ is not merely
        under-explained; applied to the numbers actually on the page, it is
        not reproducible by any of the report's own stated methods.

??? question "Exercise 25.4 — necessary but not sufficient, argued both ways"
    The correlated likelihood's slopes bracket the anchor — fine-steep above,
    binned-low below — rather than converging on it. Argue for two different
    readings of that bracket: (a) it means the true slope is near the anchor,
    $1.433$, and the correlated likelihood over- and under-corrects at
    different scales for reasons that will wash out with a better source
    model; (b) it means the anchor itself should not be trusted as a
    ground truth, and the bracket is telling you the true answer is
    scale-dependent for a physical reason. What evidence in this chapter
    favors one reading over the other, and what would settle it?

    ??? success "Solution"
        Reading (a) is favored by the sign structure: fine-steep sits close
        to the un-lensed prior mean's neighborhood only in the sense that it
        is the *least* corrected (smallest pixel count relative to
        resolution), and native-correlated is explicitly excluded as
        prior-pulled information starvation, not a genuine value — both
        symptoms of *incomplete* correction rather than of a real
        scale-dependent physical slope, which nothing in the lens equation
        predicts (the density profile does not know what pixel scale it was
        photographed at). Reading (b) would require an actual physical
        mechanism coupling the recovered slope to the resampling scale, and
        the report proposes none. What would settle it, per the report's own
        discussion (`main.tex:965`–`970`) and this campaign's own honest
        limitations list, is exactly what H2's rejection points at: a
        non-shapelet source model and a non-stationary PSF kernel, tested
        against the same three products, to see whether the bracket narrows.
        Until that run exists, "necessary but not sufficient" is the most
        the data support — not proof that $1.433$ is right, only that
        $1.103$ alone is not yet enough to say it is wrong.
