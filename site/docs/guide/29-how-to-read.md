# 29. What you know now, and where the frontier is

Twenty-eight chapters ago this book named its destination: a single galaxy's
density-profile slope, $\gamma = 1.103 \pm 0.008$
<!-- check: ch25.gamma_money = 1.103 ± 0.008 -->, and a warning that the
report producing it quotes a "$\sim17\sigma$" tension for that number against
its own anchor — a claim that reconciles with none of the uncertainties the
same report states two paragraphs later. You now own the calculus, the linear
algebra, the cosmology, and the lensing well enough to check that claim
yourself, in one division, and you have read this campaign's own record
closely enough to know it retracts its own numbers in public. This chapter
adds no new mathematics. It closes the two ledgers this book has been
keeping since Chapter 4, hands you a map from the campaign's 27-page report
to the chapters that decode each of its sections, distills five questions
worth asking at any lensing meeting — each one has already cost this
repository a wrong number at least once — and says plainly where the
frontier sits, including the one corner of it that belongs to a computer
scientist rather than an astronomer. It ends with an honest list of what this
book chose not to teach.

## The two ledgers, closed { #the-ledgers-closed }

**The Log-Det Ledger, final.** Chapter 4 opened it with two entries that
looked unrelated — the lensing magnification $\mu = 1/|\det A|$ and a
normalizing flow's density correction $\log|\det J_T|$ — Chapter 8 added a
third, stranger-looking one, and Chapter 23 closed it: one Gaussian-integral
substitution shows all three are the same change-of-variables theorem, applied
to a linear lensing map, a nonlinear flow, and a linear whitening map
implicit in an evidence integral, respectively. This chapter does not repeat
that derivation; it collects the closed state:

!!! note "Log-Det Ledger — final state"
    | # | Costume | Formula | Instance |
    |---|---|---|---|
    | 1 | Lensing magnification | $\mu = 1/\lvert\det A\rvert$ | toy $\det A = 0.485$, $\mu = 2.0619$ <!-- check: ch04.det_A = 0.485 ± 1e-9 --><!-- check: ch04.mu = 2.0619 ± 1e-4 --> ([Ch. 4](04-multivariable.md#det-j-as-area-scaling)) |
    | 2 | Normalizing-flow pullback density | $\log p_U(\mathbf u) = \log p_X(T(\mathbf u)) + \log\lvert\det J_T(\mathbf u)\rvert$ | `cgl/flows.py:20` |
    | 3 | Gaussian-evidence Occam term | $-\tfrac12\log\det A$ | production $\log\det A = 323.229$ nats <!-- check: ch08.occam_logdetA_parity = 323.229 ± 0.001 --> ([Ch. 22](22-inference.md#the-occam-term)) |

    Derived in [Ch. 23](23-samplers.md#closing-the-log-det-ledger): "no fourth
    costume is left to find." Every $\log|\det(\,\cdot\,)|$ this book computes
    is one of these three, audited from a $2\times2$ toy to a
    $46$-dimensional real fit.

**The $\gamma$ Ledger, final.** [Chapter 25](25-money-number.md#the-verdict)
already closed this one, with the full chain and the full arithmetic; this
collects it alongside [Chapter 26](26-the-saddle.md#the-map-is-a-saddle)'s
contribution into one entry. Per-basin evidence on the binned product flips
$191.1$ nats <!-- check: ch25.evidence_swing_nats = 191.1 ± 0.05 -->, from
$+162.2$ favoring steep under the diagonal likelihood
<!-- check: ch25.dlogz_diagonal = 162.2 ± 0.1 --> to $-28.9$ favoring low
under the correlated one <!-- check: ch25.dlogz_correlated = -28.9 ± 0.1 -->
— H1 (bimodality-as-artifact) confirmed. But the restored low basin,
$\gamma = 1.103\pm0.008$, sits $9$–$41\sigma$ from the diagonal-native anchor
$1.433\pm0.034$ <!-- check: ch25.gamma_anchor = 1.433 ± 0.034 --> depending on
which legitimate uncertainty convention you apply
<!-- check: ch25.sigma_vs_anchor_own_error = 9.71 ± 0.01 -->
<!-- check: ch25.sigma_vs_anchor_combined = 9.45 ± 0.01 -->
<!-- check: ch25.sigma_vs_money_error_only = 41.25 ± 0.01 --> — never the
$17\sigma$ the report's own abstract claims — and the correlated fine
($1.816$ <!-- check: ch25.gamma_fine_steep = 1.816 ± 0.117 -->),
binned-low ($1.103$), and native ($2.353$
<!-- check: ch25.gamma_native_corr = 2.353 ± 0.096 -->) slopes *bracket* the
anchor rather than converging on it: H2 (cross-scale unification) rejected.
None of this came from a clean sampler run: the correlated-likelihood MAP is
a saddle, minimum eigenvalue $-14.85$
<!-- check: ch25.saddle_min_eigenvalue = -14.85 ± 0.01 -->, and the point a
saddle-seeded optimizer settles on scores $74$ nats *lower* than the true
peak <!-- check: ch05.saddle_logp_gain = 74 ± 1 -->. Chapter 25 derives the
number; Chapter 26 derives why extracting it required tempered SMC and not
HMC at all — two chapters, one instrument, one reading.

!!! note "γ Ledger — final entry"
    **What this book rules in or out about $\gamma = 1.103$:** a real,
    audited correction (H1) restores the low basin honestly, but does not (H2)
    unify this campaign's cross-scale measurements onto one consensus slope.
    Treat $1.103$ as one defensible edge of a bracket, not a converged value
    for this galaxy's density slope. The residual disagreement is evidence of
    a real limitation in the source model and PSF treatment, not a bug in the
    noise model — and that limitation, not this number, is where the
    campaign's next move belongs (see [The frontier](#the-frontier)).

## The paper decoder { #the-paper-decoder }

`main.tex` numbers every section with a `\label{sec:...}`, and the generated
report page preserves those as literal HTML anchors — which is why this
guide's own cross-link rule insists on linking `#sec:*`, never a slugified
heading: reword any section title and the anchor still holds. Here is the map
from that report, and from LensJudge's parity report, to this book.

| Report section | What is there | This guide |
|---|---|---|
| [§1 Introduction](../current/claude-giga-lens/index.md#sec:intro) | The campaign in one paragraph: the money number, and its own $\sim17\sigma$ claim | [Ch. 1](01-orientation.md#the-money-number) |
| [§2 Data and Targets](../current/claude-giga-lens/index.md#sec:data) | The three drizzle products (fine/binned/native) and the mock generator | [Ch. 11](11-observation.md#drizzle) |
| [§3 Methods I](../current/claude-giga-lens/index.md#sec:methods1), [whitening](../current/claude-giga-lens/index.md#sec:whiten), [marginalization](../current/claude-giga-lens/index.md#sec:marg) | $C=D^{1/2}KD^{1/2}$, the convolutional whitener, the ridge-marginalized Occam term | [Ch. 24](24-correlated-noise.md#the-covariance-model), [Ch. 22](22-inference.md#the-occam-term) |
| [§4 Methods II](../current/claude-giga-lens/index.md#sec:methods2) | The nine-sampler "posterior zoo" and its budget-matched protocol | [Ch. 23](23-samplers.md#tempering-and-smc) — partial, see [What we skipped](#what-we-skipped) |
| [§5 Mock Validation](../current/claude-giga-lens/index.md#sec:mocks), [two-stage PHMC](../current/claude-giga-lens/index.md#sec:twostage) | Calibration against known truth; the fix that lowered $\hat R$ from $2.11$<!-- check: ch23.t2_rhat_twostage_before = 2.11 ± 0.01 --> to $1.003$<!-- check: ch23.t2_rhat_twostage_after = 1.003 ± 0.001 --> | [Ch. 23](23-samplers.md#hmc-and-the-metric) |
| [§6 Real-Data Verdict](../current/claude-giga-lens/index.md#sec:realdata), [the flip](../current/claude-giga-lens/index.md#sec:basinflip), [the saddle](../current/claude-giga-lens/index.md#sec:samplersaga) | H1/H2/H3, the $191$-nat flip, why HMC failed and SMC did not | [Ch. 25](25-money-number.md#the-evidence-flip), [Ch. 26](26-the-saddle.md#the-map-is-a-saddle) |
| [§7 Sampler Benchmark](../current/claude-giga-lens/index.md#sec:benchmark), [A100 phase](../current/claude-giga-lens/index.md#sec:p2c) | nautilus vs. PT-HMC vs. the GIGA-Lens baseline, at budget parity and until converged | [The frontier](#the-frontier), below — not a full chapter |
| [§8 Recipe + Euclid Q1](../current/claude-giga-lens/index.md#sec:recipe) | End-to-end export to COOLEST; an honest $1/3$-clean characterization on independent data | not covered — see [What we skipped](#what-we-skipped) |
| [§9 Discussion](../current/claude-giga-lens/index.md#sec:discussion) | "Necessary but not sufficient"; the two parked follow-on pillars | this chapter |
| [§10 Reproducibility](../current/claude-giga-lens/index.md#sec:repro) | Seven stack defects, the parity harness, the retraction culture | [Ch. 22](22-inference.md#the-gigalens-recipe) — partial |
| [LensJudge parity, Baseline](../current/lensjudge-parity/index.md#sec:baseline), [Power](../current/lensjudge-parity/index.md#sec:power) | Purity/QWK human baseline; $\mathrm{AUC}(\text{grade vs. truth})=0.577$<!-- check: ch28.human_grade_auc = 0.577 ± 0.001 --> | [Ch. 28](28-the-label.md#the-flat-line) |

Two macros are worth knowing when you read the raw report, because they are
the notation hazard this guide opened against: `main.tex` writes the money
parameter as `\gammaEPL` (always a bare $\gamma$) and external shear as
`\gext`, `\gextone`, `\gexttwo` — a bare $\gamma$ in that document is never
shear, by construction of the macro table, not by a convention this guide
imposed on it.

## Five questions to ask in any lensing meeting { #five-questions }

Five questions, phrased so you could ask them of any lens-modeling result,
not just this campaign's. Each one has already produced a wrong number in
this repository before it was caught.

**1. Are your pixel errors actually independent?** Drizzling resamples an
*HST* exposure onto a finer grid with a weighted, overlapping kernel — Chapter
11 derives why that correlates neighboring pixels — and a per-pixel diagonal
Gaussian likelihood assumes the opposite. Get this wrong and you do not get a
slightly-too-confident answer; you get a spurious *second mode*: the diagonal
likelihood's $+162.2$-nat preference for the wrong basin was entirely a
noise-covariance artifact, not a property of the galaxy.

**2. Is the PSF kernel sampled at the pixel scale your code assumes?**
`lenstronomy`'s `subgrid_kernel` upsamples internally by the supersampling
factor; feeding it an already-supersampled kernel applies that refinement
twice, quietly broadening the effective PSF by $2\times$. The cost, on this
same system, was $\chi^2_\nu$ stuck at $3.4$
<!-- check: ch08.chi2_nu_psf_broadened = 3.4 ± 0.05 --> until the convention
was fixed, at which point it dropped to $1.05$
<!-- check: ch08.chi2_nu_psf_fixed = 1.05 ± 0.05 --> with the *same* data and
the *same* noise model.

**3. Was that noise calibration performed on model-subtracted residuals?**
Calibrate a sky-noise sigma on the raw image near a bright lens galaxy and a
large fraction of what you are calling "noise" is diffuse lens-light wing
flux, not sky. This campaign's earlier report celebrated a suspiciously
*too*-good $\chi^2_\nu = 0.451$
<!-- check: ch08.chi2_nu_sky_artifact = 0.451 ± 0.001 -->; the honest
recalibration on model-subtracted residuals gave $0.92$
<!-- check: ch08.chi2_nu_sky_honest = 0.92 ± 0.01 -->, comfortably under the
group's $<1.1$ bar but more than double the artifact's implied noise floor.

**4. Is that point actually a mode, or just a place where the gradient
vanished?** [Chapter 5](05-linear-algebra.md#definiteness-and-saddles)'s
folklore — a vanishing gradient is not a verdict in non-convex optimization —
is not folklore here. The `map_polish` stage that
produced this campaign's saddle-consistent $\gamma=1.27$
<!-- check: ch05.gamma_at_saddle_map = 1.27 ± 0.01 --> had, by
construction, no way to notice that a direction existed along which the
log-posterior kept *rising*: the Hessian's minimum eigenvalue there is
$-14.85$ <!-- check: ch05.saddle_min_eig = -14.85 ± 0.01 -->, and the true
higher-density point sits $74$ nats up
<!-- check: ch05.saddle_logp_gain = 74 ± 1 --> at $\gamma=1.10$
<!-- check: ch05.gamma_at_true_peak = 1.10 ± 0.01 -->.

**5. If a convergence diagnostic is bad, what is it actually diagnosing?** A
bad $\hat R$ does not mean the parameter you care about failed to converge —
it means *something* in the chain did. The $\hat R = 22.3$
<!-- check: ch21.rhat_source_centre_hi = 22.3 ± 0.05 --> that stalled this
campaign's local-metric fixes was carried almost entirely by a
Sérsic-versus-shapelet source-*centre* degeneracy, decoupled from $\gamma$
(two clusters at $-0.15$
<!-- check: ch21.source_centre_cluster_neg = -0.15 ± 0.001 --> and $+0.09$
<!-- check: ch21.source_centre_cluster_pos = 0.09 ± 0.001 -->). Blaming the
slope for a diagnostic a *different, physically uninteresting* parameter was
carrying would have discarded a real result over a nuisance.

Two of these five are no longer questions you have to remember to ask — this
campaign converted them into code. `cgl/guards.py:74-91`
(`assert_psf_sampling`) refuses a mismatched PSF pixel scale at the door,
citing this exact incident by name in its own docstring; `cgl/guards.py:129-142`
(`assert_model_subtracted_sky`) refuses an uncalibrated noise artifact the
same way, citing the $0.451$ retraction by number. Questions 1, 4, and 5 are
not yet guards. [The frontier](#the-frontier) says why that gap is worth
closing, and by whom.

## The frontier { #the-frontier }

Two frontiers sit inside the science this campaign already ran, and one sits
outside it entirely.

**The scientific frontier is already named, by the campaign's own
Discussion section.** The bracket that will not close — fine-steep at
$1.816$ above the anchor, binned-low at $1.103$ below it — is evidence of a
real limitation, not a bug: the residual scale dependence "points to
additional systematics, most plausibly the source model and PSF"
(`reproductions/claude-giga-lens/papers/main.tex:966-970`). Two follow-on
pillars are explicitly parked to address exactly this: a Gaussian-process or
pixelated source model (the shapelet basis this campaign used is a fixed,
low-order function family, and a flexible source can absorb structure a
rigid one instead forces into the mass slope), and higher-order multipole
mass structure, whose reported $\sim1\%$-level effect on $\gamma$
(`reproductions/claude-giga-lens/papers/main.tex:1358-1368`) is the same
order as the correlated-noise correction this book just spent five chapters
deriving. Neither pillar needs new mathematics beyond what you already have —
a GP source is another linear-in-amplitude marginalization of exactly the
shape [Ch. 22](22-inference.md#marginalising-linear-amplitudes) already
profiles out, and a multipole term is a few more harmonics added to the
deflection sum of [Ch. 20](20-profiles.md#the-epl-and-gamma).

**The inference frontier is a genuinely open algorithms problem, not a
physics one.** The sampler benchmark's own verdict is two-sided: nautilus
dominates efficiency wherever it applies — $2.6$ to $307\times$
<!-- check: ch29.nautilus_ess_ratio_min = 2.6 ± 0.01 -->
<!-- check: ch29.nautilus_ess_ratio_max = 307.0 ± 0.1 --> the baseline's ESS
per gradient — but is *disqualified* precisely on the ill-conditioned,
marginalized regime this campaign's real posteriors occupy; parallel-tempered
HMC is the most reliable until-converged method but is not the fastest one;
and the flow-assisted recipe this campaign tried, GL-NT, reaches only $0.03$
to $0.05\times$ <!-- check: ch29.glnt_ess_ratio_min = 0.03 ± 0.001 -->
<!-- check: ch29.glnt_ess_ratio_max = 0.05 ± 0.001 --> the baseline against
its own pre-registered target of $3\times$
<!-- check: ch29.glnt_target_ratio = 3.0 ± 0.01 -->. No single sampler in
this benchmark is both fast and trustworthy on a $46$-dimensional,
condition-$10^{14}$, occasionally saddle-shaped real lens posterior. Building
one — a sampler, or a routing policy over several, that gets nautilus's speed
without inheriting its blindness to bad conditioning — is an algorithms and
systems problem a CS-trained collaborator is better positioned to attack
than most astronomers on this project.

**The frontier outside the science is the research process itself, and it is
the closest to home.** This report's title page states, without
qualification, that "all work reported here — code, experiments, analysis,
and text — was performed by Claude Code (Anthropic), guided by Greg Benson"
(`reproductions/claude-giga-lens/papers/main.tex:76-84`). One incident in
`reproductions/claude-giga-lens/CAMPAIGN.md`, which Chapter 1 asked you to
read end to end, stands out precisely because it is not an astrophysics
failure: a correlated-noise SMC canary failed trying to allocate $120.4$ GB
<!-- check: ch29.smc_oom_attempted_gb = 120.4 ± 0.05 --> with $300$ particles
<!-- check: ch29.smc_particles_oom = 300 ± 0 -->, and the implementation
agent handling that failure "DIED during a multi-day gap and never processed
the failure / never fired its `SMC_PARTICLES=200` fallback" — the campaign
stalled four days
<!-- check: ch29.impl_agent_stall_days = 4 ± 0 -->
(`reproductions/claude-giga-lens/CAMPAIGN.md:168-169`) until a fresh agent
converged it with $128$ particles
<!-- check: ch29.smc_particles_fixed = 128 ± 0 -->. That is a liveness bug —
a long-running worker with no heartbeat, watchdog, or idempotent resume path
— and diagnosing it requires no astrophysics whatsoever.

!!! tip "You already know this"
    "The worker died and nobody noticed for four days" is not a lensing
    problem; it is a liveness problem, and multi-agent-systems research has
    a whole vocabulary for it — heartbeats, watchdog timers, idempotent
    retries, supervisor trees. `cgl/guards.py`'s two runtime assertions
    (Questions 2 and 3, above) are the same instinct one level down: turn a
    postmortem into code that makes the same mistake impossible to repeat
    silently — exactly what this guide's own `lint_guide.py` and
    `worked_examples.py` do to its numbers (Chapter 1). You do not need to
    learn gravitational lensing to see where this campaign's process has a
    gap; you need exactly the training this book assumed you already had on
    page one.

The discovery half of the repository is already moving toward this same
frontier: a $27$B open model fine-tuned on the human grade catalog reaches
$\mathrm{AUC}(\text{truth})=0.685$
<!-- check: ch28.student_truth_auc = 0.685 ± 0.001 -->, statistically
non-inferior to the human grader's own $0.577$
<!-- check: ch28.human_grade_auc = 0.577 ± 0.001 --> ($\Delta\mathrm{AUC} =
+0.108$ <!-- check: ch28.delta_auc_student_vs_human = 0.108 ± 0.001 -->, 95%
CI $[-0.028,\ 0.241]$
<!-- check: ch28.delta_auc_student_vs_human_lo = -0.028 ± 0.001 -->
<!-- check: ch28.delta_auc_student_vs_human_hi = 0.241 ± 0.001 -->), from a
leakage-safe split, a frozen gate, and a pre-registered non-inferiority
margin — the same discipline Chapter 28 asks of any classifier, applied to
the wall it names. [Ch. 28](28-the-label.md#what-to-do-instead) has the rest
of that argument; it belongs there, not here.

## What we skipped { #what-we-skipped }

This book is honest about its own edges, the way this campaign is honest
about its numbers.

- **The full sampler-benchmark matrix.** Chapter 23 gives you the ideas —
  HMC, tempered SMC, normalizing flows, mass-matrix preconditioning — but not
  the full multi-contender, multi-target horse race. `main.tex` §7 has the
  complete matrix.
- **The end-to-end recipe and Euclid Q1.** The campaign exports its final
  posteriors to the code-independent COOLEST standard and re-runs the whole
  pipeline on three independent Euclid systems, with an honest "one of three
  clean" result. None of that is in this book; `main.tex` §8 is.
- **The seven documented stack defects.** XLA compile livelocks, a `chex`
  pin that silently upgraded `jax`, a triton GEMM abort on tiny float32 dots
  — real, documented, genuinely useful engineering, but plumbing rather than
  mathematics. `main.tex` Table "tab:defects" in §10 has all seven.
- **Full general relativity.** This book derives the deflection angle from
  the Newtonian estimate and Eddington's factor of two ([Ch.
  16](16-deflection.md#the-factor-of-two)) and never writes down the Einstein
  field equations, a geodesic, or a metric beyond the flat FRW background of
  [Ch. 14](14-frw.md#the-frw-metric). Multi-plane lensing and full
  time-delay cosmography beyond the mass-sheet argument of [Ch.
  21](21-degeneracies.md#time-delays-and-h0) are likewise out of scope.
- **LensJudge's machine-vetting pipeline in depth.** Chapter 28 covers the
  human-baseline ceiling this chapter just cited a downstream result from;
  the training-data curation, the leakage firewall, and the model-selection
  protocol behind the $27$B student model live in
  `reproductions/lensjudge/parity/FINDINGS.md` Phase C/D, not in this guide.
- **Flow architectures below the change-of-variables level.** Chapter 23
  tells you what a normalizing flow computes and why NeuTra preconditioning
  helps; it does not derive a coupling layer, a spline transform, or
  `flowjax`'s internals.

## Connect to the repo { #connect }

- [`reproductions/claude-giga-lens/cgl/marg.py:31-55`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/marg.py#L31)
  — the Occam term Exercise 29.1 asks you to require positive-definiteness of.
- [`reproductions/claude-giga-lens/cgl/flows.py:1-21`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/flows.py#L1)
  — the flow pullback-density docstring, row 2 of the closed ledger.
- [`reproductions/claude-giga-lens/cgl/guards.py:74-91`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/guards.py#L74)
  and [`:129-142`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/guards.py#L129)
  — two of the five questions already turned into runtime assertions, each
  citing its own incident by name.
- [`reproductions/claude-giga-lens/cgl/e2.py:554,557-558`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L554)
  — the Hessian eigendecomposition that caught the saddle; Exercise 29.3
  turns it into a third guard.
- [`reproductions/claude-giga-lens/CAMPAIGN.md:133`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/CAMPAIGN.md#L133)
  — the "P1c MONEY NUMBER" entry the $\gamma$ Ledger traces to;
  [`:168-169`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/CAMPAIGN.md#L168)
  — the implementation-agent stall behind [The frontier](#the-frontier).
- [`reproductions/claude-giga-lens/papers/main.tex`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/papers/main.tex)
  — the report itself; [the paper decoder](#the-paper-decoder) maps every
  `\label{sec:...}` in it to a chapter of this book.
- [`reproductions/lensjudge/parity/FINDINGS.md:238-243`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/lensjudge/parity/FINDINGS.md#L238)
  — Phase D's non-inferiority result for the trained student model.
- [`site/guide_src/worked_examples.py`](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/worked_examples.py)
  and [`site/guide_src/lint_guide.py`](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/lint_guide.py)
  — this guide's own version of `guards.py`.
- [`site/guide_src/contract/outline.yml`](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/contract/outline.yml)
  — the frozen chapter table behind the paper decoder's right-hand column.

## Exercises { #exercises }

??? question "Exercise 29.1 — the Occam term needs positive-definiteness"
    Chapter 23 derived $\int\exp(-\tfrac12\mathbf{x}^\top A\mathbf{x})
    \,d\mathbf{x} = (2\pi)^{k/2}/\sqrt{\det A}$ by the substitution
    $\mathbf{x}=A^{-1/2}\mathbf{u}$; `marg.py`'s $-\tfrac12\log\det A$ is
    that identity's logarithm. Using this, explain in two or three sentences
    why `cgl/marg.py`'s Cholesky factorization of $A$ requires $A$ to be
    positive *definite* — not merely symmetric — for the number it computes
    to mean anything, and connect that requirement to what Chapter 5 found at
    the correlated-likelihood MAP.

    ??? success "Solution"
        A Cholesky factorization $A=LL^\top$ exists in real arithmetic only
        when $A$ is positive definite: a negative or zero eigenvalue makes
        $\sqrt{\det A}$ complex or zero, and the "probability density" the
        integral computes stops meaning anything, because a Gaussian with
        negative-definite curvature has no finite total mass to normalize in
        the first place. That is exactly the fact Chapter 5 found at the
        correlated-likelihood MAP: minimum eigenvalue $-14.85$, five of
        forty-six directions negative
        <!-- check: ch05.saddle_min_eig = -14.85 ± 0.01 -->
        <!-- check: ch05.saddle_n_negative = 5 ± 0 -->. A Laplace/Occam
        computation built on that indefinite Hessian is not merely
        inaccurate — it is asking for the log of a negative number, which is
        the algebraic reason (not just the sampling reason) a saddle-seeded
        metric had to be abandoned rather than patched.

??? question "Exercise 29.2 — coincidence, or connection?"
    The Log-Det Ledger's real instance of the Occam term is $161.6145$ nats
    <!-- check: ch22.occam_correction_production_nats = 161.6145 ± 0.001 -->
    (the correction from marginalizing $28$ shapelet amplitudes at the
    production point). The $\gamma$ Ledger's evidence swing is $191.1$ nats
    <!-- check: ch25.evidence_swing_nats = 191.1 ± 0.05 -->. These are
    suspiciously close in size. Are they the same effect wearing two names,
    or an unconnected coincidence? Say what each number actually measures
    before you answer.

    ??? success "Solution"
        The Occam term is a single additive correction, evaluated once at
        (approximately) the MAP, for profiling out the $28$ linear shapelet
        amplitudes of *one* marginalized model — it appears, at nearly the
        same value, on both sides of *any* comparison between two fits of
        that same model family, diagonal or correlated, steep basin or low.
        The evidence swing is a difference of two *entire* per-basin
        evidence integrals (steep minus low), computed once under a diagonal
        likelihood and once under a correlated one — a change to the noise
        covariance and whitening operator, nothing to do with how the linear
        amplitudes are profiled out. Tracing the actual formulas, there is no
        algebraic term shared between $-\tfrac12\log\det A$ (Ch. 22) and
        $\log Z_{\mathrm{steep}} - \log Z_{\mathrm{low}}$ (Ch. 25) — the first is a
        per-evaluation constant that mostly cancels in any such difference,
        the second is exactly that kind of difference computed on an
        unrelated axis. Two numbers of similar order for a problem with
        tens of dimensions is not itself evidence of a shared cause; it is
        the same trap as the report's own $17\sigma$ claim, run in reverse —
        a coincidence in magnitude is not a derivation, and this book's
        method is to require the derivation before it accepts the pattern.

??? question "Exercise 29.3 — write the missing guard"
    `cgl/guards.py` already encodes Questions 2 and 3 from this chapter as
    runtime assertions. Sketch, in a few lines of pseudocode, a guard for
    Question 4 — the saddle — using the Hessian eigendecomposition
    `cgl/e2.py:554,557-558` already computes. What should it assert, and at
    what stage of the pipeline should it run?

    ??? success "Solution"
        A minimal version, in the same style as `assert_model_subtracted_sky`:

        ```python
        def assert_map_is_a_mode(eigvals, min_eig_floor=0.0):
            """map_polish must land on a mode, not a saddle (min_eig=-14.85,
            5/46 negative sank two rounds of PHMC budget before this was
            root-caused; CAMPAIGN.md "P1c metric-fix attempts")."""
            n_neg = int((eigvals <= min_eig_floor).sum())
            if n_neg > 0:
                raise GuardError(
                    f"MAP Hessian has {n_neg} non-positive eigenvalue(s) "
                    "(saddle, not a mode) — polish further, or route to a "
                    "global sampler (tempered SMC) instead of local HMC."
                )
        ```

        It should run immediately after `map_polish`, before any momentum
        metric is built from the Hessian — exactly where `cgl/e2.py:554`
        already computes the eigendecomposition for diagnostic printing, but
        did not stop the pipeline from spending two more rounds of budget on
        a Laplace metric anyway. The guard cannot fix the saddle; it only
        stops the campaign from re-discovering, by two expensive failed
        runs, a fact the eigenvalues already stated on the first one.

??? question "Exercise 29.4 — the vote"
    Chapter 1 asked you to guess a number, and Chapter 25 graded the guess.
    This chapter asks a different vote: given everything in [The
    frontier](#the-frontier), if you had one
    CS-trained graduate student and three months, would you point them at
    (a) the source-model and PSF systematics that keep the $\gamma$ bracket
    from closing, (b) the sampler benchmark's efficiency-versus-robustness
    gap, or (c) the multi-agent research process itself? Defend your choice
    in three or four sentences, citing at least one specific number or
    incident from this chapter.

    ??? success "Solution"
        No single answer is correct; the honest reason to weigh all three is
        that this campaign's own scope decisions named (a) and (b) as open
        items but never once flagged the agent-liveness gap as a research
        item at all. A case for (c): a four-day stall
        <!-- check: ch29.impl_agent_stall_days = 4 ± 0 --> from a missing
        watchdog will recur on every future campaign this group runs,
        regardless of which physics pillar is under study, while (a) and (b)
        each fix one number or one benchmark and stop. A counter-case for
        (b): it is a clean, well-scoped algorithms problem with a benchmark
        already built to beat ($2.6$–$307\times$ efficiency spread, a
        $3\times$ target GL-NT missed by two orders of magnitude) — exactly
        the kind of problem a CS thesis needs, where (c) is a practice, not a
        deliverable. Either answer, defended with a specific number rather
        than an adjective, is the point: a vote with no citation attached is
        exactly what this book has spent 28 chapters teaching you not to
        accept.
