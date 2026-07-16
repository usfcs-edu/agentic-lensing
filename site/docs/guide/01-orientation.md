# 1. What this repository does, and what one number costs

This repository does two jobs that share almost nothing except the word
"lens." The first job is a search: comb through tens of millions of galaxy
images for the rare few whose light has been bent by something massive
sitting in front of them. The second job is a fit: once a system is confirmed
as a genuine lens, extract a small number of physical parameters — a mass, a
shape, a slope — from the pixels of its image. This guide takes you from
calculus you already own to the point where you can derive the mathematics
behind both jobs yourself, and then check it against numbers this repository
has actually published. It has one destination: a single number, produced by
the repository's most ambitious modeling campaign to date,

$$\gamma = 1.103 \pm 0.008$$

the density slope of one galaxy's mass profile <!-- check: ch25.gamma_money = 1.103 ± 0.008 -->.
$\gamma$ here is the exponent in $\rho \sim r^{-\gamma}$: how fast mass density
falls off with radius. It is not the shear — this repository always spells
shear $\gamma_{\mathrm{ext}}$, or its components $\gamma_1, \gamma_2$, never a
bare $\gamma$. That distinction gets restated at every chapter that could
possibly confuse them, starting now, because the two most common ways to
misread this repository's mathematics are conflating that $\gamma$ with shear,
and conflating source-plane position $\boldsymbol{\beta}$ with an inverse
temperature (also usually called $\beta$ everywhere outside lensing — this
repository calls its tempering parameter $\lambda$ instead, precisely to avoid
that collision). By Chapter 25 you will know exactly what pipeline produced
$1.103 \pm 0.008$, what the repository's own report claims about how
surprising that number is, and where that claim's arithmetic actually goes
when you do it yourself, one division at a time.

## The two halves { #the-two-halves }

Read `reproductions/REPRODUCTIONS.md` and the two halves fall out immediately:
one set of lineages *finds* candidate lenses, another *models* the ones that
survive. They are different disciplines wearing the same word.

**Discovery** starts from an imaging survey — the DESI Legacy Surveys' *grz*
imaging, tens of millions of galaxy cutouts — and asks a yes/no question of
each one: is this a lens? The tools are a convolutional classifier (`huang-2020`,
`huang-2021`, `inchausti-2025`), a difference-imaging pipeline that catches a
lensed transient by what changes between exposures (`sheu-2023`, `sheu-2024a`),
and a friends-of-friends spectroscopic search that flags any sightline
carrying two incompatible redshifts (`hsu-2025`, `dawes-2022`). The output of
discovery is not a lens; it is a *candidate*, carrying a grade — a confidence
label attached by a human, or by a machine standing in for one. Chapter 27
derives why this problem gets structurally harder as the survey's resolution
runs out, and Chapter 28 (`reproductions/lensjudge/`) asks the question this
guide considers its most consequential for a machine-learning audience: how
good is that grade, actually? On the 130 systems
(`reproductions/lensjudge/parity/FINDINGS.md:39`) where independent follow-up
later confirmed or refuted the candidate, the human ordinal grade predicts
that later truth at AUC $0.577$, indistinguishable from a coin flip
<!-- check: ch28.human_grade_auc = 0.577 ± 0.001 -->. Purity is flat across the
grade itself — 95%, 92%, 91% for grades A, B, C
<!-- check: ch28.purity_A = 0.952 ± 0.001 -->
<!-- check: ch28.purity_B = 0.920 ± 0.001 -->
<!-- check: ch28.purity_C = 0.909 ± 0.001 --> —
and two trained graders agree with each other at a quadratic-weighted kappa of
only $0.42$ <!-- check: ch28.intergrader_qwk = 0.42 ± 0.001 -->. Whatever you
train against a label this noisy inherits its ceiling. That is Chapter 28's
whole argument, and it is why the outline puts it second, immediately after
this one.

**Modeling** starts one step later: a lens is confirmed, its redshifts are
measured, and the question is no longer "is it a lens" but "what, precisely,
is doing the lensing." The lineage runs `gu-2022` (the method) through
`cikota-2023` and `sheu-2024b` (single systems), `foundry-i` (the first
genuine HMC posterior on a real HST system), to `claude-giga-lens`
(`reproductions/REPRODUCTIONS.md:27`), the campaign that produces this guide's
destination number. Where discovery is millions of low-dimensional decisions
against a label nobody fully trusts, modeling is one system at a time against
a forward model you can write down exactly — a few dozen parameters, a
render, a per-pixel comparison — but whose *posterior geometry* turns out to
be vicious: nearly flat in some directions, curved by fourteen orders of
magnitude in others (a condition number of $10^{14}$,
`reproductions/claude-giga-lens/papers/main.tex:352`), and in one documented
case, a maximum-a-posteriori point that is not a maximum at all. Discovery stress-tests the label. Modeling
stress-tests the likelihood and the sampler. Keep that distinction in view: a
huge fraction of this guide (Parts I, IV, and V) exists to make modeling's
failures derivable, and Chapter 28 exists to make discovery's failure
*undeniable* — and, notably, dependent on no physics or calculus at all.

## The Rosetta stone { #the-rosetta-stone }

Every idea in the modeling half has a name you already carry from machine
learning. This repository's mathematics is not a new subject: it is a small,
specific set of linear-algebra and probability operations, run on sky
pixels instead of tensors. The table below is the guide's map from what you
own to what this repository calls it; each row gets derived properly, once,
at the chapter cited.

| What you already know | What this repository calls it | Derived at |
|---|---|---|
| The change-of-variables factor $\lvert\det J\rvert$ in a normalizing flow | Magnification, $\mu = 1/\det A$ | [Ch. 4](04-multivariable.md#det-j-as-area-scaling), [Ch. 18](18-magnification.md#magnification-is-a-jacobian) |
| Cross-entropy loss, measured in nats | The Bayesian evidence $\log Z$, always quoted in nats | [Ch. 8](08-probability.md#evidence-and-nats) |
| Weight decay, an $L_2$ penalty on a linear layer | A Gaussian prior on the linear source amplitudes | [Ch. 8](08-probability.md#ridge-is-a-prior), [Ch. 22](22-inference.md#marginalising-linear-amplitudes) |
| A BIC-style complexity penalty | The $-\tfrac12\log\det A$ Occam term that `gigalens` omits from its linear solve | [Ch. 22](22-inference.md#the-occam-term), [Ch. 23](23-samplers.md#closing-the-log-det-ledger) |
| The condition number of a loss Hessian | A parameter degeneracy: a flat direction the data cannot see | [Ch. 5](05-linear-algebra.md#conditioning), [Ch. 21](21-degeneracies.md#degeneracy-is-gauge-symmetry) |
| A saddle point a naive optimizer mistakes for a minimum | A real system's MAP fit, confirmed to be a saddle, not a mode | [Ch. 5](05-linear-algebra.md#definiteness-and-saddles), [Ch. 26](26-the-saddle.md#the-map-is-a-saddle) |
| Whitening features before trusting a diagonal covariance | Whitening drizzled pixel noise before trusting a per-pixel Gaussian likelihood | [Ch. 7](07-fourier.md#whitening), [Ch. 24](24-correlated-noise.md#convolutional-whitening) |
| A temperature schedule in simulated annealing | Tempered sequential Monte Carlo, $\lambda: 0 \to 1$ | [Ch. 23](23-samplers.md#tempering-and-smc) |
| A classifier's chosen operating point (TPR at fixed FPR) | The lens-finder's decision threshold, and the resolution wall behind it | [Ch. 27](27-discovery.md#the-operating-point) |
| A label so noisy it caps a classifier's achievable AUC | A human "grade" that predicts truth barely above chance | [Ch. 28](28-the-label.md#the-flat-line) |

!!! tip "You already know this"
    The evidence $\log Z$ this repository tallies is in nats — the same unit
    a cross-entropy loss is reported in, because it is the same integral: a
    log-probability, summed in the natural base. A swing of 191 nats in
    $\log Z$ <!-- check: ch25.evidence_swing_nats = 191.1 ± 0.05 --> is a
    likelihood ratio of $e^{191}$. Jeffreys' scale calls anything past 5 nats
    of $\log Z$ "decisive" — this is 38 times that bar, before you have
    touched a single equation of gravitational lensing.

That box previews the entire shape of the guide's central device, the **Log-Det
Ledger**: the quantity $\log|\det(\cdot)|$ of a matrix appears three times in
this repository, in three costumes — magnification, a normalizing flow's
change of variables, and a Bayesian Occam factor — and every chapter that
meets one adds a row. Chapter 23 closes it, having shown all three are the
same computation.

## The money number { #the-money-number }

The report itself calls $1.103 \pm 0.008$ "the money number"
(`reproductions/claude-giga-lens/papers/main.tex:756`), and the name is
earned: it is the number the whole `claude-giga-lens` campaign was built to
produce. Here is the shape of the argument, with no derivation yet — that is
Chapters 20 through 25's job — only the destination.

A real HST system, `foundry-i`'s target, was drizzled (resampled) to three
pixel scales for the same exposures. At the *binned* scale, a per-pixel
diagonal noise model — the statistical core of every fast GPU lens code,
including `gigalens` itself — produced a posterior for $\gamma$ that split
into two disconnected basins, with zero chains crossing between them. That
diagonal likelihood was confident the "steep" basin was the right one: its
own evidence favored steep over low by
$+162.2$ nats <!-- check: ch25.dlogz_diagonal = 162.2 ± 0.1 -->. The
campaign's contribution is a likelihood that models the drizzle noise as
genuinely correlated between pixels rather than diagonal, built and validated
in Chapter 24 on top of the marginalization machinery of Chapter 22. Applied
to the same binned product, that
correlated likelihood reverses the verdict outright: evidence now favors the
low basin by $-28.9$ nats <!-- check: ch25.dlogz_correlated = -28.9 ± 0.1 -->
— a swing of $191.1$ nats in total. The steep basin was a noise-covariance
artifact of the diagonal likelihood, not a genuine second mode. That flip is
[the campaign's real-data verdict](../current/claude-giga-lens/index.md#sec:realdata),
and Chapter 25 calls it Hypothesis 1, confirmed.

But the correlated likelihood does not stop there — it also *moves* the
recovered value of $\gamma$ itself, and moves it further than the campaign
expected. The restored low basin converges, under a 128-particle tempered SMC
sampler (`reproductions/claude-giga-lens/papers/main.tex:761`), to
$\gamma = 1.103 \pm 0.008$. The campaign's pre-registered anchor —
a diagonal fit at the native, least-resampled pixel scale, the scale where a
diagonal likelihood is closest to correct — sits at
$\gamma = 1.433 \pm 0.034$ <!-- check: ch25.gamma_anchor = 1.433 ± 0.034 -->.
Those two numbers disagree by more than either of their quoted uncertainties
can easily explain, and the report's own abstract puts a number on that
disagreement: about $17\sigma$ (`reproductions/claude-giga-lens/papers/main.tex:98`).
Hold that figure loosely. Chapter 25 has you compute the discrepancy yourself,
from nothing but the four numbers already on this page, and it does not come
out to 17 — under any of the three ways you might reasonably define "sigma"
here. That gap between a headline claim and its own arithmetic is not a
gotcha invented for this guide; the report's own footnote,
`reproductions/claude-giga-lens/papers/main.tex:763`, records a second, smaller
figure right next to the first.

Before you read any further into this guide: write down, in one sentence, your
best guess right now at whether $\gamma = 1.103 \pm 0.008$ is a trustworthy
measurement of this galaxy's density slope, and your one reason why. Do not
look ahead. Chapter 25's closing section, ["the verdict"](25-money-number.md#the-verdict),
is where you open it.

## How to read this guide { #how-to-read }

The book has seven parts after this one, and they are not meant to be read in
a straight line by default.

| Part | Chapters | What it buys |
|---|---|---|
| I — The mathematical spine | 2–8 | Derivatives through probability, built from the calculus and linear algebra you already have |
| II — The physical universe | 9–12 | Arcseconds, galaxies, photons, redshifts: the vocabulary of an observation |
| III — Cosmology | 13–15 | Detachable — the lens-modeling likelihood in this repository uses no cosmology at all |
| IV — Gravitational lensing | 16–21 | The lens equation, magnification, the Einstein radius, mass profiles, degeneracies |
| V — This repository's science, decoded | 22–26 | The forward model, the samplers, the correlated-noise likelihood, and two dependency chains — Spine 1 (Ch. 25, the money number) and Spine 2 (Ch. 26, a saddle in the same campaign's optimization) — each ending in a verdict |
| VI — Discovery | 27–28 | The survey, the finders, the resolution wall, and the label that caps all of it |
| VII — Synthesis | 29 | Both ledgers close, a decoder from this repository's own reports to this guide, and where a CS professor has comparative advantage |

One deliberate break from that order: **read Chapter 28 next.** It needs no
physics and no calculus past what you already own — it is pure ML
epistemics, a question about labels and AUC that you could ask of any
classifier you have ever shipped — and it carries this repository's most
striking result. Reading it second, before the calculus ramp of Part I, is
the difference between doing homework and being on a case.

Two recurring devices carry the argument across chapters. A boxed **"You
already know this"** tip, like the one above, marks a real bridge from ML to
lensing at the exact point it becomes load-bearing — never decorative, never
manufactured. And any chapter that constrains $\gamma$ — starting once the
symbol has a precise meaning, in Chapter 20 — adds a row to a running
**$\gamma$ Ledger**, closed in Chapter 25, that tracks only one thing: what
the evidence so far rules in or out about $1.103$.

The guide's own credibility rests on one rule, and you can enforce it
yourself: every computed number in this book is tagged with an HTML comment —
`<!-- check: ch25.gamma_money = 1.103 ± 0.008 -->` is the one attached to this
chapter's destination number — and every tag names a function in
`site/guide_src/worked_examples.py` that computes it from scratch or pins it
to a cited artifact. Run
`~/.venvs/lensjudge/bin/python site/guide_src/worked_examples.py --check`
yourself, right now if you like; it recomputes every pinned number in this
guide and fails loudly if one has drifted. This repository's own final report
carries a "$\sim17\sigma$" claim that reconciles with none of the
uncertainties it quotes elsewhere. This guide's answer to that is not a
better adjective. It is a script you can run.

## Connect to the repo { #connect }

- `reproductions/REPRODUCTIONS.md:19`, `:29`, `:38` — the three lineage
  tables behind "the two halves": GIGA-Lens modeling, image-based finders,
  specialty discovery. Row `:27` is `claude-giga-lens` itself, the source of
  this chapter's destination number.
- `reproductions/claude-giga-lens/papers/main.tex:98` — the report's
  abstract, which states the whole campaign (and its $\sim17\sigma$ claim) in
  one paragraph; `:756`–`:759` is the boxed money-number result itself;
  `:763` carries the footnote with the report's own arithmetic.
- `reproductions/claude-giga-lens/CAMPAIGN.md:133` — the dated gate record
  ("P1c MONEY NUMBER — 2026-07-14") every number in this chapter traces to;
  read the whole file once to see this repository's retraction culture in the
  raw, the register this guide tries to match.
- `reproductions/lensjudge/parity/FINDINGS.md:40` and `:108` — the
  grade-purity table and the headline AUC behind the discovery half's
  teaser in this chapter; Chapter 28 is where they get a full argument.
- `site/guide_src/worked_examples.py` — every tagged number in this guide,
  in one file, runnable; `site/guide_src/contract/outline.yml` is this
  guide's own chapter table, if you want to see the whole map at once instead
  of one part at a time.

## Exercises { #exercises }

??? question "Exercise 1.1 — The sealed envelope"
    You already wrote this down once, above. If you skipped it, do it now:
    one sentence, your best guess at whether $\gamma = 1.103 \pm 0.008$ is a
    trustworthy measurement of the real slope of this galaxy's mass profile,
    and one reason why. Note the date. Do not revise it after reading further.

    ??? success "Solution"
        There is no answer to check here yet — only a receipt. Chapter 25
        walks the same chain of evidence this chapter only named: the
        191-nat basin flip, the cross-scale bracketing, and the sigma
        arithmetic the report's own abstract does not survive. When you
        reach ["the verdict"](25-money-number.md#the-verdict), reread your
        sentence before you read the chapter's own conclusion. The only way
        to fail this exercise is to have skipped writing the guess down
        before you knew where the argument was going.

??? question "Exercise 1.2 — Which half?"
    For each task below, say whether it belongs to discovery or modeling,
    and which Part of this guide you would need to have read to follow it in
    detail.

    1. Deciding whether a 101×101-pixel DESI cutout contains a lens at all.
    2. Recovering the axis ratio $q$ and position angle $\phi$ of a
       confirmed lens's mass ellipse from its pixels.
    3. Measuring the velocity dispersion of a candidate lens galaxy from its
       spectrum, to decide whether two objects on one sightline are really
       at different redshifts.
    4. Deciding whether an evidence swing of 191 nats between two basins of
       a posterior is large enough to trust.

    ??? success "Solution"
        1. Discovery — Part VI (Ch. 27's finders and resolution wall).
        2. Modeling — Part IV (Ch. 20's profiles and ellipticity) feeding
           Part V's inference machinery.
        3. Discovery, though it leans on spectroscopy from Part II
           (Ch. 12) — this is exactly the friends-of-friends redshift-pair
           search `hsu-2025` and `dawes-2022` run, upstream of any imaging
           classifier.
        4. Modeling — Part I's probability chapter (Ch. 8, evidence and
           nats) gives you the units; Part V (Ch. 23, Ch. 25) gives you the
           campaign's own answer.

??? question "Exercise 1.3 — Nats to a Bayes factor"
    The evidence swing between the two likelihoods on the binned product is $191.1$ nats <!-- check: ch25.evidence_swing_nats = 191.1 ± 0.05 -->. A natural log and a base-10 log differ by a factor of $\ln 10 \approx 2.3026$. Convert $191.1$ nats to a base-10 log-Bayes-factor by hand (divide by $\ln 10$), then say how many orders of magnitude past Jeffreys' "decisive" threshold of 5 nats ($\approx 2.17$ in $\log_{10}$) that leaves you.

    ??? success "Solution"
        $191.1 / \ln 10 \approx 191.1 / 2.3026 \approx 83.0$ <!-- check: ch25.bayes_factor_log10 = 82.99 ± 0.01 -->. The correlated-likelihood basin flip is a Bayes factor of roughly $10^{83}$ — about 38 times past Jeffreys' decisive bar of $5 / \ln 10 \approx 2.17$ in $\log_{10}$ units. Whatever else Chapter 25 finds wrong with this campaign's headline sigma claim, the basin flip itself is not a close call in any base you choose to log it in.
