# From Calculus to the Money Number

A ground-up guide to the astrophysics, cosmology and mathematics behind this
repository — written for a computer scientist, starting at basic calculus, and
ending somewhere specific.

!!! abstract "The destination"
    In July 2026 this repo's `claude-giga-lens` campaign measured the density
    slope of one galaxy and got

    $$\gamma_{\mathrm{binned}}(\text{corr, low}) = 1.103 \pm 0.008$$

    against a trusted anchor of $\gamma = 1.433 \pm 0.034$. That number is
    **wrong**, the campaign says so itself, and understanding *why* requires
    calculus, linear algebra, Fourier analysis, Bayesian inference, general
    relativity, cosmology, and a working knowledge of how a telescope resamples
    an image. This guide builds all of it, in that order.

    By the end you should be able to derive that number, find the three
    independent reasons to distrust it, and notice that the report's own
    "$\sim 17\sigma$" claim reconciles with none of the uncertainties it quotes.

## Why this guide exists { #why }

This program has produced sixteen paper reproductions and two novel campaigns.
The reports are honest and they are dense, and they assume a reader who already
knows what a caustic is. If you are directing this work from the computer-science
side, you have two options: take the numbers on faith, or build the machinery.

This guide is the second option. It is a companion to the
[onboarding report](https://github.com/usfcs-edu/agentic-lensing/blob/main/plans/AGENTIC_LENSING_ONBOARDING_PLAN.md),
which answers *"what is this program?"*. This one answers *"why is that number
what it is, and would I notice if it were wrong?"*

## How to read it { #how-to-read }

!!! tip "Read chapter 28 second"
    The canonical order is 1 → 29, with one deliberate exception. After
    [Ch. 1](01-orientation.md), jump straight to
    **[Ch. 28 — The label is the problem](28-the-label.md)**, then come back to
    [Ch. 2](02-derivatives.md).

    Chapter 28 needs no physics at all. It is pure machine-learning epistemics,
    it carries the most startling result in the repository — a trained expert's
    lens grade predicts confirmed truth at AUC 0.577, which is consistent with
    chance — and it is the chapter where your existing expertise is worth the
    most. It earns the calculus that follows. A reader who starts at Ch. 2 is
    doing homework; a reader who starts at Ch. 28 is on a case.

Every chapter opens with a **"What you can skip"** box. Take it seriously — it
is there because you have a CS PhD and your time is the scarce resource here.

Worked examples are meant to be *worked*. Every number in this guide is computed
by [`site/guide_src/worked_examples.py`](https://github.com/usfcs-edu/agentic-lensing/tree/main/site/guide_src)
and machine-checked on every build; every figure is computed, not drawn, by
`make_figures.py`. If you want to know what an SIE caustic looks like at
$q = 0.3$, change the number and re-render — the picture will still be true.

## The three log-dets { #the-three-log-dets }

The guide has a spine. The quantity $\log|\det M|$ shows up three times in this
repository, wearing three different costumes, and you already own the third:

| Where | What it looks like | Chapter |
|---|---|---|
| Lensing magnification | $\mu = 1/\det A$ | [Ch. 18](18-magnification.md) |
| Normalizing-flow density | $\log q(x) = \log p(u) - \log\lvert\det J\rvert$ | [Ch. 23](23-samplers.md) |
| The Bayesian Occam term | $-\tfrac12 \log\det A$ | [Ch. 22](22-inference.md) |

They are the same operation. The first decides how bright a lensed arc is; the
second is how the campaign's sampler moves; the third is the term that
GIGA-Lens omits and that this repo had to add before its optimiser would
converge. Each chapter that meets one adds a row to the ledger.
[Ch. 23](23-samplers.md) closes it.

## The parts { #the-parts }

| Part | Chapters | What it buys you |
|---|---|---|
| **0 — Orientation** | [1](01-orientation.md) | The map, and the sealed envelope |
| **I — The mathematical spine** | [2](02-derivatives.md)–[8](08-probability.md) | Calculus → Jacobians → saddles → Fourier → Bayes |
| **II — The physical universe** | [9](09-units.md)–[12](12-spectroscopy.md) | Arcseconds, galaxies, PSFs, drizzle, redshifts |
| **III — Cosmology** | [13](13-expansion.md)–[15](15-distances.md) | Expansion, FRW, and distances that do not add |
| **IV — Gravitational lensing** | [16](16-deflection.md)–[21](21-degeneracies.md) | Deflection → lens equation → caustics → $\gamma$ → degeneracies |
| **V — This repo's science** | [22](22-inference.md)–[26](26-the-saddle.md) | The forward model, the samplers, and both spines |
| **VI — Discovery** | [27](27-discovery.md)–[28](28-the-label.md) | CNN finders, the resolution wall, and the label |
| **VII — Synthesis** | [29](29-how-to-read.md) | How to read the reports; where the frontier is |

Part III is the most detachable in the guide: the lens-modelling likelihood in
this repository contains **no cosmology at all** — it is entirely angular, and
cosmology appears in exactly three files. That proportion is a finding, not an
omission. Read Part III before your first meeting with an astronomer, not before
Chapter 16.

## Two things this guide will not do { #honesty }

**It will not pretend the repo is always right.** The campaign's own ledger
retracts a celebrated $\chi^2 = 0.451$, and a "science confirmed at
$\gamma = 1.27$" that inverted twice. Those retractions are the best pedagogy in
the building and the guide uses them.

**It will not pretend to be short.** Starting from calculus and ending at a
correlated-noise likelihood is a real distance. Budget roughly three to four
pages an hour if you actually work the examples, which is the only way this
works.
