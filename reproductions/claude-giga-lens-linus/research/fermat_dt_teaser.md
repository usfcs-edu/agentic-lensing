# Fermat Δφ noise-model sensitivity teaser — shift is ~60–90%, i.e. ~60–90× the TDCOSMO-relevant 1% threshold

**Campaign**: claude-giga-lens-linus, PLAN §6 cross-pollination track ("Fermat Δt teaser").
**Cost**: 0 GPU-h (budgeted <1; pure-CPU bijector transforms + analytic EPL deflections).
**Date**: 2026-07-15.

## ILLUSTRATIVE ONLY — read this first

- **DESI-165.4754-06.0423 is NOT a time-delay lens.** No source variability, placeholder
  redshifts. Nothing here is a Δt prediction for a real system.
- **Image positions are SYNTHETIC**: 4 fixed points on a θ_E=2.655″ circle about the anchor
  mass center (−0.039, 0.092)″ at PA 0/90/180/270 (the plan-suggested configuration), the same
  points for every posterior. β = each sample's Sersic source center.
- The primary comparison pair also differs by **data product** (native 80² vs binned 130²),
  not only noise model — the same-product secondary arm isolates the noise model.
- The correlated binned-low posterior is the P1c product **known to over-correct** (H2
  verdict: γ=1.103 is 17σ below the 1.433 anchor). This teaser therefore demonstrates
  *sensitivity* of Fermat observables to the coadd-covariance modeling choice — it does
  **not** calibrate which answer is right.

## The number

Fractional shift in median Fermat-potential differences Δφ between image pairs when swapping
the noise model, vs each posterior's own Δφ width:

| arm | median \|frac shift\| over 6 pairs | opposite pairs (well-conditioned) | in units of σ(Δφ) |
|---|---|---|---|
| **primary**: diag-native anchor (γ=1.433) → corr-binned-low SMC (γ=1.103) | **88%** | −87…−88% | ~10.7σ |
| **secondary, same product**: diag-binned-low (γ=1.293) → corr-binned-low (γ=1.103) | **61%** | −61…−63% | ~17σ |

Verdict against the pre-registered question ("is the noise-model-induced shift ≳1%,
TDCOSMO-relevant, or negligible?"): **≳1% by a factor of ~60–90 — far beyond negligible.**
Even the cleanest apples-to-apples arm (same binned product, diagonal → correlated drizzle
covariance) moves Δφ by ~61% while each posterior's internal Δφ precision is 4–8% — a
~17σ systematic relative to the statistical error a TD analysis would quote.

Representative numbers (opposite pair A–C, arcsec²): diag-native −0.344 ± 0.028;
diag-binned-low −0.107 ± 0.004; corr-binned-low SMC −0.040 ± 0.004. Full per-pair table in
`data/fermat_dt_teaser.json` (adjacent pairs with near-zero medians give unstable fractional
ratios, e.g. D–A; the median-over-pairs and opposite-pair numbers are the meaningful ones).

Physical reading: Δφ between images tracks (γ−1) (γ→1 is the uniform-sheet limit where Fermat
differences collapse), so the correlated likelihood's γ deflation 1.433/1.293 → 1.103
propagates into a ~4–9× collapse of Δφ. Noise-model-induced slope shifts translate ~one-to-one
into Fermat/H0-relevant observables; TDCOSMO's error budget currently has **no line item for
coadd/drizzle noise covariance**, and on this real HST product the choice moves Δφ by tens of
percent — the teaser's takeaway is that such a line item is worth pricing on an actual TD lens.

## Method

- Fermat potential φ(θ;β) = |θ−β|²/2 − ψ_EPL(θ) − ψ_shear(θ), per posterior sample.
- α_EPL: numpy port of the vendored gigalens `EPL.deriv` Tessore–Metcalf angular series
  (niter=50), **validated against the vendored jax implementation to max|diff|=1.3e−15**;
  ψ_EPL via the exact EPL Euler/homogeneity identity ψ = θ_rel·α/(3−γ), FD-gradient-checked
  (∇ψ = α to <1e−5). ψ_shear = ½γ₁(x²−y²) + γ₂xy. Fractional Δφ shifts are
  distance/redshift-free.
- Posteriors:
  1. `foundry-i/data/hmc_v13_v2d.npz` (74-dim paper-model chains, 20k subsample; transform
     validated against the stored physical arrays, γ med 1.4330);
  2. `e2_v3b_low_smc_canary_fix.npz` — P1c converged correlated SMC, 128 equal-weight
     particles through the 46-dim marg-model bijector (`cgl.likelihood._build_prior`), γ med
     1.1032 = the P1c money number (validated);
  3. `foundry-i/data/hmc_v13_v3b.npz` diagonal binned, low basin (γ<1.7 selection, 93.75% of
     draws, 20k subsample), γ med 1.2932.
- SMC particle files were not on local disk; retrieved from the previous campaign's Perlmutter
  staged copy (`~gdbenson/claude-giga-lens/repo/reproductions/claude-giga-lens/data/results/`)
  — a zero-GPU file copy of an existing campaign artifact.

## Caveats beyond the disclaimer

- 128 SMC particles (77 unique) → the green histogram in the figure is granular; its Δφ σ is
  a ~±10% estimate. Irrelevant at 60–90% shifts.
- Fixed synthetic image positions do not track each sample's own critical curve; a real TD
  analysis solves the lens equation per sample. For a sensitivity teaser the fixed-θ
  convention is standard and conservative (it removes image-position jitter from the widths).
- Shear potential is centered at the grid origin (gigalens convention); consistent across all
  posteriors, and only Δφ differences are used.

## Outputs

- `data/fermat_dt_teaser.json` — per-pair Δφ (median/σ/quantiles) for all three posteriors,
  both shift arms, headline numbers, disclaimers.
- `figs/fermat_dt_teaser.png` — Δφ posteriors for an opposite and an adjacent pair, three
  noise-model posteriors overlaid.
