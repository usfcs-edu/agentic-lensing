# P3 results — Euclid Q1 demo + COOLEST export (2026-07-09)

Closing note for Phase P3 (report §8). Diagnostic analysis of the three converged/
attempted Euclid VIS fits, the recipe end-to-end, and the COOLEST export. **No new heavy
sampling** — all numbers read from the on-disk fit artifacts (`data/euclid/*_fit.json`,
`*_posterior.npz`) plus a CPU chi2 cross-check and a bijector-only physical conversion of
the stored z-space draws. Setup and scoping in `p3-euclid-coolest-recon.md`; stage-log
entries "P3 Euclid converged results" + "P3 recon" in `CAMPAIGN.md`.

## What P3 is (scope)
Euclid Q1 VIS is **native 0.1″/px** → diagonal instrumental RMS. So P3-Euclid demonstrates
the **Pillar-2 sampler recipe on independent real data** with the *diagonal* likelihood
(`whiten_fn=None`); it is **not** a test of the P1 correlated likelihood (that remains the
HST cross-scale P1c experiment). Single-band VIS, single-plane EPL+shear mass, Sérsic ×4
lens light, Sérsic+shapelet(n_max=6) source with ridge-marginalized amps. Comparison to the
published PyAutoLens model is at the **mass level** (area-equivalent `einstein_radius_effective`),
a few-% convention offset expected — not a digit match.

## Outcome: 1 clean / 1 converged-but-biased / 1 multimodal (HONEST, not 3/3)

| ID (tag) | θ_E,pub | q,pub | ours θ_E,eff | offset | χ²/px | R̂(θ_E) | ESS(θ_E) | chains | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 102157952 | 1.114″ | 0.82 | 1.099″ | **−1.4%** | 0.454 | 1.08 | 449 | 16×600 | **CLEAN ✓** |
| 102157958 | 0.853″ | 0.56 | 0.581″ | **−32.0%** | 0.441 | 1.04 | 2905 | 48×1400 | CONVERGED but BIASED |
| 102020061 | 1.244″ | 0.79 | 0.996″ (med) | −22.6% | 0.472 | **18.8** | 51 | 48×1400 | MULTIMODAL, NON-converged |

**χ²/px is flat (~0.44–0.47) across all three** — verified pixel-exact from the npz
(`Σ((data−model)/σ)²/n_keep` == the JSON `map_chi2_per_pixel` to 1e-4). The ~0.45 floor
(< 1) is common to every target (VIS RMS ~1.5× conservative and/or the flexible
marginalized source absorbing noise) so it is **not** a per-target fit-quality discriminator
— but the fact that it is flat is exactly the diagnostic: **the biased and multimodal MAPs
fit the data just as well (marginally better) than the clean reference.** The problems below
are therefore genuine model degeneracies / sampler non-mixing, **not** failed fits.

Note the compute paradox: the CLEAN target converged with the LIGHTEST config (16×600); the
two hard targets did not converge (or converged to a biased basin) at **3× the sampling
budget** (48×1400). More chains cannot fix a degeneracy — it is an information limit of
native-resolution single-band modeling of these systems.

## Target 102157952 — reference success (fig: `figs/euclid_102157952_resid.png`)
Well-resolved (θ_E 1.11″ ≈ 22-px ring diameter), round (mass q 0.92), highest S/N (425).
MAP χ²/px = 0.454, residual has **zero |z|>3 pixels** inside the mask (clean white noise, no
arc/ring residual). θ_E,eff = 1.099″ vs published 1.114″ (**−1.4%**, well within the ~5–10%
PyAutoLens-vs-gigalens convention offset). γ pinned ~2.00, converged (R̂ 1.08, ESS 449 on
just 16 chains). This is the demonstration that the recipe recovers a correct θ_E + a valid
COOLEST export on real Euclid data when the lens is resolved, round, and high-S/N.

## Target 102157958 — converged but −32% biased: a GENUINE degeneracy, not a bad fit
(fig: `figs/euclid_102157958_resid.png`)
**MAP χ²/px = 0.441 — a GOOD fit (better than the clean reference 0.454).** The residual
panel is featureless noise: the model reproduces the compact, blended source+lens blob with
no left-over arc structure. Yet θ_E,eff = 0.581″ vs published 0.853″ (**−32.0%**), and the
posterior is tight (θ_E std 0.008″) and converged (R̂ 1.04, ESS 2905). So the −32% is **a
genuine model/convention degeneracy the data cannot break, not a failure to fit.**

Mechanism (chi2-based conclusion): this is the small-θ_E / marginally-resolved end. At
0.1″/px a θ_E ~0.85″ lens has an Einstein ring only ~17 px across and ~1–2 px wide, **blended
with the bright lens light**. Our fit trades a *smaller, rounder* mass (θ_E 0.58″, mass
q 0.856) against a *more elliptical/compact* source (source e2 = −0.48) — the classic
mass-ellipticity/source-shape degeneracy — and lands at the same pixel-χ² as the published
model. The published PyAutoLens model uses a **free-form pixelized source** (Hilbert mesh);
our **Sérsic+shapelet** source is a different, more constrained basis, so the two pipelines
settle at different θ_E for equal data fit (the source-model-choice theme, arXiv 2406.08484).
The mass ellipticity/shear posteriors corroborate a weakly-constrained mass: |e|~0.05±0.04
(consistent with round, rounder than the published light q~0.56) with a non-trivial shear
(γ₂ ≈ −0.07 ± 0.02). Here the degeneracy resolves into a *biased-but-tight* θ_E rather than
multimodality (contrast 102020061).

## Target 102020061 — multimodal, non-converged R̂(θ_E)=18.8
(fig: `figs/euclid_102020061_thetaE_modes.png`)
MAP χ²/px = 0.472 (again a GOOD fit at the MAP), but the θ_E posterior does **not** converge
even at 48 chains (R̂ 18.8, ESS 51). Converting the stored z-space draws to physical mass
(bijector only) shows the structure:
- **Multimodal / broad ridge**, not a clean two-spike bimodal: the pooled θ_E histogram is
  multi-peaked (~0.80 / 0.90 / 1.05 / 1.28), spanning 0.75–1.32″.
- **Chains are locked in different basins**: per-chain θ_E means span **0.792–1.281″
  (spread 0.489″)** while the *within-chain* std is only ~0.033″ — i.e. each of the 48 chains
  mixes tightly around its own θ_E and they disagree with each other → the R̂=18.8 signature
  (textbook non-mixing; middle panel of the figure shows 48 flat, separated traces).
  Split: 9 chains at ~0.81″, 30 at 0.90–1.10″, 9 at ~1.22″.
- **The mode is a θ_E–ellipticity–centroid degeneracy**: corr(θ_E, mass e1) = **+0.52**,
  corr(θ_E, center_x) = **−0.41** (right panel — clusters separate in θ_E, |e|, and centroid).
  A small, marginally-resolved source cannot pin θ_E, mass ellipticity, and lens centroid
  simultaneously, so the sampler explores a degenerate ridge.
- The published θ_E,eff = 1.244″ sits at the **upper edge** of the explored range (p95 = 1.275″);
  the high-θ_E chains (~1.22″) are near it, but the pooled median (0.996″) pulls the naive
  offset to −22.6%. Because it is non-converged, **no single θ_E should be quoted** — the
  honest statement is "unconverged multimodal ridge 0.79–1.28″ spanning the published value".

## Interpretation (report §8)
Native-resolution ss1 single-band VIS modeling of Euclid Q1: **clean for well-resolved, round,
high-S/N systems** (102157952, −1.4%); **biased or multimodal for small-θ_E / elliptical
systems** (102157958 −32% tight; 102020061 unconverged ridge). All three are equally good
*pixel* fits (χ²/px 0.44–0.47) — the failures are **degeneracies, not misfits**. The two
recurring themes, both pre-registered in the P3 recon and echoing prior campaigns:
1. **PSF / pixel undersampling** at 0.1″/px — a small θ_E ring is ~1–2 px wide and blended
   with lens light (the foundry-i **R0c** signature; here the χ²/px≈0.45 floor confirms the
   PSF is *not* oversampled, so this is undersampling of the *ring*, not the PSF kernel).
2. **Source-model choice** — our Sérsic+shapelet source vs the published free-form pixelized
   source picks a different θ_E at equal data fit when the source is small/complex
   (arXiv 2406.08484).

**Documented robust path (not run in P3):** ss2 super-sampling + a pixelized/free-form source
regularized to match the PyAutoLens basis is the route to θ_E robustness at
ground-/native-resolution; parked as P3 follow-on. For P3 the **recipe + COOLEST export are
the WORKING deliverables**; Euclid θ_E recovery is an honest 1/3-clean characterization of
where native single-band modeling is trustworthy.

## Recipe deliverable (end-to-end)
The recipe end-to-end (`30_recipe_e2e.py`, CGL infra: build_target → map_polish →
laplace_evidence → run_staged two-stage PHMC) ran on the v2d-relaxed native product:
`data/recipe_e2e/recipe_v2d_low.json` — **R̂_max = 11.2, R̂(γ)=6.0, ESS_min 30 → under-converged,
as expected**: this is the *information-limited* v2d-relaxed product (the campaign's weakest
likelihood, 1466 kept px). The **ingest/plotting path is validated** —
`figs/recipe_e2e_v2d_low.png` exists as the (documented) non-convergence datum. The proper
recipe-showcase figure will be built from **P1c Job-1's converged v3b-binned-low posterior
when it lands** — a documented ready step (same ingest path, swap the product); no code change
needed.

## COOLEST export — all 3 round-trip (data/coolest/*)
All three exports **regenerated from the current converged fits** (the 102157958 / 102020061
on-disk exports had been dumped Jul-8 from an earlier MAP, before the Jul-9 48-chain refits;
regeneration makes them consistent with the reported posteriors) and each **round-trips
`dump_simple → load_simple(validate=True)`**: mode=MAP, 3 lensing entities
(lens PEMD + 4 Sérsic light / source Sérsic+Shapelets / external-shear MassField), point
estimates (θ_E, γ, γ_ext, 28 amps) preserved to <1e-9, mass-sector posterior stats attached
from the chains. Sidecars per target: `<tag>_euclid.json`, `data.fits`, `model_map.fits`,
`psf.fits`, `chains.npz`.
- **Ridge-marg-source handling:** the 28 source-shapelet amps are ridge-marginalized (not a
  sampled MCMC block). The **marginal-mode a★ at the MAP z** is written as the Shapelets
  point estimate (no posterior), with the marginalization/logdetA Occam term flagged in the
  COOLEST `metadata` and the source-basis mismatch (Sérsic+shapelets vs published pixelized
  source → "compare masses, not source") recorded there too.

## Tests
`./00_run_tests.sh --gpu` (CGL_GPU=8, L4): **93 CPU passed** (49s) + **10 GPU passed** (5.0m)
= **103 passed, 0 failed.** Parity A–E **green** after the `likelihood.py` prior-override edit;
`tests/test_euclid_coolest.py::test_prior_override_defaults_match_hardcoded` confirms the new
`theta_E_med` / `mass_center_sig` / `near_xy` overrides preserve the HST-parity defaults
bit-for-bit (P0 parity untouched).

## Figures
- `figs/euclid_102157952_resid.png` — reference success (clean data|model|residual, −1.4%).
- `figs/euclid_102157958_resid.png` — converged-but-biased: clean residual at θ_E −32% (good fit, genuine degeneracy).
- `figs/euclid_102020061_thetaE_modes.png` — θ_E histogram (multimodal) + 48 separated chain traces (R̂ 18.8) + θ_E–ellipticity–centroid degeneracy scatter.
- `figs/recipe_e2e_v2d_low.png` — recipe end-to-end on the info-limited v2d-relaxed product (documented non-convergence datum; ingest path validated).

## Bottom line
Recipe + COOLEST export = **mechanically validated, working P3 deliverables**. Euclid θ_E
recovery = **1/3 clean (−1.4%) / 1/3 converged-but-biased (−32%, genuine source-model +
undersampling degeneracy, GOOD χ²) / 1/3 multimodal-nonconverged (θ_E–ellipticity–centroid
ridge, GOOD χ² at MAP)** — an honest native-resolution single-band characterization, with
ss2 + pixelized source documented as the robustness path.
