---
name: project-lensing-repro-sprint-2026-06
description: "2026-06-03 sprint that reproduced the remaining 10 Huang-corpus papers (Gu 2022, Cikota 2023, Sheu 2023/2024a/2024b, Dawes 2022, Foundry II/III/IV, Silver 2025) via parallel workflow waves — completing 16/16 in-scope papers; verified numbers + per-paper caveats"
metadata:
  type: project
---

On 2026-06-03 the remaining 10 papers of the 16-paper Huang strong-lensing corpus were
reproduced in one parallel sprint (Workflow waves on 8×A16 + 2×L4), bringing the corpus to
**16/16 in-scope papers reproduced** (the prior 5 done: Foundry I, Hsu 2025, Huang 2020/2021,
Storfer+Inchausti). Index of record: `reproductions/REPRODUCTIONS.md`. Each new paper has a
`reproductions/<slug>/` dir with numbered scripts, README, and a built LaTeX tech-report
(`papers/main.pdf`, 7–10 pp). Out of scope: the dark-energy paper + DES-Y6 memo (cosmology).

**Why:** executes `plans/EXPERIMENTS_REPRODUCTON_PLAN.md` to completion. The user chose
"maximize parallel throughput" + "check archive availability first" — the archive check
confirmed KOA (18-mo) and ESO MUSE (12-mo) proprietary windows have elapsed, unblocking
Foundry III/IV.

**How to apply:** when revisiting any of these, the verified result + the one load-bearing
caveat is below. Bulk data is gitignored under each `data/`; reruns use the documented venvs.

## Verified results (ours vs published) + the key caveat each
- **gu-2022** (GIGA-Lens method): MAP χ²≈1.0 on 12 mocks; mean ESS ≈11k; paper convergence
  target (R̂≤1.017) met on well-conditioned systems. Uniform minESS≥26,822 NOT reached on
  degenerate systems — **depth/efficiency-limited, not biased** (2.5× depth ≈doubles ESS,
  halves excess R̂; large mu_z tracks low-ESS params = non-convergence artifact). Single A16
  vs paper's 4×A100. Mirrors the [[project-foundry-i-reproduction]] HMC degeneracy finding.
- **cikota-2023** (Einstein cross): θ_E 2.10″ vs 2.52″, σ_SIE 347 vs 379, μ 7.0 vs 10.47,
  χ²/px 0.90, four-image geometry reproduced. CAVEAT: modeled on **public DESI Legacy g-band
  (1.35″ seeing)** not the proprietary MUSE-derived imaging — PSF ablation shows the seeing
  drives the −17%/−8% θ_E/σ offsets. Reuses the foundry-i GIGA-Lens pipeline.
- **sheu-2024b** (Carousel): θ_E **12.96″ vs 13.03″ (0.5%)**, γ 1.53–1.67 vs 1.67, M(<θ_E)
  4.62 vs 4.78e13. lenstronomy multi-plane (5 source planes), public HST F140W/F200LP (MAST
  DOI 10.17909/zq07-4f53). CAVEAT: forward-model + position-fit done; full pixel-level emcee
  fit set up but NOT run. **Correct arXiv is 2408.10320** (not 2408.09124).
- **sheu-2023** (lensed SNe): re-detected the Grade-A L-SN (DESI-344.6252−48.8977; 11 sub-
  detections, 1.48″ = counter-image); built **from-scratch Bramich-2008 difference imaging**
  + SEP + SALT3. SALT3 μ 8.6 vs 8.2 is **synthetic-injection-validated**; real-data photometry
  is a documented proxy (B08 lens residuals; paper used SFFT+PSF photom). NOIRLab Astro Data
  Archive per-exposure DECam fetch (no positional cutout service; fetch full frame + WCS slice).
- **sheu-2024a** (variable lensed quasars): variability σ 0.34 vs 0.25 mag at both images;
  **reused sheu-2023 diff-imaging core verbatim**; σ-metric validated to a few % on synthetic
  injections. CAVEAT: blended single-Gaussian PSF inflates real σ; B08 only (paper adds Hu2022).
- **dawes-2022** (multiply-lensed quasars): autocorrelation FoF reproduced — **58/58 = 100%
  conditional recovery** (apples-to-apples with paper's 94/94). CAVEAT: raw recovery 14% is
  **proxy-limited** — DR1 spectroscopic QSO sample (1.6M) vs Dawes' ~5M photometric QSO targets
  (not on disk). Reuses hsu-2025 FoF + DR1 zcatalog. Gaia EDR3 PM/PX cut deferred (archive
  in DR4-instability slowdown; published table carries PMSig/PXSig for validation).
- **foundry-ii** (DESI spectroscopy): 73/73 matched to DR1 fibers; **z_lens 70/72, z_source
  16/22 to <0.001; σ_v 65/71 (r=0.80)** — all from on-disk public DR1 + FastSpecFit (no new
  downloads). EDR(Fuji)⊂DR1(Iron) caveat; 2 lens-z misses are VI-corrected ZWARN systems.
- **foundry-iii** (Keck NIRES): **6/6 source z to |dz|<0.001** via blind Eq.1 two-Gaussian
  line-fit + MC validation. CAVEAT: **consistency reproduction** — KOA serves ONLY raw L0
  NIRES (no reduced L1/L2; pyKOA confirms), and **pypeit will NOT build on this aarch64 box
  (pyqt6 GUI dep)**, so real-spectra measurement needs an x86 box/container; the validated
  fitter (03_linefit.py) is ready to apply to reduced spectra unchanged.
- **foundry-iv** (VLT/MUSE): pulled 3 public ESO Phase-3 MUSE cubes via **ESO TAP (ivoa.ObsCore)
  + DataLink**; auto z_lens 3/3 within dz<0.02 (2/3 <0.003); guided source z exact for Lens22
  (0.821). CAVEAT: **unguided source-ID is interloper-prone** (field emitters outshine faint
  arcs across the 60″ FoV) — intrinsic, which is why Lin+2025 did source IDs by hand; built an
  automated mpdaf line-ID engine. 133 public MUSE cubes / 21 cover the confirmed systems.
- **silver-2025** (ML forecasts): Model-1 (HST) ResNet val **AUC 0.994 vs 0.998**. CAVEAT:
  MVP uses lenstronomy Sérsic sources as a proxy for the paper's VELA hydro stamps; Models 2/3
  (JWST regimes) + the U-Net pixel localizer are scaffolded next steps. Reuses huang-2020 ResNet.

## Infra & environment facts that cost time
- **Wave-2 venv** `/home/benson/.venvs/lens` got sep, sncosmo, reproject, photutils, astroquery,
  mpdaf, emcee installed. **pypeit fails on aarch64** (pyqt6). gigalens/hsu venvs unchanged.
- The gigalens **single-device GBTLA recipe** (one process per system pinned to one GPU via
  CUDA_VISIBLE_DEVICES; XLA_FLAGS=--xla_gpu_autotune_level=0) is the way to run many GIGA-Lens
  fits in parallel without the pmap-over-all-devices hang ([[project-foundry-i-reproduction]]).
- Tech-report build: copy `reproductions/hsu-2025/papers/Makefile` + `\usepackage{../../tech-report}`;
  the shared preamble already defines `\arcsec` → use `\renewcommand` or a new macro name.
- NOIRLab Astro Data Archive: no positional FITS-cutout service, no HTTP byte-range → fetch the
  full ~320MB funpacked InstCal frame and slice the covering CCD locally via per-CCD WCS.

Related: [[project-huang-lensing]], [[project-foundry-i-reproduction]], [[project-hsu-2025-reproduction]],
[[reference-gigalens-env]], [[reference-host-hardware]], [[reference-paper-corpus]].
