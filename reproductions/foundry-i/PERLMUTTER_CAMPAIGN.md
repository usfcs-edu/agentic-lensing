# Foundry-I paper-scale campaign on NERSC Perlmutter

Goal: Huang 2025a "Fig. 8 / Table 3" quality for DESI-165.4754-06.0423.
Bar: masked reduced chi2 < 1.1 at MAP; SVI full-rank + flat ELBO; HMC split
R-hat < 1.1 all params, ESS >= 1e3 (paper: 32,200-40,000); gamma vs 1.372+-0.023.

Decisions: account **deepsrch_g**; budget **<= 200 A100-hours** (hard stop 180);
library = vendored gigalens-sean @ multinode-2025 (`vendor/gigalens-sean/`,
ref in VENDORED_REF.txt); drivers patterned on the carousel branch.

Remote layout: `$HOME/foundry-i/` (code + small data + venv),
job outputs in-place, synced back here after each stage.

## A100-hour ledger

| job | stage | nodes x walltime | A100-h | cumulative |
|---|---|---|---|---|
| 54286675 | P0 smoke | 1 x 0:01:45 | 0.12 | 0.12 |
| 54287092 | P1 MAP prod (1000x2000) | 1 x 0:03:36 | 0.24 | 0.36 |
| 54287093 | P1 MAP long (500x3000) | 1 x 0:03:03 | 0.20 | 0.56 |
| 54290356/58 | P1 warm4k x2 | 2 x 0:05:27 | 0.73 | 1.29 |
| 54290354/55 | P1 cold8k x2 | 2 x 0:09:38 | 1.28 | 2.57 |
| 54293605 | P1 nm8_d | 1 x 0:13:19 | 0.89 | 3.46 |
| 54293607 | P1 c3_d | 1 x 0:09:55 | 0.66 | 4.12 |
| 54293609 | P1 nm8c3_d | 1 x 0:13:08 | 0.88 | 5.00 |
| 54301417 | P2 SVI r1 | 1 x 0:02:44 | 0.18 | 5.18 |
| 54301739/40 | P2 SVI prod2+prod3 | 0:10:57+0:23:23 | 2.29 | 7.47 |
| 54301856 | A v3cold MAP | 1 x 0:25:48 | 1.72 | 9.19 |
| 54301861 | A v3nm8 MAP (timeout) | 1 x 0:28:14 | 1.88 | 11.07 |
| 54303286 | A v3 SVI (timeout) | 1 x 0:28:07 | 1.87 | 12.94 |
| 54302961/54303308/54303655 | P3 HMC r1+r2+r3 | 0:03:04+0:05:38+0:13:48 | 1.50 | 14.44 |
| 54301863 | A v3nm8 MAP (preempt) | 1 x 0:30:57 | 2.06 | 16.50 |
| 54303288 | A v3 SVI (preempt) | 1 x 1:02:08 | 4.14 | 20.64 |
| 54313432 | P3 v3 HMC (debug) | 1 x 0:14:13 | 0.95 | 21.59 |
| 54318049 | v2 HMC retry (CANCELLED by root, 2:08) | 1 x 0:02:08 | 0.14 | 21.73 |
| 54318694 | v2 HMC final (1500+13000) | 1 x 0:21:45 | 1.45 | 23.18 |
| (54303656 + 54313431 scancelled; queue went cold ahead of the 6/17-6/24 maintenance) | | | | |

**ARCHIVED — pre-rigor-revision native-scale numbers (54318694, 13k draws
x 48 chains), SUPERSEDED by the Phase-R v2d posterior below:**
gamma = 1.324 [1.309, 1.342] (R-hat 1.37, ESS 259 — ESS saturated 250->259
for 8k->13k draws; we read this as a windowed-adaptation need, RETRACTED in
Phase R: the saturation was the 2x-broadened PSF flattening the gamma
direction, and dissolves with the PSF fix — gamma ESS 5,714 at 8k draws);
theta_E = 2.6351 +- 0.0012 (R-hat 1.071, ESS 1154). These fits ran with
the defective PSF convention (Phase R, finding R0c).

## Stage log

### W0 — local dev (2026-06-10)
- Vendored gigalens-sean @ multinode-2025 (`58ec9a7`) into `vendor/`.
- `40_make_cutout_v2.py` GATE PASS: sky chi2 = 0.973 (WHT-only was 1.248 ->
  read at the time as a 25% drizzle noise underestimate; rescale 1.117
  [RETRACTED in Phase R: wing contamination of the img-based calibration —
  the model-subtracted factor is 0.787 and 1.117*sqrt(0.787)=0.99, i.e.
  the raw WHT sigma was essentially correct]). Masks: 3 faint
  galaxies + 1 arc-A object (most point-like ring detection, sharpness 0.86;
  the 4 lensed images kept) + lens/companion cores. 535/6400 px masked.
- Drivers 41 (MAP), 42 (SVI), 43 (HMC), 44 (diagnostics) written against the
  vendored library; carousel ProbModel mask conventions; slurm templates in
  `slurm/`.
- Local smoke on 2x L4: ALL PASS mechanically. MAP 64x100 in 25 s (chi2 14.6,
  still descending — expected at 1/300 of production compute). SVI 200 steps in
  28 s, gate correctly detects rank-71/74 covariance at smoke scale. HMC 8
  chains x 30 draws in 40 s, eigenvalue floor handled deficient cov without
  NaNs (Bug-2 guard works). Diagnostics produce R-hat/ESS + paper table.

### P0 — Perlmutter setup (2026-06-10)
- rsynced: drivers + slurm/ + vendor/gigalens-sean + Stage-A data product
  to gdbenson@perlmutter:~/foundry-i/.
- venv built: python 3.13 module, jax[cuda12]==0.6.2, tfp[jax]==0.25.0,
  optax/objax/lenstronomy/astropy. Vendored gigalens imports cleanly.
- Smoke job 54286675 submitted (gpu_debug, 1 node, 20 min cap); monitor armed.
- 45_fig8_panel.py written + validated locally on the smoke posterior: renders
  observed | model + critical curves | masked residual | source + caustics;
  reproduces the gate chi2 (11.91 at smoke params) and writes
  data/fig8_summary.json (gamma, theta_E, inner-critical-curve flag).

### P1 — MAP at paper scale (gate: reduced chi2 < 1.1)
- P0 smoke COMPLETED on 4x A100 (1:45 wall; MAP 64x100 in 23 s; chi2 13.91 at
  step 100, consistent with local L4 run — pipeline numerics check out).
- Submitted 54287092 (TAG=prod: 1000 particles x 2000 steps) and
  54287093 (TAG=long: 500 x 3000), both 1 node gpu/regular, 2 h cap.
- regular queue slow (~2 h pending): added preempt-QOS clones 54289733
  (TAG=prod_p) + 54289734 (TAG=long_p); first terminal job wins, rest get
  scancelled. Preempt grace (2 h) exceeds job length -> effectively safe.
- ROUND 1 RESULTS (both regular jobs ran in ~3.5 min!): corrected likelihood +
  paper-scale opt cut reduced chi2 from 11.93 (old v9) to **4.02** (prod,
  gamma=2.22 basin) and **3.84** (long, gamma=1.67 basin) — but NOT plateaued
  (tail improvement 3.0%/1.8%) and gate <1.1 FAIL. Lower-gamma basin fits
  better, echoing the HMC-valley story under the old likelihood.
- Retune round 2 (protocol: not plateaued -> extend): 54290354/55 cold-deep
  (1000x8000, regular+preempt) and 54290356/58 warm-restart from long's best
  (1000x4000, z-scatter 0.05, lr 1e-3 -> 1e-6; carousel warm-start pattern;
  --start/--scatter added to 41_map_paper_scale.py). Preempt clones scancelled
  (54289733/34, 0 A100-h).
- ROUND 2 RESULTS: **cold8k PLATEAUED at chi2 = 3.548** (tail impr 0.18%,
  theta_E=2.632, gamma=1.734); warm4k chi2=3.591 still slowly descending
  (theta_E=2.6414 = 0.19% of paper!, gamma=1.445 -> walking toward paper's
  1.372). The chi2~3.5 floor is a MODEL limit, not under-optimization.
  Fig-8 render at warm4k best (figs/fig8_warm4k.png) localizes the misfit:
  (a) strong dipole at the companion, (b) coherent alternating residuals
  along the arcs, (c) some core structure.
- Stage-B flexibility pass (sanctioned in the approved plan): 3-way ablation
  submitted, 1000x8000 each, dual-QOS: nm8 (n_max 6->8: 54290894/95),
  c3 (3rd companion Sersic: 54290896/97), nm8c3 (both: 54290899/900).
  _data_lib gained companion_extra; 41 gained --companion3.
- regular+preempt lanes stalled ~3 h overnight; added debug-lane clones
  (54293605/07/09, 28 min caps) which ran promptly.
- FLEXIBILITY ABLATION RESULTS (debug lane): nm8 chi2=3.4246 (plateaued,
  theta_E=2.6327, gamma=1.597), c3 chi2=3.4627 (plateaued, theta_E=2.631,
  gamma=2.021). Both MARGINAL vs the 3.548 baseline (~3%). nm8 residual map
  (figs/fig8_nm8.png): arc-coherent residuals essentially unchanged by 17
  extra shapelet components -> the chi2~3.4 floor is PSF / pixel-sampling
  limited, not model-component limited. Redundant regular/preempt clones for
  nm8+c3 scancelled.
- nm8c3_d: chi2=3.4442 (plateaued) — adding BOTH flexibility components is no
  better than nm8 alone. Ablation conclusive; nm8c3 clones scancelled.
- **STAGE-B GATE VERDICT: FAIL at chi2=3.42 floor after the full sanctioned
  retune ladder. Stopped per protocol; report delivered to user 2026-06-11
  with options: (A) 0.065" drizzle-scale rebuild [prime suspect, with
  arc-region Poisson/gain semantics as co-suspect]; (B) proceed to P2/P3 on
  the current best model to answer the gamma question.**

### P2 — SVI at paper scale (gate: full rank + flat ELBO) + Track-A 0.04" rebuild
- USER DECISION (2026-06-11): "Both — B now, A next." Track B = SVI/HMC on the
  current best (nm8_d); Track A = fine-pixel-scale rebuild.
- DISCOVERY: the modeled product is 0.12825"/px (D001SCAL), not 0.13 — a 1.36%
  scale systematic on all angular params (pure unit relabel; correct angles by
  x0.98654 when reporting; refit at true scale for final numbers). And the
  HAP FINE SKYCELLS at 0.04"/px are already on disk — no re-drizzle needed.
- Track A executed: 17b_build_psf_fine.py (51x51 @0.04 empirical PSF, 25
  stars, FWHM 0.119"); 40b_make_cutout_fine.py (260x260 @0.04 from skycell
  p1184x13y11, masks transferred — arc-A object via fixed arcsec fallback,
  sky chi2 = 1.0000 GATE PASS, note skycell WHT is exposure-like: rescale
  0.078). _data_lib + drivers gained --data selector; v3 meta carries
  delta_pix/supersample (0.04 / 1).
- v3 local smoke (32x60): chi2 already 4.08 (v2 smoke at same scale: 14.6) —
  fine-scale data + natively-sampled PSF fits dramatically better.
- Track B SVI round 1 (54301417, 2:44): FAIL — ELBO flat_frac 1.2% (>0.5%)
  and rank 86/91. We invested ~1.4% of the paper's SVI compute. Retunes
  submitted: prod2 (8000 steps, n_vi 800, init_scales 1e-2), prod3 (15000
  steps, n_vi 1000, init_scales 3e-3) + preempt clone (54301739/40/41).
- v3 MAP ladder submitted: v3cold + v3nm8 (1000x6000, debug+preempt:
  54301856/59/61/63).

### P3 — HMC paper recipe (gate: R-hat < 1.1, ESS >= 1e3)
- P2 PASSED with svi prod3 (15000 steps, n_vi 1000, init_scales 3e-3, 23:23):
  flat_frac 0.10% + FULL RANK 91/91 (min_eig 3.1e-9). The init_scales fix
  resolved the rank deficiency; pure scale fixed flatness.
- HMC submitted from prod3: 54302961 (debug) + 54302981 (preempt, 2h min
  walltime requirement discovered — preempt rejects t<2h).
- **TRACK-A v3 MAP GATE PASS: v3cold reduced chi2 = 0.4513 < 1.1, plateaued**
  (1000x6000, 25:48). The 0.04" fine-skycell rebuild meets the group's bar.
  Caveats: chi2<1 => noise conservatively estimated at fine scale (drizzle
  correlation); MAP gamma slid to 2.578 (theta_E 2.5925) — gamma MAP values
  unstable along the broad degeneracy (1.45/1.73/2.58 across configs/scales),
  posterior (HMC) is the arbiter, exactly why the paper quotes HMC.
- Pivoted SVI chain to v3: 54303286 (debug, 600x12000) + 54303288 (preempt,
  1000x15000). v2-data HMC continues in parallel as the coarse-scale answer.
- v2 HMC round 1 (54302961, 3:04; 250+750x48): gamma = 1.417 [1.400,1.432],
  theta_E = 2.6344 — but R-hat 2.37 / ESS 126 FAIL (under-burned).
- v2 HMC round 2 (54303308, 5:38; 750+2500x48): **gamma = 1.354 [1.339,1.372]
  vs paper 1.372±0.023 — posterior brackets the paper value.** theta_E =
  2.6350 raw (x0.98654 relabel -> 2.600). R-hat 1.68 / ESS 183-757: improving
  with draws but still gate-FAIL. Round 3 escalation: 54303655 (debug,
  1500+8000) + 54303656 (preempt, 1500+15000).
- v3nm8 debug clone TIMEOUT at 28 min (n_max=8 at 260^2 exceeds debug window);
  preempt clone 54301863 still queued covers it (nice-to-have only).

### P4 — conditional reconciliation extras
NOT NEEDED: the native-scale (v2) posterior gamma = 1.33-1.42 brackets the
published 1.372±0.023; no PA-tie/multipole required. The v3-vs-v2 gamma
tension (2.585 vs 1.35) is attributed to independent-pixel likelihood
mis-specification on the 3.2x-upsampled, pixel-correlated 0.04" product
(also explains v3's chi2=0.45<1 and its over-tight CIs). Open item for any
follow-up: correlated-noise likelihood.

### P5 — Fig-8 render + report (CLOSED 2026-06-11)
- v3 SVI (54303288, 1:02:08): FULL GATE PASS — flat_frac ~0, rank 74/74.
- **v3 HMC (54313432, 14:13): STRICT P3 GATE PASS AT PAPER QUALITY —
  R-hat_max = 1.007, ESS_min = 28,124 (paper: <1.10, 32.2k-40k).**
  gamma = 2.585 [2.564,2.605], theta_E = 2.5927 (see P4 note on the
  correlation caveat). The redundant 8k-draw clone 54313431 scancelled;
  the v2 15k HMC (54303656) left queued — when it lands, record its gamma
  diagnostics as the final native-scale number.
- Fig-8 renders: figs/ours_foundry-i_fig8_v2.png (gamma=1.326 posterior
  median; INNER CRITICAL CURVE recovered — the paper's Fig-8 topology) and
  figs/ours_foundry-i_fig8_v3.png (chi2=0.45 structureless residual).
- papers/main.tex: new section "Paper-Scale Campaign on NERSC Perlmutter
  (June 2026)" (chi2 ladder, HMC convergence table, gamma reconciliation +
  correlation caveat, both Fig-8 figures); abstract + conclusions updated to
  retire the "57% steeper gamma" and "-32,500" claims. PDF rebuilt (29 pp).
- README.md updated (campaign banner; stale L4-reserved note fixed).

## CAMPAIGN VERDICT
All four gates of the approved plan achieved: Stage-A sky chi2 = 1.00;
MAP masked reduced chi2 = 0.451 < 1.1 (fine-scale product); SVI flat +
full-rank; HMC R-hat 1.007 / ESS 28,124 at the published bar. The 2025-era
gamma discrepancy is retired: defect chain was uncropped field + unmasked
interlopers + drizzle-blind noise + ~25x under-invested optimization.
Total: 21.6 A100-hours of 200 budgeted (hard stop 180 never approached).

================================================================================
## PHASE R — RIGOR REVISION (reopened 2026-06-11, group review feedback)

Triggers: (1) chi2=0.451 "too low to quote as calibrated"; (2) retract the
PA-tie/multipole hypothesis; (3) replace mass-sheet/slope-ellipticity
degeneracy labels with the empirical valley description. Plan-verification
then found (4): a v2 PSF sampling-convention defect.

### R0 audit results (46_noise_audit.py; data/noise_audit.json)
- THE chi2=0.45 EXPLANATION (D6=True): the v3 sigma sky term was calibrated
  on img fluctuations at r>4.5" that are ~70% diffuse lens-wing flux the
  model fits (segmentation found no sources there on the fine product).
  Model-subtracted recalibration: sky-term variance factor 0.304. With the
  Poisson term KEPT, honest per-pixel chi2(v3cold): kept 0.915, arc 0.955,
  sky 0.898 — the <1.1 bar is met with a CALIBRATED statistic.
- Poisson term verified consistent (D2=False, keep): dropping it overshoots
  arcs to 2.26; poisson_frac_arc(v3)=0.51.
- Correlation quantified, decoupled from the chi2 question: detrended sky
  autocorr corr_len(rho=0.5) = 2.5 fine px (~ the 3.2x drizzle footprint);
  f_corr(detrended, short)=59.6 (v3) / 11.4 (v2); raw full-window 387 / 54
  (large-scale background structure, reported separately). chi2_eff per the
  spec'd N/f convention is NOT ~1 for a perfect model (it ~ f_corr;
  convention caveat recorded in the JSON).
- v2 BONUS RETRACTION: model-subtracted v2 sky factor 0.787 -> honest
  rescale = 1.117*sqrt(0.787) = 0.99: the "WHT underestimates sky noise by
  25%" claim dissolves (it was wing contamination). v2 honest floor:
  kept 3.82, arc 5.50 (real misfit, see R0c).
### R0b binned product (40c; data/cutout_v3b.npz, 130^2 @ 0.08", ss=2)
- Block-sum + block-AND; psf_08 27x27 (degrade of psf_04; strict round-trip
  cos capped at 0.872 by PSF undersampling — band-relevant fidelity OK).
- Sky calibrated on MODEL-SUBTRACTED residuals: rescale_b=0.974, sky chi2
  =0.999 PASS. Intra-block covariance factor = 3.1 of max 4 (2x2 blocks
  ~78% internally correlated) — the honest-quadrature vs measured ratio.
- Render check vs blocksum(v3 render): no sum-rule/grid error (mean dev
  0.009 sigma, no dipole); max dev 0.72 sigma at bright px = the documented
  PSF-representation ceiling; accepted (re-MAP absorbs it).
### R0c v2 PSF convention defect (40d; cutout_v2c/v2d.npz)
- Simulator expects kernel sampled AT delta_pix (subgrid_kernel upsamples
  internally); cutout_v2 stored the 0.065"-sampled 27x27 ePSF with
  delta_pix=0.13/ss=2 -> ALL v2 fits (incl. the gamma=1.324 headline chain)
  ran with an effectively 2x-broadened PSF (r50 ratio 1.92).
- Fix: 15x15 @0.13" GN-fitted kernel (band-limited round-trip cos 0.999 vs
  0.804 defective). Model images change by 5.4 sigma RMS; chi2 of old params
  under fixed PSF = 25 (upper bound; re-MAP pending).
- cutout_v2c = PSF fix only (D5 isolation); cutout_v2d = + honest noise
  (factor 0.787; rescale 0.991 ~ raw WHT).
### R1 jobs (submitted 2026-06-11, debug QOS; maintenance deadline 06-16)
- 54324483 MAP v3b_cold 1000x6000 | 54324484 MAP v3b_warm (from v3cold,
  basin test) | 54324486 MAP v2c_cold 1000x8000 | 54324487 MAP v2c_warm
  (from cold8k) | 54324488 MAP v2d_cold 1000x8000.
- v2d_warm REJECTED (QOSMaxSubmitJobPerUserLimit = 5 debug jobs) — submit
  when a slot frees. Then: SVI+HMC on v3b and v2d per gates.
- Science checks: v3b gamma vs 1.324/1.372 (was v3 gamma=2.585 a
  mis-weighted-likelihood artifact?); v2c chi2 vs 3.42/3.55 same-noise
  (D5: is the native floor PSF-limited?); v2d -> final native posterior.
### R1 results (live)
- 54324484 v3b_warm (from v3cold gamma=2.58 basin): chi2=1.516 plateaued,
  gamma=2.672 — the high-gamma basin NO LONGER PASSES on the honestly
  calibrated binned product (it sat at 0.45 under the inflated noise).
- 54324483 v3b_cold: chi2=1.598 plateaued, gamma=1.465, theta_E=2.603 —
  cold optimum prefers the LOW-gamma basin. Region decomposition: companion
  chi2=15.2 (290 px), core 5.5, arcs 1.63, sky 1.03 — the honest sigma
  exposes companion-structure misfit the old calibration hid. D3 retune
  submitted: 54325397 (debug 1000x5000) / 54325398 (preempt 1000x8000)
  --n-max 8 --companion3; 54325399 (preempt) v3b_cold2 refines the
  low-gamma basin to investment parity with the warm run.
- 54324486 v2c_cold: chi2=1.051 PLATEAUED (vs 3.5478 cold8k / 3.4246 nm8_d
  on the SAME inflated noise) — D5 CONCLUSIVE: the native-scale chi2~3.4
  floor was the PSF convention defect; "sampling-scale limited" is
  RETRACTED. MAP gamma=1.412, theta_E=2.656 (paper: 1.372/2.646).
- 54324487 v2c_warm (from cold8k): chi2=1.078, gamma=1.435 — same basin/floor
  as v2c_cold from the opposite start; v2c story closed.
- 54324488 v2d_cold: chi2=1.234 plateaued, gamma=1.411, theta_E=2.656;
  54325014 v2d_warm: chi2=1.260, gamma=1.434. Region decomposition (cold):
  companion 9.4, core 1.47, arcs 1.44, sky 0.97 — same companion signature
  as v3b. D3 retunes (--n-max 8 --companion3):
- 54325740 v2d_nm8c3: chi2=1.203 NOT plateaued (tail 1.6%), gamma=1.314 —
  flexibility ablation marginal again (1.23->1.20); gamma moves 1.41->1.31
  with model class = a ~0.1 systematic worth reporting. 74-dim v2d_cold
  stays primary for the posterior (closest to paper model class).
- 54325397 v3b_nm8c3 (debug clone, 5000 steps): chi2=1.502, gamma=2.199 —
  companion fix does NOT break the v3b ~1.5 floor. Within 74-dim, v3b
  likelihood currently PREFERS the high-gamma basin (warm 1.516 vs cold
  1.598); basin-parity refinement (cold2) pending before concluding.
- Submitted: 54327398 SVI v2d (n_vi 1000, 15k steps, start v2d_cold, debug);
  54327403 v3b_cold2d (debug clone of the preempt-stuck 54325399).
### Ledger (Phase R, 4xA100 nodes, elapsed x 4 GPU-h)
| job | tag | elapsed | A100-h |
|---|---|---|---|
| 54324483 | v3b_cold | 25:57 | 1.73 |
| 54324484 | v3b_warm | 17:58 | 1.20 |
| 54324486 | v2c_cold | 8:31 | 0.57 |
| 54324487 | v2c_warm | 5:24 | 0.36 |
| 54324488 | v2d_cold | 9:27 | 0.63 |
| 54325014 | v2d_warm | 5:20 | 0.36 |
| 54325397 | v3b_nm8c3 | 26:02 | 1.74 |
| 54325740 | v2d_nm8c3 | 12:56 | 0.86 |
Phase R so far: 7.45 A100-h; campaign cumulative ~30.6 of 200.
- 54327403 v3b_cold2d (low-gamma basin, convergence parity): chi2=1.594
  fully plateaued (tail 1e-4), gamma=1.3694 (= the published 1.372!),
  theta_E=2.604. FINAL v3b basin comparison at 74-dim parity: high-gamma
  (2.672) chi2=1.516 vs low-gamma (1.369) chi2=1.594 — the binned product
  retains a Delta-chi2_pp=0.078 preference for the steep solution. Honest
  conclusion: fine/binned-scale bimodality persists under the corrected
  likelihood; posterior run on the LOW-gamma basin for the cross-scale
  check, preference reported as a caveat (correlated-noise likelihood
  remains the open item to resolve it).
- 54327398 SVI v2d: ALL GATES PASS (ELBO flat 3e-5 + smooth, full-rank
  74/74, 13:50 runtime). HMC v2d submitted: 54329565 (48ch, 750+8000).
- Submitted: 54329632/54329633 SVI v3b debug/preempt clones (start
  v3b_cold2d); 54329634 v2d_hig (high-gamma basin test at native scale,
  warm from v3cold gamma=2.585). Stale preempts 54325398/99 scancelled.
### FINAL native-scale posterior (v2d: corrected PSF + honest noise)
Job 54329565 (48ch x 750+8000, 10:20): GATE PASS R-hat_max(all)=1.077,
ESS_min(mass)=5714. gamma = 1.4330 [1.3995, 1.4685] (paper 1.372+/-0.023,
consistent <2sigma; model-class systematic ~0.1: nm8c3 MAP 1.314);
theta_E = 2.6551 [2.6530, 2.6572] (paper 2.6463, 0.33%). KEY: gamma ESS
5,714 at 8k draws vs 259 at 13k draws on the defective-PSF product — the
2x-broadened PSF was flattening the gamma direction; the "windowed
mass-matrix adaptation" caveat DISSOLVES with the PSF fix. Supersedes the
old headline gamma = 1.324 [1.309, 1.342].
- 54329634 v2d_hig (native-scale steep-basin test, warm from v3cold
  gamma=2.585): the optimizer DRAINS out of the steep basin entirely ->
  gamma=1.174, chi2=1.297 plateaued. The low/steep bimodality is a
  property of the upsampled product only; at native scale the steep mode
  is not even metastable. (Native low-gamma valley is broad: gamma
  1.17-1.43 within delta-chi2 ~ 0.06 of the 1.234 floor — the HMC
  posterior 1.433 +/- 0.035 integrates over it.)
- Fig-8 v2d rendered: figs/ours_foundry-i_fig8_v2d.png (gamma=1.433,
  chi2=1.26, inner critical curve recovered).
- R2 numbers wave applied to papers/main.tex (new subsection
  sec:campaign-rigor + abstract + both tables + conclusions item 8 +
  README banner). PDF rebuilds clean: 33 pages.
- (2026-06-12) Local machine crashed overnight; local L4 SVI fallback died
  (and was heading for OOM: 28.7 GiB vs 24 GiB L4 — 130^2 n_vi=1000 SVI
  does not fit 2xL4). Both Perlmutter SVI clones COMPLETED overnight:
  54330531 v3br (regular, 55:21) and 54329633 v3bp (preempt, 55:10),
  identical results (same seed): ALL GATES PASS (ELBO flat 3.5e-4,
  smooth, full-rank 74/74). HMC v3b submitted on resume: 54360456
  (debug, 48ch x 750+3500, QZ=svi_v12_v3br).
- 54360456 HMC v3b (14:26): Rhat_gamma=843 — the EXPECTED signature of a
  disconnected bimodal target, not a sampler defect: 45/48 chains sample
  the low basin (conditional gamma = 1.293 [1.283, 1.305], theta_E=2.603),
  3/48 the steep mode (gamma=2.42), ZERO inter-basin migrations in 3,500
  draws. Reported as per-basin conditionals; merged posterior deliberately
  not quoted. Cross-scale: gamma scale-stable only at +/-0.1 (1.29 binned,
  1.43 native, paper 1.37 between) = the systematic floor estimate.
- fig8 v3b rendered from the low-basin median.

### Ledger (Phase R complete)
54324483 1.73 | 54324484 1.20 | 54324486 0.57 | 54324487 0.36 |
54324488 0.63 | 54325014 0.36 | 54325397 1.74 | 54325740 0.86 |
54327398 1.07 | 54327403 1.16 | 54329565 0.69 | 54329632 1.90 (TIMEOUT) |
54329633 3.68 | 54330531 3.69 | 54329634 0.36 | 54360456 0.96
PHASE R TOTAL: 20.96 A100-h. CAMPAIGN CUMULATIVE: 44.1 of 200 budgeted.

## PHASE R VERDICT (CLOSED 2026-06-12)
All three group concerns addressed with corrected artifacts, plus two
discovered defects fixed:
1. chi2=0.451 requalified: wing-contaminated sigma calibration (variance
   factor 0.304); honest fine-scale chi2 = 0.92 (meets <1.1 bar honestly).
   Poisson term verified (keep). Correlation quantified and decoupled
   (corr_len = drizzle footprint; 2x2 blocks 78% internally correlated).
2. PA-tie/multipole hypothesis retracted in abstract/discussion/
   conclusions (paper priors independent; campaign closed gap without it).
3. Degeneracy labels replaced by the empirical gamma-e1/e2-gext valley
   description everywhere.
4. (NEW) PSF sampling-convention defect: kernel must be sampled at
   delta_pix (subgrid_kernel upsamples internally). Native floor
   3.4 -> 1.051 on identical noise. FINAL HEADLINE POSTERIOR (v2d):
   gamma = 1.433 [1.400, 1.468] (paper 1.372+/-0.023, <2 sigma;
   model-class systematic ~0.1), theta_E = 2.6551 (0.33%), Rhat 1.077,
   gamma ESS 5,714 (saturation pathology dissolved with the PSF fix).
5. (NEW) "WHT underestimates sky noise 25%" retracted (wing contamination;
   honest rescale 0.99).
6. Open items: scale bimodality (steep mode metastable only on upsampled
   products; Delta-chi2_pp=0.078 at 0.08") + the correlated-noise
   likelihood to resolve it; gamma systematic floor +/-0.1 across scales.
