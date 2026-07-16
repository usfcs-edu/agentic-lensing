# T1.1 injection-recovery readout — pre-registered gate evaluation (2026-07-15)

**Status: NO-CONFIRM / NO-EXONERATE — result lands OUTSIDE both pre-registered
zones, on the POSITIVE side (opposite the predicted direction), and the
diagonal control fails its own prediction window high — which, per the
pre-registered honesty clause, implicates the injection construction
(bright-object scene-subtraction residue), not the whitener. T1.1 is
CONFOUNDED; the whitener is neither confirmed as the 0.33-gap driver nor
exonerated.**

Extraction: exactly the P1-harvest conventions (`08_harvest_t11.py`, OLD cgl
venv, bijector-only CPU; per-run JSON weighted quantiles via
`cgl.e2._weighted_quantile` are authoritative; equal-weight particles for
plots/nuisances, eqw-vs-weighted median cross-check ≤ 0.005). Plot inspected
BEFORE gate math (`figs/t11_recovery_overlay.png`); plot and numbers agree.
Full numbers: `data/t11_gate_eval.json`.

## Inputs and provenance (all clean)

- Jobs: 55952480 (inj1), 55952481 (inj2), 55958518 (inj3 SMC-only resubmit of
  FAILED 55952482), 55952483 (diag control). All COMPLETED; artifacts pulled
  from CFS to `data/results-perlmutter/t11_*`.
- md5 echoes in the slurm logs match the local build report exactly:
  inj1 `98b2b825…`, inj2 `175166b8…`, inj3 `81271c7f…`, delta whitener
  `fa167fe2…`; e2.py `782a268a…` (the T1.1-trued production file) on all jobs.
- inj3 resubmit provenance CONFIRMED: slurm-55958518.out shows `SKIP_PREP=1`,
  same datafile md5 `81271c7f…`, and its SMC run.log warm-starts
  `q from …/t11_inj3_canary_svicov.npz` — the prep written by the failed job
  (CFS timestamp 17:27, before the resubmit ran). No numerics change.
- Config asserts (harvest script): seed 2, p128, production whitener_v3b
  (M=10, n_keep_w=9273) on the three injections; delta whitener (M=0,
  n_keep_w=16653) on the control; correct `data_file` per run.
- γ_truth = 1.4329787059806258, identical across the three truth jsons
  (`data/t11_inj{1,2,3}_truth.json`).

## Sanity (before gates)

Every run reached λ=1 by construction (artifact written ⇒
`run_adaptive_tempered_smc` terminated at λ=1); λ-steps 27/34/20/37 ≪ cap 400;
`n_floored_q=0`; basin purity clean (`frac_γ>1.9 = 0.000` everywhere).

| run | w_ess/128 | λ-steps | unique particle rows /128 | flags |
|---|---|---|---|---|
| inj1 | 125.6 | 27 | 52 | none (healthier than production family) |
| inj2 | 124.4 | 34 | 18 | dominant duplicate cluster at TOP quantile (γ_med == γ_q84 exactly); same class as T0.2 seed3-low; reported, not sick |
| inj3 | 107.9 | 20 | 81 | none (healthiest of the set) |
| diag | 128.0 | 37 | **1** | **SICK: total resample collapse — final population is a single 46-dim point copied 128×; γ_σ = 4.4e-16; several params railed (srcS.Ie ≈ 10.09, LL0/LL2/LL3.Ie, LL2.center_x)** |

Baseline for "unique rows": the certified-healthy production family spans
14–37/128 (seed2 37, seed3 14, seed4 27, compmask 17) — so inj1/2/3 are all
within or better than the certified envelope; only the control is pathological.
The control's collapse mechanism: the diagonal likelihood is far sharper
(logp scale −12.3k vs −4.6k) and the frozen production SMC moves (step 0.1,
4 MCMC steps, whitened-geometry mass matrix) evidently had ~zero acceptance
late in tempering → systematic resampling concentrated all weight on one
particle. The "Traceback" lines in the canary logs are the known benign TFP
WeakStructRef cleanup noise, present in healthy runs too.

Prep sanity (canary logs): MAP γ_map in-basin on all four (1.5113 / 1.4946 /
1.4266 / 1.3108); SVI full_rank on all; stage-2 Rhat_max 3.8 / 9.3 / 3.0 /
**307.6 (diag — prep also unconverged)**. Preps are warm starts only; SMC is
the authority.

## Pre-registered gates (finalized thresholds; NOT moved)

Gates: CONFIRM median(γ_rec − 1.43298) < −0.078; EXONERATE |median| < 0.026;
between = partial with sign. σ_inj = **the injection's own SMC posterior σ**
(weighted, run_correlated_smc extraction). No coverage claims at n=3.

| run | γ_rec median [q16, q84] | σ_inj | bias vs 1.43298 | z (own σ) | logZ |
|---|---|---|---|---|---|
| inj1 (0,0)″ | 1.5151 [1.4877, 1.5543] | 0.0365 | **+0.0822** | +2.25 | −4654.84 |
| inj2 (+.030,−.014)″ | 1.5719 [1.4796, 1.5719] | 0.0431 | **+0.1389** | +3.23 | −4644.91 |
| inj3 (−.022,+.034)″ | 1.5076 [1.4685, 1.5482] | 0.0411 | **+0.0747** | +1.82 | −4634.52 |
| diag control | 1.5677 [1.5677, 1.5677] | 4e-16 (degenerate) | +0.1347 | undefined | −12799.37 (diag dof — not comparable) |

- **median bias (n=3) = +0.0822** → not < −0.078 (NO CONFIRM); |+0.0822| ≥
  0.026 (NO EXONERATE); not in the negative partial zone either. The result is
  **outside the pre-registered zone structure, positive-signed** — the
  mechanism's directional prediction failed at face value.
- Robustness without the dup-cluster-flagged inj2: median(+0.0822, +0.0747) =
  **+0.0784** — same conclusion. (inj2's median is inflated by quantile
  discreteness: its weighted mean is 1.5399, bias +0.107, z≈+2.5.)
- logZ across injections (−4654.8 / −4644.9 / −4634.5) are on different data
  files — reported, not interpreted.
- Under the superseded provisional thresholds (−0.072/0.024) the verdict is
  identical in kind.
- **Control gate: FAIL.** γ_rec(diag) = 1.5677 ∉ [1.29, 1.43] — biased HIGH by
  +0.135, farther from truth than the correlated runs — AND the run is sick
  (point mass). Gate evaluated with and without it: the injection-side verdict
  does not depend on the control; the control's failure changes the
  *interpretation* (below).

## Nuisance recovery vs injected truth

Per-injection truth from `t11_inj{i}_truth.json` (shifts baked in);
equal-weight particle quantiles; dev in units of the run's own half-[q16,q84].

- Source centers recover well where the run is healthy: inj1 srcS.center_x/y
  within 0.2σ; inj3 within 1σ. inj2 (dup cluster) shows srcS.center_x off
  −8σ and srcS.e1 sign-flipped (+6.6σ) — quantile discreteness + genuine
  distortion.
- **Systematic absorption signature (all three correlated runs): the recovered
  source is BIGGER and BRIGHTER than injected truth** — srcS.R_sersic
  0.18–0.32 vs truth 0.107 (+3.4σ, +2.6σ, +3.7σ), srcS.Ie ~0.42 vs 0.17
  (×2.4) in inj1/inj3; srcShp.center_x pulled +2.8 to +4.3σ. Lens-light blocks
  distort in sympathy (LL1.R_sersic/Ie, LL2.Ie > 3σ in all three).
- Railed injected-truth parameters (unconstrained |z_med| > 3.5): inj1 LL0.Ie;
  **diag control rails srcS.Ie at 10.09 ≈ 58× truth** plus LL0/LL2/LL3.Ie and
  LL2.center_x — the diagonal fit pumped component amplitudes to the rails to
  absorb residue flux. FLAGGED per pre-registration.

## Interpretation (tied to the mechanism)

1. **Common-mode upward pull ⇒ data, not likelihood.** Both likelihood classes
   (correlated AND diagonal) on the same injected field recover γ high by
   +0.07..+0.14. The only shared element is the data. The injection = real
   residual field + in-class synthetic scene, and the residual field carries
   the real lens's imperfect scene subtraction — the G1 decomposition measured
   this before submission (bright-object χ²_pp 2.71, center r<1.2″ 5.46; 103%
   of the full-keep excess). The nuisance readout shows exactly the absorption
   signature (source grows ×1.8–3 in radius, ×2.4 in flux; amplitudes rail in
   the diag fit): the fits eat the structured residue and the mass slope
   steepens. **Per the pre-registered honesty clause — the control failing
   high implicates the INJECTION CONSTRUCTION (scene residue), not the
   whitener. That clause governs this readout.**
2. **The only whitener-isolating number:** same-data differential on inj1,
   γ(corr) − γ(diag) = 1.5151 − 1.5677 = **−0.0526**. Sign consistent with the
   T0.4 stationary-whitener low-bias mechanism; magnitude ≈ 16% of the 0.330
   real-data gap (1.4330 − 1.1032). CAVEAT: the diag leg is a degenerate point
   mass, so this differential carries no defensible error bar — indicative
   only. At the CONFIRM threshold the whitener would have had to explain ≥
   0.078/0.330 ≈ 24% via the median bias; that level is NOT supported here.
3. **What T1.1 could and could not test.** The injected scene is exactly
   in-class (truth source = the fitted shapelet ridge solve). The P1/T0.4
   mechanism is an INTERACTION: stationary whitening discounts large-scale
   real-space MISFIT. With an in-class scene there is little structural misfit
   for the whitener to discount, so the mechanism's lever arm was largely
   absent BY CONSTRUCTION — even a clean EXONERATE here would not have
   contradicted T0.4-1's direct stationarity rejection (p=0.010), which
   STANDS unrefuted. And we did not get a clean anything: the scene-residue
   confound (comparable magnitude, opposite sign to the prediction) dominates.
4. **Verdict: T1.1 INCONCLUSIVE-BY-CONFOUND.** The 1.103 over-correction
   diagnosis (noise-model-CLASS misspecification) still rests on T0.4's direct
   evidence; T1.1 neither strengthens nor weakens it decisively. The whitener
   contribution measurable at in-class scene specification is ≤ ~0.05 in γ
   (differential, unquantified error).

## Implications

- **P3 CorrelatedImageData:** the pluggable locally-stationary whitener seam
  KEEPS its priority — T0.4-1's class rejection is untouched by this readout.
  Add a second seam requirement learned here: injection-style validation needs
  a residue-free data path (noise-only injections from the fitted kernel /
  sky-set bootstrap, or a deeper multi-start scene subtraction) before any
  whitener-bias number is quotable.
- **P2 benchmark framing:** the diag-control collapse is itself a benchmark
  datum — the production two-stage recipe + SMC settings tuned on the
  correlated geometry do NOT transfer to the sharper diagonal likelihood
  (total particle collapse at p128, prep Rhat 307). Strengthens the "real
  ill-conditioned lens posteriors need per-target tuning" thesis; a diag arm
  needs its own step-size/metric tuning.
- **Engagement memo:** the honest line is "injection-recovery on the real
  residual field is confounded by scene-subtraction residue (+0.08..+0.14
  common-mode γ bias, both likelihoods); the correlated-vs-diagonal
  differential on identical data is −0.05 (sign consistent with the
  over-correction mechanism, small vs the 0.33 gap at in-class scene
  specification); the definitive whitener test needs residue-free injections
  plus a locally-stationary arm." No "necessary but not sufficient" upgrade or
  downgrade from T1.1.

## Cost

sacct actuals (single A100, shared QOS): 55952480 1.89 h, 55952481 2.00 h,
55952482 1.49 h (FAILED, ledgered), 55952483 1.99 h, 55958518 0.43 h —
**T1.1 total 7.80 A100-h actual vs 10.0 est (8.0 + 2.0 resubmit)**.
Campaign actual to date: 2.21 (P1 T0.2/T0.3) + 7.80 = **10.01 A100-h** of the
100 h cap (D5).

## Artifacts

- `data/results-perlmutter/t11_*` (4 SMC sets + 4 prep sets + run logs + 5 slurm logs)
- `figs/t11_recovery_overlay.png` (inspected before gate math; agrees)
- `data/t11_gate_eval.json` (full numbers; `08_harvest_t11.py`)
- n=3 caveat: quoted per pre-registration — per-injection z only, no coverage
  claims; median-of-3 is order-statistics coarse (it equals inj1's bias).
