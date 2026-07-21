# P5 Synthesis Tables — the decision matrix + results master tables

**Campaign:** claude-giga-lens-linus (EXPERIMENTAL program COMPLETE 2026-07-21; 53.51/100 A100-h
actual — P1 2.21, T1.1 7.80, P2 22.45/24, P2b 15.85/18, P3 5.21/17; ledger CAMPAIGN.md).
**Rule of this file:** every number is traced to its artifact path (relative to this repo root);
where an on-disk artifact and the CAMPAIGN.md prose disagree, the discrepancy is FLAGGED in §4,
never smoothed. Nothing here moves a gate or reinterprets a frozen verdict.

Vehicles (columns of Table 1):

- **S1 = prior-seeded MAMS-SMC** ("MC-SMC"): cold-start tempered SMC, MAMS mutation, adaptive
  λ-schedule, per-stage bootstrap logZ (frozen P0 protocol).
- **Warm MAMS** = MAP-warm-started MAMS-alone (no tempering; S6b/S6br arms).
- **MCLMC** = MCLMC-mutation SMC, formally DEMOTED to diagnostic/cost-frontier-only at B0
  (data/smc_b0_report.json `bias_screen`, see FLAG-1) — never an evidence kernel.
- **Classic** = MAP→SVI→HMC (the GIGA-Lens baseline recipe; B3 frozen settings).
- **Evidence quality** = what logZ/ΔlogZ content the cell yields and how trustworthy it is.

---

## 1. THE SAMPLER DECISION MATRIX (P2 deliverable)

### 1.1 Headline matrix

| Target class (dim, character) | S1 prior-seeded MAMS-SMC | Warm MAMS | MCLMC (demoted) | Classic MAP→SVI→HMC | Evidence quality |
|---|---|---|---|---|---|
| **T0 analytic** (mix2 2-d bimodal; funnel10; illcond46) — B0, 0 A100-h (local GPU) | mix2 **PASS** (logZ err 0.073 ≤ 3σ_boot, ŵ_minor 0.219 vs 0.2); funnel10 **FAIL** −0.66 nat (class limitation, reproduced in the old validated stack); illcond46 **PASS** worst-z 1.83, but logZ −145.80 vs ref −148.20 = 2.40-nat err at σ_boot 0.37 (6.6σ_boot) | not run at T0 (HMC-mutation SMC arm ran instead: mix2 PASS; funnel FAIL −0.54; illcond worst-z PASS with 21.6-nat logZ err at σ_boot 0.41) | mix2 PASS (err 0.009); illcond46 ΔlogZ vs MAMS **+10.79 ± 0.54 = 20.1σ ⇒ DEMOTED** (FLAG-1: ledger says "30σ") | n/a | Analytic refs exist; **σ_boot provably understates logZ error on ill-conditioned targets** (cross-cutting finding A) — data/smc_b0_report.json |
| **hs2 easy scene** (22-d unimodal ×4 systems) — B3, 12.38 phoenix-L4 GPU-h (free tier), 0 A100-h | **BLOCKED-BY-BUDGET 0/4**: no SMC arm finished inside the pre-registered 5 h/arm L4 fence at N=512 — the readable result is the COST row | not run | not run | 4/4 posteriors COMPLETE but **all 4 reference_accepted=false** — escalated R̂_worst 4.36–7.87, ESS_min 63.7–84 vs the frozen acceptance rule R̂<1.05 ∧ ESS≥200 (FLAG-4) ⇒ cells also BLOCKED-BY-REFERENCE | none (no accepted posterior on either arm) — data/b3_cells.json, data/b3_run_ref_sys{0..3}_l4-8.json, research/checkpoints_b3.md |
| **DSPL thin-ridge** (21-d, orig vs ratio coords) — B2/B2′, 1.12 + 1.82 A100-h | orig coords: λ=1 @64 stages, sanity CLEAN (min unique 358 ≥ N/4); minor-arm m̂ = **0/512** vs pre-registered 0.103±0.045 ⇒ FAIL-as-written; control (exact reparam, cannot mode-die) m̂ = **5/512 = 0.0098±0.0043** ⇒ band UNCALIBRATED for this realization; falsifier **NOT-DECIDABLE-AS-REGISTERED**; dominant arm statistically identical (Om0 med 0.4701 vs 0.4702); orig-vs-control z 2.24 (Fisher p 0.031) — suggestive minor-arm under-coverage, ~2σ, underpowered ⇒ **MIXED** | not run | not run | not run | **ΔlogZ(orig−ratio) = +3.06 ± 0.35_boot nats on identical data where the true Δ≈0** ⇒ σ_boot understates cross-parameterization logZ error (cross-cutting finding A, measured) — data/b2_gate_eval.json |
| **T2 46-d cond~1e14** (v2d marg46 / v2d diagonal scene) — B4 3.51 h burned + arbitration 0.02+0.90 h | B4: **NOT EVALUABLE** (TIMEOUT 3:30, zero artifacts, zero stage observability; carried writer defect; true cost > 3.5 h at N=256). Same-class v2d arbitration MC-SMC: L4 λ≈0.45 @~18 h (FLAG-6) + A100-hbm80g λ=0.587 @3.5 h — **sampler healthy, VEHICLE budget-infeasible** (3rd consecutive wall-cap) | n/a (classic arm was warm-started; see next col) | not run | **NO-VERDICT (unconverged)**: warm B3-frozen-settings arm on the parity-certified v2d target — R̂_worst 44.3 (escalated 11.7 ≫ 1.05), ESS_min 63 (escalated; FLAG-5), all chains railed γ≈1.987 [1.983, 1.991] at the prior edge ⇒ γ NOT quotable. **Reproduces the known T2 single-stage-HMC pathology on the scene API** (old-stack lineage: single-stage R̂ 3.1; only the two-stage re-preconditioned recipe converges) — target-intrinsic difficulty, not a stack discrepancy | none; arbitration closed **PARTIAL-BY-VEHICLE-EXHAUSTION** (premise carried by F1–F6 likelihood parity + L0-G2 posterior reproduction) — data/results-perlmutter/{b4_marg46_s1_seed2_run.log, l0_arbitration_classic.json}, research/checkpoints_l0.md |
| **T3 74-d bimodal** (foundry_v3b74, diagonal, f32) — B5, 0.92+0.37+0.58 A100-h (+1.66 h infra-fail) | **G1 PASS both basins**: low λ=1 @31 stages, retention 1.000, γ_eqw 1.0919, logZ 38376.33±0.33_boot; steep λ=1 @22 stages, retention 1.000, γ_eqw 2.5552, logZ 38518.23±0.31_boot. **G2 FAIL-as-written 11.0σ**: Δ(steep−low) 141.90±0.45 vs P2c ref 162.2±1.8 ⇒ −20.30 nats vs band 5.54; decomposition: **+25.16 (low) / +4.86 (steep) MORE evidence than P2c in BOTH basins**, both λ=1 ensembles γ-DISJOINT from the reference chains ⇒ the FAIL **indicts the frozen reference's within-basin coverage** at least as much as the vehicle (n=1 seed) | not run | **G3 AMBIGUOUS (frozen clauses conflict)**: inflates minor-mode weight **×56** [CI95_boot 22–146] = +4.03±0.49 nats, 8.3σ_boot ≠ 0 and 6.0σ_boot above the pre-registered 1.5–3× ceiling — resampling does NOT launder the bias at point estimate; BUT inside the imported 5.56-nat σ_seed repeatability band ⇒ the frozen laundering falsifier technically fires. Within-basin γ untouched (1.0935 vs 1.0919) | not run | Minor-mode **occupancy robust across every estimator** (w_low ≤ 1.3e-60 all constructions; P2c 3.6e-71); **absolute evidence calibration NOT robust at the ±5-nat gate scale** (cross-cutting finding B) — data/b5_gate_eval.json, data/results-perlmutter/b5_v3b74_*_seed2.{npz,json} |
| **Carousel 33-d multi-plane REAL** (MUSE cutouts) — B1 wave-1 11.02 h zero-artifact + B1r 11.33+4.52 h | S1r (N=128, 3 chained ckpt legs): 11.33 A100-h → **λ=0.1506 @36 stages, 0 posterior samples** (sampler healthy: uniq 92–104/128, accept 0.871–0.934 — killed by cost; λ=1 infeasible under ANY campaign budget) | S6br budget-matched at B*=3,078,912 grads: 4.5-h wall fence fired at burn round 11/30 — **sampling never started** (0 draws, ESS=0, R̂ n/a) at 1,277,632 grads = 41.5% of B*; burn healthy (accept 0.873–0.902); throughputs within ~5% (284k vs 272k grads/h) ⇒ truncation is a TARGET-COST fact | not run | not run on carousel (identified as the candidate vehicle class for this target; until-converged S6b est 17–20 h from wave-1 anchors) | none — **NEITHER CONVERGES AT THIS BUDGET**; the D9 cost row is the deliverable — data/b1r_decision_matrix.json, data/results-perlmutter/b1r128_carousel33_*, research/checkpoint_b1r_close.md |
| **v3b correlated REAL** (46-d whitened-correlated likelihood, scene API) — L0-G2, 0.53 A100-h | **PASS (cross-stack reproduction)**: γ_low 1.1005 [1.0992, 1.1065] vs money 1.1032 ⇒ Δ 0.0027 ≤ 0.017; logZ_low −4770.97 vs old-stack −4771.08 (**0.11 nats across stacks/machines**); ΔlogZ(steep−low) −29.36, in the P1 seed family (−28.9/−32.3/−31.2) | not run | not run | not run | **Best evidence row of the campaign**: converged cross-stack posterior + logZ agreement to 0.11 nats; CorrelatedImageData port LICENSED — data/results-perlmutter/l0g2_v3b_scene_seed2.{json,npz} |

### 1.2 Capability-boundary framing (honest, measured)

- **Prior-seeded MAMS-SMC works** — converges AND yields evidence — on analytic targets, the
  DSPL ridge (λ=1 @64 stages), the 74-d bimodal diagonal target (both basins, retention 1.000),
  and the correlated real-data v3b refit (0.5 h). It is the ONLY vehicle in the campaign that
  produced logZ at all.
- **Prior-seeded MAMS-SMC fails by COST, never by health**, on scene-API real/carousel/T2-class
  targets: every non-completion (B1 mocks, B1r real, B3 hs2 N=512, l0arb v2d ×3) is a wall-cap
  on a healthy sampler. λ-progress per A100-h is the binding constraint.
- **Warm MAMS-alone** shares the cost wall on carousel class (0 draws at 4.5 h); its wins, if
  any, live on cheaper targets that this campaign closed via SMC instead.
- **MCLMC** is demoted and stays demoted: evidence distortion 20.1σ (illcond46) and ×56
  minor-mode inflation (B5-G3) with within-basin location intact — a diagnostic, never an
  evidence kernel.
- **Classic MAP→SVI→HMC** at frozen single-stage settings did not produce one accepted posterior
  in this campaign (B3 0/4; v2d NO-VERDICT; X2 30/32 unhealthy at reduced budget) — consistent
  with the prior campaign's finding that this posterior family needs the two-stage
  re-preconditioned recipe. Its unique value here was diagnostic: reproducing the T2 pathology
  on the scene API at 0.9 A100-h.

### 1.3 The two cross-cutting findings

- **(A) σ_boot understates evidence error.** B0: MAMS illcond46 logZ err 2.40 nats at
  σ_boot 0.37 (6.6σ_boot); HMC-mutation 21.6 nats at 0.41 (data/smc_b0_report.json).
  Measured on DSPL on identical data: ΔlogZ(orig−ratio) = +3.06 ± 0.35_boot where the exact
  reparameterization forces true Δ≈0 (data/b2_gate_eval.json `delta_logZ_orig_minus_ratio`).
  This is why P1's σ_seed = 1.79 nats (data/t02_t03_gate_eval.json) is load-bearing for every
  ΔlogZ gate.
- **(B) Frozen-reference coverage indictment (B5-G2).** The 11σ G2 FAIL decomposes into
  BOTH-basin positive evidence offsets (+25.16/+4.86) with both λ=1 ensembles γ-disjoint from
  the hmc_v13_v3b chains the P2c reference was seeded AND mutated from; two independent kernels
  (MAMS 1.0919 / MCLMC 1.0935) agree on the low-basin location. Frozen references built from
  short fixed-step chains can carry within-basin coverage error ≫ the gate σ-scale
  (data/b5_gate_eval.json `G2.reference_comparability`). X2's frozen-set invalidity (Table 3)
  is the same genus on the SBC side.

---

## 2. THE BRACKET / MECHANISM TABLE (why γ_binned,corr = 1.103 ≠ 1.433 anchor)

### 2.1 Suspects scoreboard — FINAL

| Suspect | Verdict | Discriminating measurement | Artifact |
|---|---|---|---|
| Profile curvature (radial mass structure) | **DEAD** (X1-G0 entry-gate FAIL ⇒ P4 retired at 0 GPU-h, D7) | No monotone r_eff ordering in 24/24 robustness variants; fine & binned constrain the slope at the SAME radius (Δr_eff ≈ 0.008″ < ¼ px: r_mean 2.5964 vs 2.5882) yet Δγ = 0.71 ⇒ would need \|dγ_loc/dln r\| ≈ **226** vs O(1) physical | data/x1_g0_effective_radii.json, research/x1_g0_mechanism_check.md |
| Companion galaxy (LL2/LL3 misfit) | **EXONERATED** (T0.3) | Whiten-then-drop (keep_w 9273→8247): γ 1.1011 vs 1.1032 ⇒ shift **−0.0021** (0.26σ_stat; band was ≥ +0.024); convergence indistinguishable from production | data/t02_t03_gate_eval.json `t03` |
| Spectrum flooring ("information discard at spectral zeros") | **FALSIFIED** (T0.4-2) | Data reject floored kernels by **~3.2k nats (λ=0.1) to ~33k nats (λ=1)** per point (−3187/−3328/−3304 → −33,060/−33,450/−33,370); production s_floor = 0.05 confirmed (plan's "0.1" corrected), taps reproduced to 1e-9; pathology localized to down-weighting of high-S large-scale modes (w_b≈0.27 background — exactly T0.4-1's nonstationary component) | data/t04_lambda_arm.json, research/t04_free_checks.md |
| **Stationarity of the noise-model class** | **REJECTED — the STANDING mechanism** (T0.4-1) | v3b max\|z\| = 3.67, calibrated p = **0.010**; arc-excluded v3 p = **0.010** (replicated spatial pattern); cross-block ρ(0,1) spread ~0.28 peak-to-peak = **16.4×** the ±0.008 drizzle-registration envelope. The stationary kernel class behind γ=1.103 is provably violated by the field | data/t04_stationarity.json, data/t04_stationarity_noarc.json, figs/t04_stationarity_*.png |
| Injection methodology (T1.1 construction) | **CONFOUNDED → data-side redesign queued (P3)** | See §2.2 | data/t11_gate_eval.json, data/t11b_residue_mask_report.json |
| Source model / PSF representation | **ALIVE (untested)** | Inherits the bracket question from D7 (X1-G0's positive content: bracket driver differs between products AT FIXED RADIUS ⇒ noise/likelihood + PSF, not mass structure); PSF-marg MVP + evidence-scored source ladder queued | CAMPAIGN.md D7 |

Context row (report-only, T0.4-3): on the SAME v3b pixels in real space, diag-low γ≈1.29 fits
best (χ²_pp 1.58) vs anchor 1.433 (7.44) vs corr-low 1.103 (8.32) — and the production whitened
metric INVERTS this ordering (+501 nats for corr-low); corr-low's residual is smooth lens-center
misfit, the same currency as the fine-low gaming (data/t04_realspace_headtohead.json; the
anchor's full-field number carries a cross-product resolution handicap — honest caveat).

**P1 mechanism synthesis** (CAMPAIGN.md): one mechanism spans T0.4-1/2/3 + X1-G0 — a
NONSTATIONARY correlated-background component priced as stationary lets the whitened metric
discount large-scale real-space misfit, biasing γ low. Noise-model-CLASS misspecification, not
source/PSF, is the prime suspect. T0.4-1's rejection is UNREFUTED by T1.1 (in-class injection
scene ⇒ the mechanism's misfit lever arm is absent by construction).

### 2.2 The T1.1 / T1.1b story (injection-recovery, 7.80 A100-h)

- **Design:** inject γ_truth = 1.433 arcs into the REAL v3b drizzle residual field ×3 shifts;
  pre-registered zones (post σ_seed finalization): confirm-LOW if median(γ_rec−1.433) < −0.078;
  exonerate if \|median\| < 0.026; control (diag likelihood, same data) predicted in [1.29, 1.43].
- **Result: NO-CONFIRM / NO-EXONERATE — positive-signed, outside BOTH zones.** γ_rec med
  1.5151 / 1.5719 / 1.5076; biases **+0.0822 / +0.1389 / +0.0747** (own-σ z +2.25/+3.23/+1.82);
  median **+0.0822** (dropping dup-flagged inj2: +0.0784, same zone). **Control FAILS HIGH AND
  SICK**: γ 1.5677 ∉ [1.29, 1.43] with total resample collapse (1/128 unique, γ_σ = 4e-16;
  srcS.Ie railed ~58× truth) ⇒ per the pre-registered honesty clause the INJECTION CONSTRUCTION
  is implicated (bright-object scene-subtraction residue; recovered sources ×1.8–3 bigger,
  ×2.4 brighter than truth in all 3 corr runs). n=3, no coverage claims.
- **Whitener-isolating datum:** same-data differential corr−diag on inj1 = **−0.0526** — sign
  consistent with the mechanism, ≈16% of the 0.33 gap; no defensible error bar (diag leg
  degenerate). (data/t11_gate_eval.json `differential_corr_minus_diag_inj1_same_data`)
- **T1.1b-G0 (residue-masked refit entry gate): FAIL — STOPPED at 0 A100-h.** Whiten-then-drop
  dof loss **85.8%** (keep_w 9273→1320) vs the ≤40% pre-declared budget; region drop alone loses
  47.6%; arc-band whitened px 6373→108 (pure outer sky). Fit-side residue masking is
  STRUCTURALLY incompatible with this injection design — residue-free injections must be built
  DATA-side (kernel-sampled noise-only / sky-set bootstrap / deeper multi-start subtraction),
  queued for P3. (data/t11b_residue_mask_report.json)

### 2.3 γ certification + cross-stack reproduction

- **γ_binned(corr, low) = 1.1032 ± 0.0086 CERTIFIED** (T0.2): σ_tot = √(σ_stat² + σ_seed²) =
  √(0.0080² + 0.00325²) = 0.00864 ≈ 1.08×σ_stat; seeds {2,3,4} γ_low {1.1032, 1.0967, 1.1005}
  ⇒ σ_seed 0.00325 (kill σ>0.024 not tripped); steep {2.6393, 2.6522, 2.6485} ⇒ σ_seed 0.00664;
  ΔlogZ(steep−low) −28.88/−32.27/−31.19 per matched seed ⇒ LOW-basin preference SEED-STABLE
  (≥16σ_seed); σ_seed(ΔlogZ) = 1.79 nats. n=3 caveat: σ estimates carry ±46% χ-dist sampling
  error, quoted with every use. (data/t02_t03_gate_eval.json)
- **Cross-stack reproduction (L0-G2): PASS** — the scene-API CorrelatedImageData port reproduces
  γ 1.1005 (Δ 0.0027 ≤ 0.017) and logZ to 0.11 nats on an independent stack + machine; ΔlogZ
  sign and magnitude in the seed family (−29.36). γ=1.103 now stands on two independent stacks.
  (data/results-perlmutter/l0g2_v3b_scene_seed2.json)

---

## 3. GIFTS / CERTIFICATION TABLE

| Deliverable | Content (measured) | Artifact |
|---|---|---|
| **Scene-API certification (F1–F8, both machines)** | F1 fwd image 5.7e-15 (≤1e-12); F2 design cols 3.9e-15; F3 diag loglik+χ² 7.5e-9 (≤1e-8); F4 grads 1.5e-11 rel-L2; F5 delta-kernel ≡ stock 5.8e-11/0.0; F6 Occam vs **fp128 truth** PASS under the signed exception (v2d 1.19e-11; v3b 4.48e-10 ≤ 7.76e-10 at cond 7e7 — the original numpy-slogdet reference itself errs ~7e-10 there); F7 roundtrip exact; **F8 Perlmutter-native A100 re-run: ALL HARD GATES PASS** (F1 5.682e-15 bit-consistent with phoenix, F4 1.314e-11, F6 3.562e-10; 58 s, job 55980038) | data/parity_report_scene.json, data/results-perlmutter/parity_report_scene_pmnative_55980038.json |
| **Sersic-bn finding** (+2 more convention reconciliations) | The bn approximant DIFFERS between stacks: old 1.9992n−0.3271 vs theirs exp(0.6950+ln n−0.1789/n) — a real **~2e-3 model-level difference**, measured and reported (informational native_profiles arm); plus f32 shapelet Hermite prefactor (~1e-8) and f32 grids + un-renormalized subgrid PSF kernel (~1e-4/2e-6 at v2d/v3b). Parity achieved via cgl2-side overrides; **vendor UNPATCHED** | data/parity_report_scene.json; CAMPAIGN.md 2026-07-15 parity entry |
| **Correlated-noise likelihood port** | CorrelatedImageData/-LikelihoodTerm on their Dataset/LikelihoodTerm seam: dense-Cholesky exact ref agrees to 5.5e-12 nats (gate ≤0.1); delta identities ≤9.1e-13/0.0; whitener bundles re-validated (e_op ≤0.02 strict, 4/4; v2d_relaxed correctly inadmissible-by-design) | data/correlated_term_validation.json, data/whitener_manifest.json |
| **X2 — first formal SBC of the GIGALens pipeline class** (28.3+1.0 A16-h free tier, N=32 of 64 — under-delivery stated) | **(i) Frozen-set invalidity**: their 100SystemsStandard80px.npz has generation prior ≠ fitting prior (their own provenance) ⇒ SBC on it fails by construction; arm ran prior-matched regeneration instead. **(ii) Severe lens-light rank miscalibration**: LL.Ie z=−5.27 (p 7.6e-11, 18/32 bottom bin), LL.R_sersic z=+5.76 (p 1.5e-10, 18/32 top bin), one-sided coherent — not an under-mixing shape; 3 candidate channels not separable (doubt report in checkpoint). **(iii)** Mass block: no location bias (worst \|z\| 2.25) but 6/8 fail uniformity via U-shapes (under-mixing signature; 30/32 fits unhealthy, 2/32 healthy at this budget). **(iv) Glass-house**: our OLD E1c SBC failed the same γ gate (p 6.5e-5, cov68 0.34) and the healthy-only amendment flipped it to PASS — sampler pathology can fake rank failures; that amendment path is exactly what n_healthy=2 blocks here. All verdicts `proposed (UNCERTIFIED — external)` | data/x2_sbc.json, research/x2_sbc_gift.md, figs/x2_rank_hist_grid.png, figs/x2_rank_hist_glasshouse_gamma.png |
| **Fermat Δφ teaser** (0 GPU-h, prominently disclaimed: NOT a TD lens, synthetic image positions, corr posterior is the known over-correcting product) | Noise-model choice moves synthetic Fermat-potential differences **88%** median \|frac shift\| anchor→corr (10.7σ) and **61%** on the same-product diag→corr arm (17σ) — vs the ~1% TDCOSMO-relevant scale. The motivation number for correlated noise in any future TD work | data/fermat_dt_teaser.json, research/fermat_dt_teaser.md |
| **Vendor/substrate defect finding** (external, UNCERTIFIED) | The vendored scene-API `ModellingSequence.MAP` raises a shard_map cotangent TypeError under the library's own declared jax-0.6.2 pin (reworked for jax-0.10) — B3/X2 replicate the recipe with attribution (single-device identity); X2 additionally documents the `check_vma=False` monkeypatch and an EPL e1²+e2² prior-edge silent-clip channel (~5e-6 prior mass) | research/checkpoints_b3.md pre-run amendment; data/x2_sbc.json `physicality` |
| **Ops/watchdog lessons** (each cost real budget once) | **(1) hbm80g pin**: T1.1 inj3 OOM on an hbm40g node (1.49 h burned) and l0arb's 21.04-GiB mutate alloc OOM on plain-gpu 40 GB (the "~12 GB" sizing was an L4 measurement taken WITH remat) — `-C gpu&hbm80g` fixed both; memory-sizing figures don't transfer across remat settings. **(2) Script-snapshot ordering**: sbatch snapshots the script AT SUBMISSION — the B5-steep chunk-48 fix "resubmit" ran the old script (56168443, 2 failed attempts); protocol now verifies the submitted snapshot md5 via `scontrol write batch_script`. **(3) Single-writer manifest**: concurrent fronts clobbered deploy.md5 (caught + repaired); rule = pull remote first, UNION-merge, one writer per wave. **(4) Checkpoint retrofit** (cgl2/samplers/ckpt.py): per-stage full-state ckpts + exit-3 PARTIAL protocol + resume BIT-IDENTICAL incl. full logZ+bootstrap (58-test suite); turned the 94%-zero-artifact wave-1 failure mode into 3/4 readable recoveries and made the 12-h B1r chain and λ-readouts possible at all. Also: chunk must divide N AND pilot_size (96/48 vs pilot 64 assert); watchdog COMPLETED_NO_ARTIFACT doubles as the harvest reminder; OPENBLAS_NUM_THREADS=4 on aarch64 (livelock ~100×); never run the lensing likelihood on CPU | CAMPAIGN.md ledger rows 55952482/55958518, 56006048/56168446, 56168443/56170614/56251555, 2026-07-16 recovery-lane entry, 2026-07-20/21 stage entries |

---

## 4. DISCREPANCY FLAGS (artifact vs ledger — reported, not smoothed)

1. **MCLMC bias-screen σ**: CAMPAIGN.md says "bias screen 30σ" (P0 exit decision; repeated in
   the B5-G3 gate row). The artifact of record data/smc_b0_report.json `bias_screen.t0_illcond46`
   has ΔlogZ 10.79, σ 0.536 ⇒ **n_sigma = 20.1**, not 30. Demotion (>3σ rule) unaffected.
2. **B0 illcond "−118 vs −146 across N"**: the P0-exit note's "mams illcond logZ −118 vs −146"
   matches NO pair in data/smc_b0_report.json (MAMS: −145.80 vs ref −148.20, err 2.40 nats;
   HMC: −169.75, err 21.6; MCLMC: −135.01, err 13.2). Possibly from an earlier quick/other-N run
   not preserved on disk. The qualitative claim (σ_boot understates on illcond) is SUPPORTED by
   the on-disk numbers either way (6.6σ_boot–52σ_boot equivalent errors).
3. **Gate-record B0 row still reads "PENDING"** although B0 was evaluated (json `b0_pass: false`;
   funnel10 FAIL + MCLMC demotion are recorded in the P0-exit decision text below the table).
   Ledger-housekeeping only.
4. **B3 "4/4 reference posteriors COMPLETE" (stage log 2026-07-18) omits that all 4 FAILED the
   pre-registered reference-acceptance rule** (escalated R̂_worst 4.36–7.87 ≫ 1.05; ESS_min
   63.7–84 < 200; per-cell JSONs record `reference_accepted: false`, "blocked-by-reference").
   Honest cell status = SMC BLOCKED-BY-BUDGET **and** classic BLOCKED-BY-REFERENCE; "references
   only" must not be read as "reference posteriors usable". (Consistent with — and additional
   evidence for — the T2/X2 single-stage-recipe convergence finding.)
5. **Classic-arm ESS_min 63** (stage log) is the ESCALATED stage (62.98); the initial 50-chain
   stage was 53.6 (data/results-perlmutter/l0_arbitration_classic.json). Quote with stage.
6. **L4 MC-SMC partial λ**: ledger says "λ≈0.45 @~18 h"; the on-disk
   data/l0_arbitration_smc.json records λ=0.4183 at a 12.26-h resume leg (resumed from stage 26).
   The 0.45/18-h figures presumably include the earlier leg(s)/checkpoints (stage_000–067 on
   ckpt dir); the on-disk summary json does not itself show λ=0.45.
7. **Rounding notes (no action)**: campaign total 53.51 vs component sum 53.52 (per-row sacct
   rounding); σ_seed(ΔlogZ) 1.786 quoted as 1.79; B5 band 5.537 quoted as 5.54; Fermat 0.8773
   quoted as 88%.

---

*Written by the P5 synthesis Task B agent, 2026-07-21. Sources: CAMPAIGN.md (decisions D1–D9,
gate record, ledger, stage log) cross-checked against data/*.json, data/results-perlmutter/*,
research/*.md as cited per row. No gate re-scored; no threshold moved.*
