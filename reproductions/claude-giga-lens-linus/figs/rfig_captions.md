# P5 report figure captions (rfig_* + reused existing figures)

Drafted for `papers/` (P5 build). Report-prefixed `rfig_*.png` are new, built by
`tools/rfig_build.py` (cgl2 venv, CPU); all other figures listed in §2 are existing
campaign artifacts included as-is. Every caption carries provenance per house style;
DISCREPANCY-FLAGGED numbers quote the ARTIFACT value (synthesis_tables.md §4 binding).

---

## 1. New report figures (rfig_*)

### rfig_decision_matrix.png
**The sampler decision matrix (P2 deliverable): target class × vehicle.**
Cell status is the gate-ledger final disposition (green = converged with evidence
delivered; yellow = partial/mixed/ambiguous; salmon = blocked by budget, cost, or
reference; red = no-verdict/fail/demoted; gray = not run), with the discriminating
measurement and realized cost in each run cell. Honest reading: S1 prior-seeded
MAMS-SMC is the only vehicle that produced logZ at all, and every S1 non-completion
(B3, B4/arbitration, B1r) is a wall-cap on a *healthy* sampler — the capability
boundary is λ-progress per A100-h, not sampler health. Classic MAP→SVI→HMC at the
frozen single-stage settings produced no accepted posterior anywhere in the campaign
(B3 0/4 blocked-by-reference; T2 arbitration arm no-verdict with all chains railed at
the prior edge — reproducing the known T2 single-stage pathology on the scene API).
MCLMC was demoted at B0 (illcond46 ΔlogZ +10.79 ± 0.54 = 20.1σ vs MAMS, artifact
value; the ledger's "30σ" is FLAG-1) and stayed diagnostic-only.
*Sources:* research/synthesis_tables.md §1, research/gate_ledger_final.md;
per-cell artifacts data/smc_b0_report.json, data/b3_cells.json,
data/b2_gate_eval.json, data/results-perlmutter/{b4_marg46_s1_seed2_run.log,
l0_arbitration_classic.json}, data/b5_gate_eval.json, data/b1r_decision_matrix.json,
data/results-perlmutter/l0g2_v3b_scene_seed2.json. Carousel row uses real MUSE
cutouts provided by the team, used with permission; comparison against their own
results is deferred pending sign-off.

### rfig_bracket_scoreboard.png
**The bracket: why γ_binned,corr = 1.103 ≠ 1.433 native anchor — final suspects
scoreboard.** Each card is one pre-registered discriminating measurement, in program
order, with its frozen verdict: profile curvature DEAD (X1-G0 entry-gate kill executed
as written, P4 retired at zero GPU cost, D7); companion EXONERATED (T0.3); spectrum
flooring FALSIFIED by 3.2k–33k nats (T0.4-2); stationarity of the noise-model class
REJECTED at calibrated p = 0.010 — the STANDING mechanism (T0.4-1); real-space
head-to-head as context (T0.4-3; the anchor's full-field χ²_pp carries a
cross-product resolution handicap — honest caveat); injection-recovery CONFOUNDED by
scene-subtraction residue per its pre-registered honesty clause (T1.1, n=3, no
coverage claims); the residue-masked refit STOPPED at its entry gate (T1.1b-G0,
0 A100-h); source/PSF ALIVE-untested. The surviving synthesis: a nonstationary
correlated background priced as stationary biases γ low; γ_binned(corr, low) =
1.1032 ± 0.0086 is certified (T0.2) and reproduced cross-stack (L0-G2). The E3
kernel-scan deferral remains the standing caveat on absolute-γ claims.
*Sources:* data/x1_g0_effective_radii.json, data/t02_t03_gate_eval.json,
data/t04_{stationarity,stationarity_noarc,lambda_arm,realspace_headtohead}.json,
data/t11_gate_eval.json, data/t11b_residue_mask_report.json, CAMPAIGN.md D7.

### rfig_crossstack_l0g2.png
**L0-G2 cross-stack reproduction of the correlated-likelihood v3b refit.**
(a) γ_low posterior: the previous campaign's production SMC ensemble (128 final
particles, phoenix, old validated stack; particles forwarded through that stack's own
bijector — the median reproduces the ledgered 1.103198 exactly) against the scene-API
port's summary quantiles (Perlmutter A100, seed 2); Δγ(median) = 0.0027 ≤ 0.017 gate,
inside the certified 1.1032 ± 0.0086 band. (b) logZ agreement to 0.11 nats across
stacks *and* machines (−4771.08 vs −4770.97), with the P1 seed-spread σ_seed = 1.22
nats shown as the honest scale bar (per-run bootstrap σ not recorded in these summary
artifacts). (c) ΔlogZ(steep−low) = −29.36 on the scene API, inside the old-stack seed
family (−28.88/−32.27/−31.19). This is the campaign's best evidence row: γ = 1.103 now
stands on two independent stacks, and the CorrelatedImageData port is licensed by it.
*Sources:* old stack …/claude-giga-lens/data/results/e2_v3b_low_smc_canary_fix.{json,npz}
(job lineage T0.2 seed 2); scene API data/results-perlmutter/l0g2_v3b_scene_seed2.json
(0.53 A100-h; draws npz resident on Perlmutter $PSCRATCH — summary quantiles plotted);
derived data/rfig_oldstack_gamma_low_constrained.npy; seed family
data/t02_t03_gate_eval.json.

### rfig_b5_evidence.png
**B5 (74-d bimodal foundry_v3b74): the per-basin evidence picture.**
(a) The S1 λ=1 ensembles find MORE evidence than the frozen P2c reference in BOTH
basins (+25.16 ± 0.74 low, +4.86 ± 1.74 steep — same sign). (b) The G2 gate as
written therefore FAILS at 11.0σ (Δ measured 141.90 ± 0.45_boot vs reference
162.2 ± 1.8, band ±5.54) — but the decomposition in (a) plus (c,d) indicts the frozen
reference's within-basin coverage at least as much as the vehicle: both λ=1 ensembles
are γ-DISJOINT from the reference-chain support that P2c was seeded AND mutated from,
while two independent kernels (MAMS 1.0919, MCLMC 1.0935) agree on the low-basin
location (cross-cutting finding B; n=1 seed, attribution needs an out-of-cap
seed-repeat). (e) The MCLMC diagnostic arm inflates the minor-mode weight ×56
[CI95_boot 22–146] = +4.03 ± 0.49 nats — 6.0σ above the pre-registered 1.5–3×
ceiling yet inside the imported 5.56-nat σ_seed repeatability band, so the two frozen
G3 clauses conflict: AMBIGUOUS, and the B0 demotion stands. Minor-mode occupancy is
robust across every estimator (w_low ≤ 1.3e−60).
*Source:* data/b5_gate_eval.json (jobs 56006049, 56251555 attempt #4 chunk-32,
56006052); reference constants therein (P2c, diagonal, 300 particles, 5 reps/basin).

### rfig_budget_burn.png
**A100-hour burn.** (a) Actual vs cap by phase: P1 2.21 (est committed 9.0), T1.1
7.80 (D7 freed pool 10.0), P2 22.45/24, P2b 15.85/18 (D8), P3 5.21/17 — every fence
respected. (b) Cumulative burn at the ledger's harvest points (sub-day x-offsets are
display-only; events plotted in ledger order), closing at 53.51 of the 100-h hard
stop (D5). Component sum 53.52 vs quoted 53.51 is per-row sacct rounding (FLAG-7).
Free-tier GPU-h are additional and uncounted here: B3 12.38 phoenix-L4 h, X2 SBC
29.3 A16-h. Zero-cost closures: X1-G0, T1.1b-G0, B2′, Fermat teaser.
*Source:* CAMPAIGN.md A100-hour ledger (append-before-read protocol).

---

## 2. Existing figures the report should include AS-IS

| File | Report slot | Caption sketch (provenance) |
|---|---|---|
| figs/b5_basin_overlay.png | B5 section, beside rfig_b5_evidence | Both-basin λ=1 posterior overlay drawn BEFORE gate math per house rule (jobs 56006049/56251555/56006052; data/b5_gate_eval.json). |
| figs/b1r_s1_partial_traces.png | B1r/carousel | S1r 3-leg chained-checkpoint traces to λ=0.1506 @36 stages, 11.33 A100-h — healthy sampler (uniq 92–104/128, accept 0.871–0.934), killed by cost; plotted before gate text. Data provided by the team, used with permission; comparison deferred pending sign-off. |
| figs/b1r_s6br_traces.png | B1r/carousel | S6br budget-matched warm-MAMS arm: wall fence at burn round 11/30, sampling never started (ESS = 0; 41.5% of B* = 3,078,912 grads); job 56252401. Same data acknowledgment as above. |
| figs/t04_stationarity_v3b.png + figs/t04_stationarity_v3b_noarc.png | T0.4-1 | The stationarity rejection (max\|z\| 3.67, calibrated p = 0.010) and its arc-excluded replication (p = 0.010) — the standing mechanism's direct evidence (data/t04_stationarity*.json). |
| figs/t04_headtohead_residuals.png | T0.4-3 | Real-space head-to-head residual maps behind the χ²_pp ordering and its whitened-metric inversion (data/t04_realspace_headtohead.json). |
| figs/t04_lambda_arm.png | T0.4-2 | λ-arm flooring falsification, −3.2k to −33k nats (data/t04_lambda_arm.json). |
| figs/x1_g0_arc_maps.png + figs/x1_g0_radial_profiles.png | X1-G0 / D7 | The profile-fork entry-gate kill: fine & binned constrain the slope at the same radius; 24/24 variants non-monotone (data/x1_g0_effective_radii.json). |
| figs/t02_seed_overlay.png | T0.2 | Seed-repeat certification overlay, seeds {2,3,4} (data/t02_t03_gate_eval.json). |
| figs/t03_compmask_overlay.png | T0.3 | Companion-mask discriminator overlay, shift −0.0021 (data/t02_t03_gate_eval.json). |
| figs/t11_recovery_overlay.png | T1.1 | Injection-recovery overlay for the confounded readout (positive-signed biases; control failure) (data/t11_gate_eval.json). |
| figs/b3_cost.png | B3 | The B3 cost row — the cell's readable result (data/b3_cells.json). |
| figs/b2_om0_posterior.png + figs/b2_ratio_control.png | B2/B2′ | DSPL dominant-arm agreement + the ratio-coords control that proved the band uncalibrated (data/b2_gate_eval.json). |
| figs/l0_smc_traces_partial.png | L0 arbitration | The healthy-but-budget-infeasible MC-SMC partial (λ=0.587 @3.5 h A100, job 56168446) behind the PARTIAL-BY-VEHICLE-EXHAUSTION closure. |
| figs/x2_rank_hist_grid.png + figs/x2_rank_hist_glasshouse_gamma.png | X2 gift | SBC rank histograms (LL-block miscalibration, \|z\| up to 5.76) + the glass-house E1c γ disclosure. Label verdicts `proposed (UNCERTIFIED — external)`; scene-substrate results run on the upstream development branches, not characterized further. |
| figs/fermat_dt_teaser.png | Fermat teaser | 88%/61% fractional Δφ shifts vs the ~1% TDCOSMO scale — prominently disclaimed (not a TD lens; synthetic positions; corr posterior is the known over-correcting product) (data/fermat_dt_teaser.json). |

**Note for the LaTeX build:** all rfig_* PNGs are 200 dpi with light (print) styling
only; rebuild with `source /raid/benson/.venvs/cgl2/bin/activate && python3
tools/rfig_build.py`. The one derived input,
data/rfig_oldstack_gamma_low_constrained.npy, was produced under the OLD cgl venv
(bijector-only CPU forward of e2_v3b_low_smc_canary_fix.npz low_particles through
cgl.e2.build_target("v3b").model.bij, the 25a/01a precedent; median check 1.103198
exact, q16 to 1e-6 — the json's q84/σ differ in the 3rd decimal, consistent with
weighted-vs-resampled summary conventions).

**Artifact not locatable locally:** data/results-perlmutter/l0g2_v3b_scene_seed2.npz
(scene-API posterior draws) — cited by the gate ledger but only the summary .json is
on this repo; the draws npz remained on Perlmutter $PSCRATCH. rfig_crossstack_l0g2
therefore plots the scene side from summary quantiles (stated in its caption).

<!-- BEGIN rfig_modeler_summary (40_modeler_summary_figs.py) -->

## Modeler-summary figures (PI request, 40_modeler_summary_figs.py)

| figure | placement | caption |
|---|---|---|
| figs/rfig_diagnostic_rows.png | up-front modeler summary (THE diagnostic figure) | Summary diagnostic, one model per row — the three surviving models only; the corr-low γ=1.103 model is DROPPED from the summary diagnostics by PI direction (rejected; it remains in the mechanism sections as the diagnosed artifact). Rows: (1) E1 v2d-NATIVE MCLMC γ=1.46834 (lead, convergence-certified; 80²@0.13"), (2) diag-low γ=1.29325 (v3b binned 130²@0.08"), (3) steep γ=2.63861 (v3b binned; evidence-disfavored ΔlogZ≈−29). Panels per row: (a) observed cutout (asinh stretch a=0.02, normalized at the data's 99.9th percentile — the SAME stretch is applied to the model panel; 1" scale bar); (b) model image at the model's per-dim posterior-median params with the ridge-SOLVED linear amplitudes (delta-bundle diagonal path, term.internals a*), critical curve overlaid; (c) reduced residual (data−model)/σ over the keep mask, symmetric diverging scale CLIPPED at ±5σ (masked px grey); (d) reconstructed unlensed source surface brightness (sampled-Ie Sérsic + 28 shapelet columns at their solved amplitudes; grid sized to 1.25× the caustic extent, half-width stated per panel) with the caustic overlaid. χ²/px assertions (mean reduced-χ² over keep px): MCLMC row 1.2281 vs report 1.228 (artifact e1_mclmc_home_chi2.json), diag-low 1.57816 vs report 1.578 (artifact t04_realspace_headtohead.json), both to <1%; steep's value 7.58783 is computed and stated here (no prior artifact). θ_E sanity: outer critical-curve median radius matches θ_E to <5% per row. Renders: parity-certified scene stack (10_anchor_arbitration.build_pm diagonal delta bundle), vendored gigalens-linus; ApJ Gu et al. (2022) Fig-8 display conventions. |
| figs/rfig_critcurves.png | up-front modeler summary (compact comparison) | Critical curves (lens plane, over the v3b binned cutout, asinh stretch) and caustics (source plane; stars = median Sérsic-source centers) for the surviving posterior-median mass models — the corr-low γ=1.103 curve was REMOVED from this figure at PI direction (rejected model; discussed via the mechanism sections only): diagonal-low γ=1.29325 (hmc_v13_v3b low basin — the real-space preference), steep γ=2.639 (evidence-disfavored, ΔlogZ≈−29), and the E1 v2d-NATIVE MCLMC diagonal fit γ=1.468336 (the PI-requested converged fit, R̂_worst 1.0031; pooled per-dim scene-z median through the scene bijector, t04 convention; native 80²@0.13" product, curves overlaid on the binned cutout of the same sky). Curve overlays follow the team's GIGA-Lens (Gu et al. 2022) conventions; per-model colors + line styles retain their established identities (MCLMC: slot-7 violet on the white source panel, neutral WHITE dotted on the image panel). Sanity: outer critical-curve median radius matches θ_E to <5% for all three (diag 2.6115" vs θ_E 2.60336"; mclmc 2.6523" vs 2.65353"); the γ<2 models show the expected inner radial critical curve/caustic. Deflections: vendored gigalens-linus EPL(50)+Shear at the certified scene conventions; grid Jacobian via jax autodiff (1201², ±5.2"). |
| figs/rfig_corner.png | up-front modeler summary | Corner plot of the six mass parameters (θ_E, γ, e_1, e_2, γ_ext,1, γ_ext,2) from the stored equal-weight v3b-low correlated SMC particles, seeds 2/3/4 overlaid (this is the SMC convergence visual: SMC has no R̂; the seed-repeat spread σ_seed = 0.00325 plus per-stage ESS/acceptance are the analogues), steep basin (seeds 2–4 pooled) in grey, PLUS the E1 v2d-native MCLMC diagonal posterior in violet (256,000 pooled draws; scene-native physical extraction KEYED via mass_names; equal-weight draw median asserted equal to the run json's pooled γ q50 = 1.468336). Weighted medians of record (from the run JSONs) γ = 1.1032/1.0967/1.1005; the equal-weight particle medians reproduce them to ≤0.0027 (≪ σ_stat 0.008; the resampled-particle convention). γ zoom panel shows the three correlated repeats with weighted medians (the MCLMC γ≈1.47 lies outside the zoom by construction). The cross-stack scene-API repeat (γ 1.1005, Δγ 0.0027, logZ to 0.11 nats) is reported in the text; its draws remain on Perlmutter $PSCRATCH (summary-JSON only locally). |

Derived input: data/rfig_modeler_params.npz — Stage-A (OLD cgl venv, CPU,
bijector-only) physical-space export of the six stored particle sets + the three
median mass models (25a/01a/07 precedent). Regenerate with --refresh.

<!-- END rfig_modeler_summary -->
