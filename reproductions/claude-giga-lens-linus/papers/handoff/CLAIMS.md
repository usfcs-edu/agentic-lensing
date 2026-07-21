# Lab Notebook — claude-giga-lens-linus external campaign (claims register for the GIGALens team)

Claims produced by the claude-giga-lens-linus bridge campaign (Benson group) on the vendored
scene-API substrate `gigalens-linus @ 80916d2` (UNPATCHED) and on the old validated
claude-giga-lens stack. This register follows `docs/logs/lab-notebook-TEMPLATE.md`.

**Last updated:** 2026-07-21

> **Every verdict below is `proposed (UNCERTIFIED — external)`.** We are the producer; the
> grader is the GIGALens team (and Greg Benson for our-side scope). Certification requires
> inspecting the named artifacts, never our summaries. Claims marked **[SIGN-OFF GATED]**
> derive from your unpublished repos/data: internal-memo material only, publication-gated on
> your explicit sign-off (our campaign bright line §8.2), regardless of what you decide about
> the rest. "Validated" is reserved for our old certified stack; scene-API results are
> "reproduced/measured, UNCERTIFIED (external)" in your vocabulary.

---

## Current state

Experimental program COMPLETE 2026-07-21 (53.51 of 100 A100-h; nothing in flight). Authoritative
record: `reproductions/claude-giga-lens-linus/CAMPAIGN.md` (locked decisions D1–D9, full gate
record) + `research/*.md` checkpoints + `data/*.json` gate evals. All runs pre-registered
(design checkpoints with derived thresholds logged before submission); plots produced before
gate text (your house rule, adopted).

---

## Claims register

### CGL2-C1 — The scene-API forward model is certified against the validated old stack for the EPL+Shear + 4×Sérsic / Sérsic+Shapelets(n_max=6) config class
- **Status:** `proposed (UNCERTIFIED — external)`
- **Criterion (pre-registered):** F1 forward image ≤1e-12 rel; F2 design columns ≤1e-12 rel; F3 diagonal masked loglik+χ² ≤1e-8; F4 constrained-space gradient ≤1e-8 rel-L2 — old validated 58ec9a7 stack vs `SceneSimulator`/`ImageLikelihoodTerm`, same foundry-i v2d/v3b inputs, z_ref + 3 seeded perturbations, compared in constrained space.
- **Numbers:** F1 5.7e-15 · F2 3.9e-15 · F3 7.5e-9 · F4 1.5e-11 — **all PASS**. First external certification evidence for the scene API (your `tests/validation` README marks the suite unrun).
- **Evidence:** `data/parity_report_scene.json`, harness `01_parity_scene.py` (+`01a_gen_parity_refs.py`), F8 repeat on Perlmutter native env `data/results-perlmutter/parity_report_scene_pmnative_55980038.json`.
- **Doubt report:** (1) parity holds GIVEN three documented convention reconciliations (Sérsic b_n approximant, f32 shapelet Hermite prefactor, f32 grids + un-renormalized subgrid PSF — `cgl2/scene_build.py` docstring; the reconciliations live cgl2-side, vendor UNPATCHED), so stock-vs-stock differs at those items' scale (~2e-3/1e-8/1e-4); (2) one config class only — no statement about dPIE/BPL/multi-plane paths; (3) our numpy is 2.4.6 vs your 2.1.3 pin (declared deviation, covered by the gate battery).
- **Proposed by / on:** cgl2-linus campaign · 2026-07-15 · **Grader:** _pending (GIGALens team)_

### CGL2-C2 — The Sérsic b_n approximants differ between stacks at a real ~2e-3 model level
- **Status:** `proposed (UNCERTIFIED — external)`
- **Criterion:** informational (measured native_profiles arm of the parity harness; no threshold).
- **Numbers:** old stack `1.9992 n − 0.3271` vs scene API `exp(0.6950 + ln n − 0.1789/n)`; ~2e-3 model-level image difference for the tested class.
- **Evidence:** `data/parity_report_scene.json` (native_profiles arm), `cgl2/scene_build.py` docstring item 1.
- **Doubt report:** both are literature approximants to the exact b_n equation; we did not adjudicate which is closer to exact for your use range — this is a heads-up that cross-stack comparisons at <1e-3 need the reconciliation, not a defect report.
- **Proposed by / on:** cgl2-linus campaign · 2026-07-15 · **Grader:** _pending_

### CGL2-C3 — The correlated-noise LikelihoodTerm (upstream-shaped on your Dataset/LikelihoodTerm seam) is exact in its analytic limits
- **Status:** `proposed (UNCERTIFIED — external)`
- **Criterion (pre-registered):** F5 delta-kernel term ≡ stock `ImageLikelihoodTerm` ≤1e-10; 03-A vs dense-Cholesky exact reference ≤0.1 nat (32² toy); F6 Occam −½logdet A vs fp128 truth ≤ max(1e-10, 5·eps·cond(A)·1e-2) (restated gate, signed exception 2026-07-15).
- **Numbers:** F5 5.8e-11 · 03-A 5.5e-12 nats · F6 v2d 1.19e-11, v3b 4.48e-10 ≤ 7.76e-10 — **all PASS**. Sub-finding worth your attention: at cond(A)=7e7 the original 1e-10 cross-algorithm gate sits BELOW the f64 noise floor — numpy slogdet itself errs ~7e-10 vs fp128 truth (relevant to any logdet gates you write for your own lstsq/Occam work).
- **Evidence:** `cgl2/correlated.py`, `cgl2/marg.py`, `data/parity_report_scene.json`, `data/correlated_term_validation.json`.
- **Doubt report:** single-forward-eval contract honored (one `lstsq_simulate(return_stacked=True)` per call) but validated only for our whitener-bundle class (e_op ≤ 0.02, `data/whitener_manifest.json`); the −½logdet A term your current lstsq drops is included here — it matters for any downstream Bayes factor.
- **Proposed by / on:** cgl2-linus campaign · 2026-07-15 · **Grader:** _pending_

### CGL2-C4 — The certification battery passes identically under the Perlmutter native pinned env (both machines)
- **Status:** `proposed (UNCERTIFIED — external)`
- **Criterion (pre-registered):** F8, report-only — full battery under the production A100 env.
- **Numbers:** all hard gates PASS (job 55980038, 58 s); F1 5.682e-15 bit-consistent with phoenix; F4 1.314e-11; F6 3.562e-10 within restated tol.
- **Evidence:** `data/results-perlmutter/parity_report_scene_pmnative_55980038.json`.
- **Doubt report:** jax 0.6.2 both machines (your declared pin); the jax-0.10-nightly cell in the original F8 wording was superseded by the Path-A native env (ledgered) — no nightly coverage claim.
- **Proposed by / on:** cgl2-linus campaign · 2026-07-16 · **Grader:** _pending_

### CGL2-C5 — Cross-stack posterior-level reproduction: the scene-API correlated refit reproduces γ_binned = 1.103 and its logZ on an independent stack (L0-G2)
- **Status:** `proposed (UNCERTIFIED — external)`
- **Criterion (pre-registered):** |γ − 1.1032| ≤ 0.017 AND ΔlogZ(steep−low) < 0 (v3b-low per-basin refit, production-recipe mirror, seed 2).
- **Numbers:** γ_low = 1.1005 [1.0992, 1.1065], Δ = 0.0027 — **PASS**; logZ_low −4770.97 vs old-stack −4771.08 (0.11 nats across stacks AND machines); ΔlogZ(steep−low) = −29.36, sign+magnitude inside the P1 seed family (−28.9/−32.3/−31.2). σ_seed context (old stack, seeds 2/3/4): σ_seed(γ) = 0.00325 (low) / 0.00664 (steep), σ_seed(ΔlogZ) = 1.79 nats.
- **Evidence:** `data/results-perlmutter/l0g2_v3b_scene_seed2.{json,npz}`, `data/t02_t03_gate_eval.json`, `25_run_l0g2_scene.py`.
- **Doubt report:** (1) same frozen whitener bundle feeds both stacks (shared data product — this is a stack-implementation cross-check, not an independent noise model); (2) single seed on the new stack (σ_seed measured old-stack, n=3, ±46% χ-dist error quoted with every use); (3) the companion v2d-diagonal anchor arbitration closed PARTIAL-BY-VEHICLE-EXHAUSTION — see CGL2-C13.
- **Proposed by / on:** cgl2-linus campaign · 2026-07-16 · **Grader:** _pending_

### CGL2-C6 — B5-G2 reference-coverage finding: our OWN old campaign's frozen per-basin evidence reference has inadequate within-basin coverage (gate FAIL indicts the reference at least as much as the sampler)
- **Status:** `proposed (UNCERTIFIED — external)` — **scope note: this claim concerns the claude-giga-lens (OUR) campaign's P2c frozen reference. It is NOT a statement about any GIGALens-team artifact.**
- **Criterion (pre-registered):** basin ΔlogZ(steep−low) within 3·√(σ_boot² + 1.79²) nats (floor 5.36) of the P2c reference on foundry_v3b74.
- **Numbers:** Δ_meas = +141.90 ± 0.45_boot vs P2c +162.2 ± 1.8 ⇒ −20.30 nats = 11.0× the gate σ-scale — **FAIL as written** (comparability first verified line-by-line: same target, same q_k evidence construction, q_k cancels). Honest decomposition: per-basin offsets are +25.16 (low) and +4.86 (steep), BOTH positive; both λ=1 ensembles are γ-DISJOINT from the reference-chain support (low: ours [1.071,1.126] vs chains [1.265,1.381]) with two independent kernels agreeing (MAMS 1.0919 / MCLMC 1.0935) — the adaptive anneals reach higher-log_prob shelves the reference's fixed-step HMC(0.1, L=8) never visited. Minor-mode occupancy conclusion robust across all estimators (w_low ≤ 1.3e-60); absolute evidence calibration at the ±5-nat scale is NOT.
- **Evidence:** `data/b5_gate_eval.json`, `figs/b5_basin_overlay.png`, `data/results-perlmutter/b5_v3b74_{low_mams,low_mclmc,steep_mams}_seed2.{npz,json}`.
- **Doubt report:** n=1 seed on the measuring side — attribution (reference coverage vs vehicle) is decidable only with a seed repeat that sat outside the P2 cap; the reference's own on-disk smoke reproduces its Δ=162.7 under its kernel, so the discrepancy is real, not a transcription error. Lesson we are acting on ourselves: frozen MCMC-derived evidence references need coverage certificates before being used as gates.
- **Proposed by / on:** cgl2-linus campaign · 2026-07-21 · **Grader:** _pending_

### CGL2-C7 — WARNING for evidence work: unadjusted-MCLMC mutations inside tempered SMC inflate minor-mode evidence ×56; resampling does NOT launder the bias
- **Status:** `proposed (UNCERTIFIED — external)`
- **Criterion (pre-registered):** B5-G3 predicted MCLMC minor-mode distortion 1.5–3×; B0 MCLMC-vs-MAMS bias screen (>3σ ⇒ demotion).
- **Numbers:** ΔlogZ_low(MCLMC−MAMS) = +4.03 ± 0.49_boot nats (8.3σ_boot from zero) ⇒ minor-mode weight inflated **×56 [CI95_boot 22–146]** — 6.0σ_boot above the pre-registered ln3 ceiling. Within-basin γ essentially untouched (Δ 0.0016): the bias lands in the EVIDENCE, not the location. B0 screen: 30σ ⇒ MCLMC demoted to diagnostic/cost-frontier-only; MAMS (Metropolis-adjusted ⇒ exact π_λ invariance) is the evidence kernel.
- **Evidence:** `data/b5_gate_eval.json`, `data/smc_b0_report.json`, `data/results-perlmutter/b5_v3b74_low_{mams,mclmc}_seed2.*`.
- **Doubt report:** frozen clauses conflict on this measurement — under the imported cross-target σ_seed=1.79 convention the 4.03-nat shift sits inside the 5.56-nat MAMS repeatability band, so the "launders" falsifier technically fires; n=1 seed per kernel; decidable only with a MAMS seed repeat on this target. Directly relevant to (but measured independently of) your MCLMC sampler work: if you ever wrap MCLMC in an annealing/SMC layer for logZ, adjust it or budget for this bias channel.
- **Proposed by / on:** cgl2-linus campaign · 2026-07-19/21 · **Grader:** _pending_

### CGL2-C8 — X2 SBC: the (vendored, reduced-budget) pipeline class is severely rank-miscalibrated in the lens-light photometric block; mass block fails uniformity via U-shapes; 2/32 fits healthy
- **Status:** `proposed (UNCERTIFIED — external)` — ranks are the deliverable; interpretation and verdict are YOURS.
- **Criterion (pre-registered):** severe = |rank-location z| > 5 anywhere (the checkpoint falsifier; it tripped); uniformity χ² p > 0.01/param.
- **Numbers (N=32 prior-matched mocks, MAP→SVI→HMC at reduced A16 budgets):** LL.Ie z=−5.27 (p=7.6e-11), LL.R_sersic z=+5.76 (p=1.5e-10), LL.n_sersic z=+4.75 — one-sided, sign-coherent pile-ups; mass block worst |z|=2.25 but 6/8 fail uniformity via U-shapes (EPL.gamma p=1.4e-8); source block mildest (all |z| ≤ 1.8); healthy fits 2/32.
- **Evidence:** `research/x2_sbc_gift.md` (full claims register X2-C1..C4 in your format, incl. mandatory doubt reports), `data/x2_sbc.json`, `figs/x2_rank_hist_grid.png`, harness `30_sbc_gift.py` (turnkey for an N≥64 rerun at your reference settings).
- **Doubt report (summary):** 30/32 fits fail our health rule — and our own glass-house precedent (old-campaign E1c: γ-rank p=6.5e-5 that FLIPPED to p=0.53 on healthy-only deep reruns) shows sampler pathology can fake rank failures; reduced MAP/SVI budget and the plug-in error map are alternative channels; NOT separable from this run. This is a statement about the pipeline CLASS at this budget on our certified port — NOT about your production runs at reference settings (those were infeasible on A16 memory).
- **Proposed by / on:** cgl2-linus campaign · 2026-07-16 · **Grader:** _pending (GIGALens team)_

### CGL2-C9 — **[SIGN-OFF GATED]** Validity question on the frozen SBC set: `100SystemsStandard80px.npz` was generated from a NARROWER prior than today's modeling prior, so SBC on the frozen set would fail by construction
- **Status:** `proposed (UNCERTIFIED — external)` — **posed to you as a QUESTION, not a finding of fault; internal memo material only, publication-gated on your sign-off.**
- **Criterion:** SBC validity precondition (Talts et al.): truth-draws ~ fitting prior. No threshold — a documentation-derived precondition check.
- **Basis (your own provenance, read from our local mirror @eb2a09b6):** `experiments/why_hard_to_sample/t13_resim.py` STEP 0 ("GENERATION prior … NARROWER than today's MODELING prior", attic/Linus-FourSim.ipynb cell 13, PRNGKey(0)) and `gigalens_research/simtests/experiments/gl2_sersic.py` (`gl2_simulation_prior` vs `gl2_inference_prior == make_default_prior`). Our X2 arm therefore ran on prior-matched REGENERATED mocks, not the frozen npz.
- **Evidence:** `research/x2_sbc_gift.md` §Validity precondition; your files cited above.
- **Doubt report:** the prior mismatch may be intentional and known to you (robustness-to-misspecification testing is a legitimate design); the frozen npz itself is absent from our mirror so we verified provenance from code/notebook text, not the file; if the generation prior is documented somewhere we missed, this question dissolves. The question for you: is prior-matched regeneration the SBC substrate you want, or is there a matched-prior npz we should use?
- **Proposed by / on:** cgl2-linus campaign · 2026-07-16 · **Grader:** _pending (GIGALens team)_

### CGL2-C10 — Stationarity of our drizzle noise-model class is REJECTED on the real field; the working hypothesis for the 1.103 over-correction is noise-model-CLASS misspecification
- **Status:** `proposed (UNCERTIFIED — external)` — our-stack result; included because it bounds what the ported CorrelatedImageData can claim.
- **Criterion (pre-registered):** T0.4-1 per-block kernel homogeneity, 2σ blockwise + calibrated p.
- **Numbers:** money product v3b max|z| = 3.67, calibrated p = 0.010; replicated spatial pattern; cross-block ρ(0,1) spread 16.4× the drizzle-registration envelope — **the stationary kernel class behind γ_binned=1.103 is provably violated by the field.** Companions: spectrum-flooring cure FALSIFIED (rejected by 3.3k–33k nats, T0.4-2); real-space head-to-head prefers γ≈1.29 over both 1.433 and 1.103 while the whitened metric inverts this (T0.4-3); companion-mask discriminator EXONERATED the companion (shift −0.0021).
- **Evidence:** `data/t04_stationarity*.json`, `data/t04_lambda_arm.json`, `data/t04_realspace_headtohead.json`, `research/t04_free_checks.md`, `data/t02_t03_gate_eval.json`.
- **Doubt report:** the confirmatory injection-recovery (T1.1) came back CONFOUNDED by scene-subtraction residue (control arm failed high AND sick; construction implicated by the pre-registered honesty clause) — the mechanism has a directional differential in its favor (corr−diag −0.0526 on same data, no error bar) but is NOT confirmed; design implication carried into the port: CorrelatedImageData keeps a pluggable whitener seam for a locally-stationary class.
- **Proposed by / on:** cgl2-linus campaign · 2026-07-15 · **Grader:** _pending_

### CGL2-C11 — Fermat-potential sensitivity: the noise-model choice moves Δφ by 60–90% on a real drizzled lens (illustrative)
- **Status:** `proposed (UNCERTIFIED — external)` — report-only, prominently disclaimed.
- **Criterion:** report-only teaser (pre-declared).
- **Numbers:** median |fractional Δφ shift| **88%** anchor→correlated (10.7σ) and **61%** diag→correlated same-product (17σ) — vs the ~1% TDCOSMO-relevant scale.
- **Evidence:** `data/fermat_dt_teaser.json`, `research/fermat_dt_teaser.md`.
- **Doubt report:** NOT a TD lens; synthetic image pairs; the correlated posterior is the known over-correcting product (CGL2-C10) — the number motivates a coadd-covariance error-budget line item for TD work, nothing more.
- **Proposed by / on:** cgl2-linus campaign · 2026-07-15 · **Grader:** _pending_

### CGL2-C12 — **[SIGN-OFF GATED]** Carousel real-data cost row: NEITHER prior-seeded MC-SMC nor warm MAMS-alone converges on carousel33 (real MUSE cutouts) at campaign-affordable budget
- **Status:** `proposed (UNCERTIFIED — external)` — runs on YOUR unpublished cutouts; results-to-you-first honored; publication-gated.
- **Criterion (pre-registered):** D8/D9 descoped B1r cell + close-out framing pre-stated BEFORE the baseline ran (incl. the branch that did NOT fire: "any nonzero baseline ESS ⇒ decisive cold-start LOSS").
- **Numbers:** S1r (prior-seeded MAMS-SMC, N=128): 11.33 A100-h → λ=0.1506 @36 stages, 0 posterior samples — sampler HEALTHY (unique 92–104/128, accept 0.87–0.93), killed by cost ⇒ λ=1 infeasible under any campaign budget. S6br (warm MAMS-alone, budget-matched at S1r's realized 3,078,912 grads): 4.5-h wall fence fired in burn round 11/30 — sampling never started, ESS=0 (burn healthy, accept mean 0.887). Per-grad throughput comparable (~284k vs ~272k grads/h): target cost, not arm inefficiency. **Row: carousel-class real-data targets are out of reach for BOTH vehicles at these budgets** — consistent with your own min-ESS 12/16000 experience.
- **Evidence:** `data/b1r_decision_matrix.json`, `figs/b1r_s6br_traces.png`, `data/results-perlmutter/b1r128_carousel33_s1_seed2.PARTIAL.json`, `…_s6b_seed2.{json,npz}`, `research/checkpoint_b1_reduced.md`, `research/checkpoint_b1r_close.md`.
- **Doubt report:** burn-dominated comparison (would bias toward S1r — moot, both ESS exactly 0); 2-seed self-consistency and logZ repeatability were DESCOPED (stated plainly in the amendment); until-converged S6b estimated 17–20 h from wave-1 anchors, untested; comparison vs your carousel runs is possible at summary-statistics level ONLY (your posterior arrays are gitignored/absent from our mirror — the posterior-array ask is in the memo addendum).
- **Proposed by / on:** cgl2-linus campaign · 2026-07-20/21 · **Grader:** _pending (GIGALens team)_

### CGL2-C13 — Anchor arbitration closed PARTIAL-BY-VEHICLE-EXHAUSTION: every vehicle reproduces the KNOWN target-intrinsic difficulty of the 46-dim cond-1e14 v2d posterior; no stack discrepancy found
- **Status:** `proposed (UNCERTIFIED — external)`
- **Criterion (pre-registered):** two-stack anchor check (does 1.433 reproduce end-to-end on the scene API?) + worst-parameter rule R̂<1.05/ESS≥200 for any quotable γ.
- **Numbers:** four vehicles — L4 MC-SMC λ=0.45@~18 h; A100 MC-SMC λ=0.587@3.5 h (healthy, wall-capped); classic MAP→SVI→HMC ×2 settings R̂_worst 44.3 (escalated 11.7), all chains railed γ≈1.99 ⇒ NO-VERDICT. The single-stage pathology mirrors the old stack (R̂ 3.1; only the two-stage re-preconditioned recipe converges) — difficulty is target-intrinsic, not stack-specific. Premise-check value carried by F1–F6 (machine-precision likelihood parity) + L0-G2's CONVERGED cross-stack reproduction on the harder correlated target.
- **Evidence:** `research/checkpoints_l0.md`, `research/l0_anchor_arbitration.md`, `data/results-perlmutter/l0arb/*`, `11_arbitration_classic.py`, CAMPAIGN.md 2026-07-21 stage entry.
- **Doubt report:** a converged v2d scene-API posterior needs the two-stage recipe ported (~0.5 day + 2–4 A100-h, not spent); until then "1.433 reproduces at posterior level" is UNTESTED, and the warm classic arm inherits the warm-start basin by construction — alternative-basin coverage remains open.
- **Proposed by / on:** cgl2-linus campaign · 2026-07-21 · **Grader:** _pending_

### CGL2-C14 — **[SIGN-OFF GATED]** DSPL (B2) note: the pre-registered 0.103 minor-arm-mass band does not transfer across data realizations; dominant-arm posterior is sampler-independent; a 2σ suggestive (not conclusive) minor-arm under-coverage signal in the original coordinates
- **Status:** `proposed (UNCERTIFIED — external)` — target/coords derive from your DSPL work; publication-gated.
- **Criterion (pre-registered):** |m̂(Om0<0.146) − 0.103| ≤ 0.045 in ORIGINAL pathological coords; control dspl20_ratio reproduces Run A.
- **Numbers:** orig m̂ = 0.000 (0/512) — outside the band; BUT the exact-reparameterization control (which cannot suffer coordinate mode-death) measures THIS realization's true arm mass at 5/512 = 0.98% ± 0.43%, itself far outside 0.103±0.045 ⇒ the band is realization-specific (pre-registered branch 2 executed as written; a decisive B2′ needs re-registration against the control's own arm mass). Orig vs control minor-arm: Fisher one-sided p = 0.031 (~2σ, n=1 seed each — below our 3σ convention). Dominant arm statistically identical (Om0 med 0.4702 vs 0.4701). Evidence-error datum: ΔlogZ(orig−ratio) = +3.06 ± 0.35 nats on identical data under exact reparameterization (true Δ≈0) ⇒ σ_boot understates cross-parameterization logZ error.
- **Evidence:** `data/b2_gate_eval.json`, `figs/b2_om0_posterior.png`, `figs/b2_ratio_control.png`, `data/results-perlmutter/b2_dspl20_{orig,ratio}_s1_seed2.*`.
- **Doubt report:** n=1 seed per arm at N=512; fresh data realization (data_seed=0) ≠ your unreproducible Run-A baseline — location comparisons are vs your summary stats only; SMC sanity clean both arms (no resample collapse — if the minor arm died in orig coords it died by weight decay).
- **Proposed by / on:** cgl2-linus campaign · 2026-07-19 · **Grader:** _pending (GIGALens team)_

---

## Open questions

- CGL2-C9's question to the team: intended prior mismatch on the frozen SBC set, or should a
  matched-prior set exist? (Blocks nothing; X2 used prior-matched regeneration.)
- Posterior-array transfer for a draw-level carousel comparison (CGL2-C12 doubt report).
- Attribution seed-repeats: C6 (reference-vs-vehicle) and C7 (MAMS repeatability on-target) —
  each ~1–2 A100-h, outside the closed budget; offered, not scheduled.
