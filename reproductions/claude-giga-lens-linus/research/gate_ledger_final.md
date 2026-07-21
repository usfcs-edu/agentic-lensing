# Gate ledger — FINAL closure sweep (2026-07-21, P5 synthesis)

Every gate ever registered in this campaign, with its final status and a one-line
disposition. Compiled from: CAMPAIGN.md gate record + locked decisions + stage log,
`research/*.md` checkpoints, `data/*.json` gate evals, and a completeness cross-check
against **PLAN §6's full program list (P0–P4 + X-track + stretch)** so nothing silently
vanishes. Statuses are quoted from the authoritative record; items marked **proposed**
await orchestrator confirmation. Zero GPU; artifact-only. Companion:
`research/checkpoint_b2_prime.md` (the one gate re-scored in this sweep).

Legend: PASS / FAIL / NO-VERDICT / PARTIAL / RETIRED / NOT-EVALUABLE / MIXED — as
recorded; DANGLING = no final disposition anywhere in the record (section B).

## A. Gates with final dispositions

| Gate | Phase | Final status | One-line disposition | Artifact |
|---|---|---|---|---|
| F1 forward-image parity | P0 | **PASS** 5.7e-15 | machine-precision cross-stack parity | data/parity_report_scene.json |
| F2 design columns | P0 | **PASS** 3.9e-15 | ditto | data/parity_report_scene.json |
| F3 diag masked loglik+χ² | P0 | **PASS** 7.5e-9 | ditto (≤1e-8) | data/parity_report_scene.json |
| F4 marg-loglik grad | P0 | **PASS** 1.5e-11 | ditto | data/parity_report_scene.json |
| F5 (+F5b) delta-kernel CorrelatedTerm | P0 | **PASS** 5.8e-11 / 0.0 | correlated term ≡ stock at delta kernel | data/parity_report_scene.json |
| F6 Occam −½logdetA | P0 | **PASS under signed exception** (v2d 1.19e-11; v3b 4.48e-10 ≤ 7.76e-10) | restated vs fp128 truth w/ cond-scaled tol (Benson sign-off 2026-07-15); original numpy-slogdet form had FAILed at the f64 noise floor | data/parity_report_scene.json (products.*.F6) |
| F7 flat-z roundtrip audit | P0 | **PASS 0.0** | 46-name bijection clean | data/parity_report_scene.json |
| F8 parity battery on PM native env | P0/deploy | **PASS** (all hard gates, job 55980038) | scene-API stack certified on both machines | data/results-perlmutter/parity_report_scene_pmnative_55980038.json |
| 03-A corr term vs dense-Cholesky | P0 | **PASS** 5.5e-12 | exact-reference validation | data/correlated_term_validation.json |
| 03-B delta identities | P0 | **PASS** 9.1e-13 / 0.0 | ditto | data/correlated_term_validation.json |
| W whitener bundles revalidation | P0 | **PASS** (v2d/v3b/v3 admissible) | e_op ≤ 0.02 strict; v2d_relaxed inadmissible-by-design | data/whitener_manifest.json |
| B0 MC-SMC correctness | P0 | **PARTIAL** (see §C stale-cell flag) | lens-relevant content ALL PASS (mix2 evidence, illcond46, adapters 11/11, MAMS-vs-MCLMC bias screen ⇒ MCLMC demoted per pre-registered rule); funnel10 logZ deficit FAIL = pre-existing adaptive-tempered-SMC class limitation reproduced in the old validated stack, carried as caveat; σ_boot-understatement known issue carried | data/smc_b0_report.json + P0-exit-decision prose in CAMPAIGN.md |
| T0.2 seed-repeat certification | P1 | **PASS** (kill not tripped) | σ_seed(γ)=0.0033/0.0066, σ_seed(ΔlogZ)=1.79; γ=1.1032 certified at σ_tot=0.0086 | data/t02_t03_gate_eval.json |
| T0.3 companion-mask discriminator | P1 | **EVALUATED — COMPANION EXONERATED** | shift −0.0021 (statically zero); noise-model class stays prime suspect | data/t02_t03_gate_eval.json |
| σ_seed FINALIZATION | P1 | **FINALIZED** | downstream thresholds re-parameterized from measured σ_seed (ledgered, not a goalpost move) | data/t02_t03_gate_eval.json |
| T0.4-1 stationarity | P1 | **REJECTED** (finding) | stationary kernel class behind γ=1.103 provably violated (p=0.010, verifier clean) | data/t04_stationarity*.json |
| T0.4-2 λ-arm flooring | P1 | **EVALUATED — flooring hypothesis FALSIFIED** | data reject flooring by 3.3k–33k nats; pathology localized to nonstationary background component | data/t04_lambda_arm.json |
| T0.4-3 real-space head-to-head | P1 | **RECORDED** (report-only by design) | real-space prefers γ≈1.29 over both 1.433 and 1.103; whitened metric inverts it | data/t04_realspace_headtohead.json |
| E3 kernel scan | P1 | **DEFERRED-BY-PLAN, never funded → lapses at program close** | written justification in PLAN §6 P1; remains THE standing caveat on absolute-γ claims (must appear in P5 caveats) | plans/PLAN.md §6 |
| T1.1 injection-recovery (D7 pool) | T-track | **NO-CONFIRM / NO-EXONERATE — CONFOUNDED** | positive-signed result outside both zones + control fails high AND sick ⇒ injection construction implicated (scene residue) per pre-registered honesty clause; n=3 | data/t11_gate_eval.json, research/t11_injection_recovery.md |
| T1.1b-G0 residue-mask entry gate | T-track | **FAIL — STOPPED at build gate** (0 A100-h) | kept-dof loss 85.8% ≫ 40% ⇒ information-starved; data-side injection redesign queued as follow-on | data/t11b_residue_mask_report.json |
| X1-G0 profile-fork entry gate | P4 | **FAIL — hypothesis structurally dead** | 24/24 variants non-monotone; kill executed as written ⇒ D7 retires P4 | data/x1_g0_effective_radii.json, research/x1_g0_mechanism_check.md |
| X1-G1 ΔlogZ(BPL−EPL) | P4 | **RETIRED with P4 (D7)** | ~15-nat placeholder never finalized, never evaluated | CAMPAIGN.md σ_seed-finalization row |
| X1-G2 γ_loc(θ_E) spread | P4 | **RETIRED with P4 (D7)** | never run | — |
| X1-G3 r_break/r_core railing | P4 | **RETIRED with P4 (D7)** | never run | — |
| Fermat Δt teaser | X-track | **DONE** (report-only by design) | 88%/61% fractional Δφ shifts vs ~1% TDCOSMO scale; illustrative only | data/fermat_dt_teaser.json, research/fermat_dt_teaser.md |
| B1 mock arms | P2 | **NOT-EVALUABLE** (final by amendment) | all 3 arms TIMEOUT zero-artifact (11.02 h); mock arms retired FOREVER by the B1-REAL-DATA-LANDED amendment — superseded by B1r, so NOT dangling | research/p2_wave1_postmortem_redesign.md |
| B1r decision matrix (D8/D9) | P2b | **EVALUATED — NEITHER CONVERGES at matched budget** | S1r λ=0.151/0 samples at 11.33 h; S6br wall-fence at 41.5% of B*, ESS 0 ⇒ carousel-class real targets out of reach for BOTH vehicles at campaign budgets; cell CLOSED (D9) | data/b1r_decision_matrix.json, research/checkpoint_b1r_close.md |
| B2 (as written) | P2 | **FAIL-AS-WRITTEN → SUPERSEDED by B2′** | m̂=0.000 vs 0.103±0.045, but pre-registered falsifier control proved the band UNCALIBRATED for this realization (NOT-DECIDABLE-AS-REGISTERED) ⇒ mandated re-registration | data/b2_gate_eval.json |
| **B2′ (re-registered, THIS SWEEP)** | P2 | **PASS-UNDERPOWERED (CONSISTENT)** | Fisher two-sided 0/512 vs 5/512 p=0.0619 ≥ α=0.01 (rule + attainable-significance clause declared before scoring: with 5 total minors no outcome could fail at α); under-coverage ≤ ~×3 not excluded; headline stands at "no detected coordinate pathology" strength | research/checkpoint_b2_prime.md, data/b2_gate_eval.json |
| B3 hundred_systems-8 | P2 (phoenix) | **NOT-EVALUABLE-WITHIN-BUDGET** (final; fence worked as designed) | 4/4 reference posteriors complete; 0/4 SMC arms inside the pre-registered 5 h/arm fence ⇒ gates unevaluable; readable result = the COST row (scene-API MC-SMC N=512 > 5 h/target on L4); rerun path documented, unfunded | data/b3_cells.json, figs/b3_cost.png |
| B4 T2 foundry_marg46 | P2 | **DANGLING → proposed NOT-EVALUABLE-WITHIN-BUDGET** (§B.1; CAMPAIGN.md row appended, orchestrator to confirm) | never rerun after wave-1 zero-artifact TIMEOUT | see §B.1 |
| B5-G1 both basins at λ=1 | P2 | **PASS (both basins)** | low λ=1 @31 stages γ 1.0919; steep λ=1 @22 stages γ 2.5552 (attempt #4, chunk 32) | data/b5_gate_eval.json |
| B5-G2 basin-ΔlogZ vs P2c | P2 | **FAIL AS WRITTEN** (comparability first resolved) | Δ_meas +141.90 vs +162.2, 11.0× gate σ-scale; honest decomposition: BOTH per-basin offsets positive + both ensembles γ-disjoint from the reference chains ⇒ the FAIL indicts the frozen reference's within-basin coverage at least as much as S1; n=1 seed, attribution needs out-of-cap seed-repeat | data/b5_gate_eval.json |
| B5-G3 MCLMC laundering | P2 | **AMBIGUOUS-AT-FROZEN-THRESHOLDS** (final; n=1) | +4.03±0.49 nats = ×56 minor-mode inflation exceeds the 1.5–3× band (bias real, not laundered at point estimate) BUT sits inside the imported 5.56-nat MAMS repeatability band (laundering falsifier technically fires) — the two frozen clauses conflict; occupancy conclusion robust (w_low ≤ 1.3e-60 every estimator) | data/b5_gate_eval.json |
| S4/S5 LAPS arms | P2 | **OMITTED-BY-BRIGHT-LINE** (pre-declared, final) | their unpublished research code never deployed (PLAN §8.2); benchmark baselines = S6b + published-class kernels, stated in the B1 checkpoint | CAMPAIGN.md wave-scope entry 2026-07-16 |
| S7 flow-MAMS | P2 | **DANGLING → proposed NOT-RUN — PREREQUISITE-UNMET** (§B.2) | repeatedly deferred "post-B1"; the prerequisite (a B1 λ=1 ensemble) never came to exist | see §B.2 |
| L0 port parity + wiring | P3 | **PASS** | correlated port licensed via F5/03-A/03-B + whitened logp/grad parity (worst marg_dlogL 2.4e-8); wiring sanity S1–S3 PASS pre-production; remat exactness gate PASS (diff exactly 0.0) | data/parity_report_scene.json, data/l0_sanity_report.json, data/l0_remat_gate.json |
| L0 anchor arbitration | P3 | **CLOSED — PARTIAL-BY-VEHICLE-EXHAUSTION / NO-VERDICT** | 4 vehicles exhausted (MC-SMC λ=0.45@18h L4, λ=0.587@3.5h A100 healthy-but-infeasible; classic arm unconverged ×2 = the KNOWN single-stage pathology reproduced on scene API); γ not quotable; premise-check value carried by F1–F6 parity + L0-G2's converged cross-stack reproduction; two-stage-recipe port left as USER follow-on | research/checkpoints_l0.md, research/l0_anchor_arbitration.md, CAMPAIGN.md 2026-07-21 stage entry |
| L0-G2 scene-API v3b refit | P3 | **PASS** | γ Δ0.0027, ΔlogZ sign + magnitude in seed family, logZ Δ0.11 nats cross-stack; CorrelatedImageData LICENSED; γ=1.103 reproduced on two stacks | data/results-perlmutter/l0g2_v3b_scene_seed2.{json,npz} |
| L1 full SBC (16 drizzle mocks) | P3 | **DANGLING → proposed NOT-RUN — NEVER-STARTED** (§B.3) | no design checkpoint, no deferral decision on record | see §B.3 |
| L2 supersampling decomposition | P3 | **DANGLING → proposed NOT-RUN — NEVER-STARTED** (§B.4) | no record since PLAN | see §B.4 |
| X2 SBC gift | X-track | **DELIVERED** (claims register; verdicts theirs) | pre-registered "mild at worst" falsifier TRIPPED ⇒ X2-C1 severe LL-block miscalibration (|z| up to 5.76) proposed UNCERTIFIED-external; N=32 under-delivery + 2/32-healthy caveat declared; glass-house E1c disclosure included | research/x2_sbc_gift.md, data/x2_sbc.json |
| X3 Vela ladder prototype | X-track | **DANGLING → proposed NOT-RUN** (§B.5) | never mobilized | see §B.5 |
| Hessian-stage byproduct | X-track | **DANGLING → proposed NOT-RUN** (§B.6) | never started | see §B.6 |

## B. DANGLING — gates/program items with NO final disposition on record

1. **B4 (T2 foundry_marg46, cond-1e14 preconditioning question).** Wave-1 verdict was
   explicitly non-final ("NOT EVALUABLE … the one-line fix is a precondition for ANY B4
   resubmission"); the writer-defect precondition WAS fixed + regression-locked (RC3,
   2026-07-16) but no resubmission was ever funded, and no closure row ever landed.
   **Proposed disposition: NOT-EVALUABLE-WITHIN-BUDGET** — measured cost lower bound 3.5 h,
   realistic 3–5 h, vs P2 headroom 1.55 h at program close (22.45/24), never rerun; the
   scientific question (does per-λ ensemble preconditioning replace the two-stage recipe on
   cond~1e14) stays **OPEN** and is NOT absorbed by any other cell — nearest datum is the
   arbitration classic-arm's reproduction of the single-stage-HMC pathology (target
   difficulty context, not a B4 answer). Two-stage-seeded variant remains OFF-PROTOCOL
   (would be a new arm B4b). Row appended to CAMPAIGN.md gate record marked
   **proposed, orchestrator to confirm** (this sweep's second owned row).
2. **S7 minimal flow-MAMS (D4 user-included arm).** Pre-declared "not this wave" at wave-1,
   "post-B1 as before" at B1r — and B1 never produced the λ=1 ensemble its comparison arm
   needs (mock arms zero-artifact; B1r PARTIAL at λ=0.15). P2 closed 22.45/24 vs S7's ~4 h
   plan line. **Proposed: NOT-RUN — PREREQUISITE-UNMET + BUDGET-CLOSED.** Consequence:
   stretch item 1 (flow-within-SMC synthesis, conditioned "only if S7 + B1 both land
   cleanly") is MOOT. D4's "results to the team first" never triggered (nothing produced).
3. **L1 full SBC (P3; 16 binned drizzle mocks, production SMC @96).** Never started; no
   checkpoint, no deferral. X2 is NOT a substitute (different target class, diagonal plug-in
   likelihood, classic pipeline, phoenix A16) — the calibration-certificate slice of the
   PLAN P5 report has no L1 data behind it. **Proposed: NOT-RUN — NEVER-STARTED** (P3 closed
   at 5.21/17; the freed P3 budget went to the arbitration vehicles). P5 must not imply a
   production-SMC coverage certificate exists.
4. **L2 supersampling decomposition (P3 arms a/c, sub-pixel-PSF term).** Never started; no
   record since PLAN. **Proposed: NOT-RUN — NEVER-STARTED**; the sub-pixel-PSF caveat on the
   1.433 anchor product remains open and belongs in P5's caveat list (it was also D7's
   pointer: "noise/likelihood treatment AND PSF representation").
5. **X3 evidence-scored Vela ladder prototype.** Phoenix-only, 0 A100-h — not
   budget-blocked; simply never mobilized (fronts went to B3/X2/L0). **Proposed: NOT-RUN**;
   the memo-offer framing (never applied to their staged campaign) lapses to the P5 handoff
   list as an offer, not a result.
6. **Hessian-stage byproduct (restore HessianSurrogateStage).** Never started. **Proposed:
   NOT-RUN**; carries to P5 handoff as the owed offer-PR item, explicitly unbuilt.
7. **PSF-marginalization MVP (not a gate — D7 promotion).** D7 promoted it "toward core"
   on the freed P4 pool ("First claim … T1.1 …, then PSF-marg MVP"); T1.1 consumed 7.80 h
   and D8 reallocated the remaining 10 h to P2b — PSF-marg was never designed or run and no
   decision ever closed it. **Proposed: NOT-RUN — POOL-REALLOCATED (D8)**; flag so D7's
   promotion language doesn't read as delivered work.

## C. Bookkeeping flags (dispositions exist; record cells stale — orchestrator fixes, not this sweep)

- **B0 gate-record cell still reads "PENDING"** while the P0-exit-decision prose directly
  below the table records the full split verdict (lens-relevant PASS / funnel10 FAIL
  carried / MCLMC demoted / σ_boot caveat). Recommend the orchestrator update the cell to
  PARTIAL pointing at that prose; §A row above carries the reconciled status.
- **P2 kill criterion** ("S1 fails B5 both-basins AND B1 diversity") — **NOT TRIPPED**:
  B5-G1 passed both basins; the B1 diversity clause was never evaluable. Recorded so the
  bet's survival is explicit.
- **P0 exit criterion** was NOT met as written (F6 exception + funnel10); proceeding was a
  ledgered, flagged decision (UNCERTIFIED, for Benson) — already on record, no action.

## D. Completeness cross-check vs PLAN §6 (nothing silently vanished)

- **P0:** F1–F7 ✓, B0 ✓ (+F8, 03-A/B, W added during execution, all closed).
- **P1:** T0.2 ✓, T0.3 ✓, T0.4-1/2/3 ✓, E3 kernel scan ✓ (deferred-by-plan, §A);
  σ_seed finalization ✓; P1 kill not tripped.
- **P2 cells:** B1 ✓ (mock arms retired; real-data successor B1r closed by D9),
  B2 ✓ (→ B2′ closed this sweep), B3 ✓, B4 → §B.1 (the one GPU-gate dangler),
  B5 G1/G2/G3 ✓.
- **P2 arms:** S1 ✓ (the bet — exercised across B1r/B2/B3/B4-attempted/B5),
  S2 MCLMC ✓ (ran as B5's diagnostic arm per the B0 demotion), S3 HMC-SMC — no gate of its
  own; it was the kill-fallback evidence layer and the kill never tripped (nothing
  dangling), S6a MCLMC-alone — mooted by the B0 demotion (no gate attached), S6b ✓ (ran
  wave-1 TIMEOUT; budget-matched successor S6br closed by D9), S4/S5 ✓ (omitted by bright
  line), S7 → §B.2.
- **P3:** L0 port+parity ✓, anchor arbitration ✓ (closed NO-VERDICT), L0-G2 ✓,
  L1 → §B.3, L2 → §B.4.
- **P4:** X1-G0 ✓ (FAIL, kill executed = D7), X1-G1/G2/G3 ✓ (retired with P4).
- **X-track:** X2 ✓, X3 → §B.5, Fermat teaser ✓, Hessian byproduct → §B.6.
- **Stretch pool:** item 1 flow-within-SMC — moot (§B.2); item 2 PSF-marg MVP — §B.7;
  item 3 second real system (DESI J238) — never claimed, pool partially reallocated by D8
  (8 h to P2b), lapses with program close.
- **D7/D8/D9 additions:** T1.1 ✓, T1.1b-G0 ✓, B1r decision matrix ✓, S6br ✓ (inside B1r
  row), B2′ ✓ (this sweep).

**Net dangling count: 6 gates/arms (B4, S7, L1, L2, X3, Hessian) + 1 decision-promoted item
(PSF-marg MVP)** — all proposed-closed above; only B4's proposed row is appended to
CAMPAIGN.md (task ownership), the rest await orchestrator adoption from this file.
