# Parity program findings log

Goal (approved plan, 2026-07-11): determine whether a machine system can grade
DESI-resolution lens candidates at trained-human level, whether the data exists to
decide it, and how to prove it. Phase A = measure the human baseline from published
data alone (no new grading campaign). Primary endpoint (user decision): E2
truth-referenced non-inferiority; E1 agreement is secondary. Site-report-first.

## Phase A — the human baseline (DONE, 2026-07-11)

Report: `papers/human_baseline.tex` → site `current/lensjudge-human-baseline/`
(registered in `site/build_reports.py` EXTRA_REPORTS + mkdocs nav).

### A1 — Paper II per-grader data recovered
`fetch_vizier_paper2.py`: VizieR `J/ApJ/909/27/cand` (n=1,312) preserves `Score`
(2-grader average) and `delSc` (absolute difference) — the ONLY per-grader signal
published anywhere in the program. Unordered pairs recovered as {Score±delSc/2};
parity check passes 1,310/1,312 (2 impossible rows flagged `pair_ok=False`:
DESI-094.5639+50.3059, DESI-227.9362+06.5090). 100% name-join vs local catalog.

### A2 — first inter-grader reliability statistics (`intergrader_stats.py`)
n=1,310 accepted candidates; 2,000-bootstrap CIs; ALL upper bounds (truncated at
acceptance, Score>=2.0):
- exact score agreement 0.554 [0.527, 0.580]; one-step 0.383; two-step 0.063
- symmetrized QWK 0.420 [0.371, 0.468] (= Krippendorff interval alpha)
- individual grader vs consensus QWK 0.776 [0.757, 0.793] (upper bound: grader is
  half the consensus) — THE E1 parity target
- within published grade A, both graders "4" only 46.5% [40.2, 53.2]
- 48/1,310 (3.7%) accepted with a {1,3} pair — one grader at reject level
- score-pair table: {2,2}=461 {2,3}=387 {3,3}=165 {3,4}=115 {4,4}=100 {2,4}=34 {1,3}=48

### A3 — first grade-stratified confirmation table (`grade_purity_table.py`)
Truth tables (7 campaigns, 551 targets) in `parity/data/external/*.csv`, each
machine-parsed from arXiv LaTeX/VizieR/Zenodo (workflow agents + inline AGEL parse;
all structurally audited; Foundry-IV/HSC×DESI counts spot-checked vs abstracts).
Crossmatch <2" vs master_candidates.csv (4,354 unique graded; regenerating
crossmatch_external reproduced the historical 104 SuGOHI / 24 Euclid matches after
staging sugohi_full.csv — NOTE: `_xmatch_sugohi` now prefers the full A/B/C catalog).
Results: 162 followed-up, 130 decided —
- A: 79/83 = 0.952 [0.889, 0.984] (Jeffreys)  ← NOT 100%: 4 grade-A refuted by MUSE
- B: 23/25 = 0.920 [0.767, 0.983]
- C: 20/22 = 0.909 [0.739, 0.981]  ← grade C is not mostly junk
Purity flat across grades on the curated follow-up mix. All 4 refuted A's are
arc-at-wrong/same-redshift mimics (morphology-only failure mode). The 6 AGEL
imposters are all DES/Jacobs-provenance (nearest Huang candidate >1000").

### A4 — combined baseline (`human_baseline_summary.py`)
Adds cross-team QWK (0.288 [0.121,0.444] n=103 SuGOHI; 0.165 n=24 Euclid) and
literature anchors. IMPORTANT correction: `claudenet/campaign/report/
consensus_full_737.csv` `my_grade`/`first_grade` are BOTH machine grades (visual
workflow first pass + skeptic gate; see claudenet/campaign/230_collect_visual.py) —
NOT two human passes. Excluded as a human anchor.

### Machine-parity targets now defined
- E1 (agreement): QWK vs consensus; anchors 0.29 (independent team) / 0.42 (the two
  graders' mutual) / 0.776 (team member, upper bound). LensJudge v1 sat at ~0.
- E2 (truth): paired non-inferiority AUC vs confirmed/refuted on followed-up
  candidates; today n=130 decided but only 8 refutations → estimation-only; the
  grade-agnostic DESI STP (2,157 systems) + Euclid Q1/DR1 will power a verdict.

## Standing data notes
- Euclid Q1 catalog staged: `reproductions/euclid-q1/data/raw/q1_discovery_engine_
  lens_catalog.csv` (2,584 rows; catalog only, no cutout zips).
- SuGOHI full catalog staged: `reproductions/aion-1/data/raw/sugohi/sugohi_full.csv`
  (3,961; sentinel -99 for missing z; 80 with both spec-z) + A/B parquet for the
  aion-1 pipeline.
- `human_ceiling.py` now reads the reproducible `xmatch_sugohi.csv` (renames
  grade→desi_grade) instead of the unreproducible `desi_x_sugohi_matches.csv`.
- Foundry-IV arXiv ID is 2509.18078 (2509.18087 is an unrelated paper; repo script
  01_build_confirmed_catalog.py cites the wrong ID).
- AGEL false-positive coords parsed from truncated sexagesimal names (~7" error) —
  fine for the >1000" non-match conclusion, do not use for tight crossmatch.

## Phase B — truth-anchored parity bench + power check (2026-07-11)

### Bench built (`eval/build_parity_bench.py`, seed 2026, SHA-pinned)
- **arm1** (paired human-vs-machine, DESI res): 162 truth-followed candidates
  (130 decided: 122 lens / 8 refuted; grades A=99/B=30/C=33) + 750 non-truth
  augmentation rows (250 grade-D, 500 random). `outputs/parity_bench_arm1.csv`.
- **arm2** (machine-at-scale): all 2,584 Euclid Q1 SLDE-A positions as ls-dr10
  cutout targets (A=309/B=267/C=2008, expert_score + votes carried) + 21 decided
  SLDE-B objects outside the Q1 catalog = 2,605 rows; 44 truth-labeled
  (38 lens / 6 refuted, tier=modeling). `outputs/parity_bench_arm2.csv`.
- Cutout resolution verified 8/8 on both arms (Euclid field serves via ls-dr10).
- Cutout staging COMPLETE: arm1 162/162, arm2 2,605/2,605 (cache/cubes/); the
  first-pass failures were endpoint rate-limit collisions — a 3-worker retry
  cleared every one. Fetch politely (<=3-4 workers) against legacysurvey.org.

### Euclid truth: two integrity findings that shrink the expected arm2 truth set
1. **The "NISP spectroscopy of Q1 lenses" paper (arXiv:2604.02726) is a
   single-author paper WITHDRAWN at v6** — 385/440 of its "deflector redshifts"
   are photo-z, its own v5 validation reports 35% blind deflector-z recovery, and
   its per-object catalog was never properly released. The extraction
   (`parity/data/external/euclid_q1_nisp.csv`, 473 rows, every row's source_ref
   carries the warning) is kept for REFERENCE ONLY and is deliberately not
   consumed by build_parity_bench. The deep-research brief's "178 NISP z-pairs /
   10 projected" numbers traced to this paper's v1 — treat as unsupported.
2. **SLDE-A's per-object modeling verdicts are unpublished**: the paper gives only
   aggregates (374 modeled, 315 judged lens, 59 judged non-lens); the Zenodo
   release's unsuccess.zip mixes failures/non-lens/ambiguous with no verdict
   field. Only SLDE-B (arXiv:2503.15325) publishes per-object verdicts:
   38 modeling+expert-confirmed, 6 ruled out, 11 model-failed (inconclusive).
   So recoverable per-object Euclid truth TODAY = 44 decided, not ~390.
   Extraction: `parity/data/external/euclid_q1_modeling.csv` (555 rows);
   rerunnable builders preserved in `parity/external_builders/`.

### Power check (`parity/power_check.py` -> outputs/parity_power_check.csv)
- **HEADLINE: AUC(human ordinal grade vs confirmed/refuted truth) = 0.577
  [0.396, 0.757] on the 130 decided rows — statistically consistent with chance.**
  High purity everywhere + flat across grades = the *selection* did the work; the
  fine grade adds ~no incremental truth signal on the followed-up pool. (Caveats:
  curated follow-up compresses range; 3 score levels; 8 negatives.)
- Paired-ΔAUC noise floor (equal-AUC binormal, rho=0.6): today SE=0.095 → min
  non-inferiority margin ~0.24 (coarse only); with ~30 refuted → 0.13; Euclid-era
  (222/60) → 0.10; reader-study scale (250/100) → 0.078. **Q2 answer: estimation
  yes, small-margin verdict no.** With the Euclid truth findings above, the
  near-term refuted-negative ledger is: 8 (arm1 spectroscopic) + 6 (SLDE-B
  modeling) = 14 per-object refutations in existence. The verdict-grade test
  genuinely awaits the grade-agnostic DESI STP outcomes (2,157 systems) and/or
  Euclid-Collaboration-published per-object adjudications (DR1 era).
- **First real paired E2 comparison** (CNN p_meta/probability as today's machine,
  n=105 with both predictors, 99/6): AUC human 0.460 [0.278,0.677], CNN 0.641
  [0.362,0.898], paired ΔAUC (CNN−human) = +0.180 [−0.029, +0.419]. Direction
  favors the machine; not significant at 6 negatives. Note: 'score' mixes
  huang2021 ResNet probability (94 rows) with storfer p_meta (10) — per-catalog
  calibration caveat.

## Phase C — machine systems (2026-07-11, in progress)

### Leak-aware training pool (`parity/build_train_splits.py`)
69,228 rows (2,657 graded + 2,313 grade-D + 64,507 random after firewall), splits
train/valsel/gate stratified by source x grade, seed 2026. Firewall: 801 rows
excluded (769 by bench name incl. the arm1 augmentation strata, 32 by <2" position
vs arm2); audit asserts 0 collisions. Gate = frozen evidence set (139 graded /
60 D / 60 random), touched only for one-shot final numbers.

### C1a — frozen-CNN predictor, and a score-provenance trap
**TRAP (recorded so nobody refalls into it):** graded rows carry the REPRODUCTION
ensemble score (our_p_meta) while grade-D rows only have the PUBLISHED NeuraLens
probability. Mixing them across classes fabricates discrimination: gate HARD
(A/B vs D) AUC = 0.847 mismatched vs **0.646 [0.557, 0.734] like-for-like**
(published p on both sides). The pool now carries p_pub and p_repro explicitly.
C1a verdict: the frozen CNN has real but modest rank signal on the human-judged
HARD contrast — 0.646, far above LensJudge v1's ~0.5, well below deployment grade.

### C1b — representation probe: the wall is real for engineered features (DONE)
Gate (frozen, touched once; 2000-boot, seed 2026): **HARD A/B-vs-D AUC 0.425
[0.330, 0.517]** (every valsel grid point <0.5 too); EASY A-vs-random 0.655;
A/B/C-vs-random 0.716. Tier-1 engineered features carry ~no signal on the
human-reject wall — confirms and extends the v1 RepresentationKit result with a
properly trained/gated probe. Chosen: soft-target logistic, C=0.01, isotonic
(valsel), score = 0.995*iso + 0.005*raw (monotone). Bench scored blind:
arm1 912/912, arm2 2,605/2,605 -> outputs/rep_probe/bench_scores_arm{1,2}.csv;
model + features cache in outputs/rep_probe/. Machine bar on HARD remains the
frozen CNN's 0.646.

### C3 — matched-inputs grader (BUILT, dry-run validated)
`imaging/grader_matched.py` (+ `--mode matched` in run_batch.py): [full]+[zoom]
composites, [channels] per-band g|r|z montage (render.band_montage), [wide] 401px
~105" context (fetch_endpoint size param + get_wide_cube, distinct cache keys),
metadata block (tractor gloss, region, CNN score, spec-z/photo-z via huang2021
name/coord join, 5" aperture grz mags from the cutout — no local catalog has
mags), `prompts/rubric_matched.md` = five Huang criteria verbatim + relative-
triage framing. Delegates transport to grader_direct via pre-rendered-content
seam (shared JSON contract/cost/traces). Dropped vs direct: residual view +
aperture-color JSON (machine-only affordances humans lacked). Smoke: DRY RUN
6/6 (2A/2C/2D train-split, outputs/matched_smoke/) — no Messages-API key in this
env; live smoke = `parity/smoke_matched.py --live` on a credentialed host
(~<=$0.10). Mocked end-to-end grade path + repo no-API tests pass.

### C3 SELECTION (valsel slice, 149 rows, Sonnet 5 both arms): MATCHED WINS
Live smoke first: 6/6 parsed, $0.121 ($0.02/cand). Then matched-vs-direct on
outputs/c3_valsel_slice.csv (29A/30B/30C/40D/20rand, sha ebeeb688):
- HARD A/B-vs-D: matched **0.623** [0.518,0.725] vs direct 0.436 [0.312,0.553];
  paired ΔAUC **+0.151 [+0.020,+0.280]** — CI excludes 0.
- ALL: matched 0.640 vs direct 0.401; ΔAUC +0.161 [+0.040,+0.281].
**First statistically significant tier-1 lever in the program** (v1–v5 found
none): the human graders' information set (channels + wide context + metadata)
is what was missing. CAVEATS: (i) metadata includes the CNN score (fair for
parity — humans had it — but Phase D must carry CNN-alone as a baseline to
isolate LLM added value; CNN-alone gate HARD = 0.646 ≈ matched 0.623);
(ii) raw grades stay cold (QWK ~0.05; mean p_lens on true-A 0.09) — the score
ranks, the labels don't; deployment goes through the operating point.
Frozen selection: mode=matched, model=claude-sonnet-5, op point (valsel-fit)
p_lens>0.050 @ FPR 0.20 + isotonic map -> outputs/c3_matched_op_point.json.
Robustness: matched 149/149 rows scored; direct 115/149 usable (14 random_neg
never graded — photometry path failure; ~20 parse-soft). Cost ~$6 total.
Code: parity/build_c3_valsel_slice.py, parity/c3_select_analysis.py.

### C3 GATE (one-shot, 2026-07-13): the valsel win DID NOT REPLICATE
Frozen matched/Sonnet-5 on the 259-row gate (preds_gate_matched.parquet,
258/259, $4.36): **HARD A/B-vs-D AUC 0.487 [0.395, 0.577]** (valsel had 0.623
[0.518,0.725] — selection optimism on a 149-row slice; the disjoint-draw rule
caught it, again). ALL A/B/C-vs-D+random holds: 0.633 [0.566, 0.699]. Frozen op
point runs hot on gate grade-D (FPR 0.35 vs 0.20 target); QWK 0.025. Standing
conclusion: C3 is a real lens-vs-random vetter, NOT a HARD-contrast detector;
the matched-vs-direct relative gain (+0.15) remains valid but the absolute HARD
level is chance. The frozen-CNN 0.646 stays the only machine HARD signal so
far; C2's gate result decides Phase C.

### C2 — DESI 27B student: trained, selected, gate-evaluated (DONE 2026-07-13)
Corpus: 12,250 aug train / 189 val / 1,174 valsel (build_corpus_desi.py, leakage
self-check PASS). Training: Perlmutter job 55804993, r3 recipe verbatim
(Qwen3.5-27B bf16 LoRA + unfrozen ViT, 4-GPU DDP), 6h33m, RC=0, 11 ckpts.
Selection (job 55848953, merge+serve+valsel-grade 7 plateau ckpts; my inline
scorer crashed on schema — preds parquets carry no label col, join
valsel_manifest.csv; scored off-band): MONOTONE plateau, **BEST=ckpt-750**
(valsel A/B-vs-D+rand 0.852, HARD 0.659). GATE one-shot (job 55859082):
**HARD 0.644 [0.550, 0.733]** — held from valsel, NO collapse;
**A/B-vs-D+rand 0.817 [0.757, 0.873]**. Zero-shot control (same protocol):
HARD 0.530 [0.430, 0.621], A/B-vs-D+rand 0.576. Training delta +0.11 HARD /
+0.24 vetting — direct human-label SFT moves tier-1 where v4's Claude
distillation could not (teacher-signal argument doesn't bind when the teacher
is the human catalog). Artifacts on Perlmutter: $SCRATCH/ljdesi/
{ckpt_desi_27b/v0-*/checkpoint-750(-merged), preds_valsel_*.parquet,
preds_gate_student750.parquet, preds_gate_zeroshot.parquet}.

## PHASE C FINAL GATE SCOREBOARD (HARD A/B vs D, one-shot, frozen)
| system | HARD | A/B vs D+rand |
|---|---|---|
| student-27B ckpt-750 | **0.644** [0.550,0.733] | **0.817** [0.757,0.873] |
| frozen CNN (published p) | 0.646 [0.557,0.734] | — |
| zero-shot 27B | 0.530 | 0.576 |
| C3 matched Sonnet 5 | 0.487 | 0.633 (ALL incl. C) |
| rep-feature probe | 0.425 | 0.716 (ALL incl. C) |
Student ties the CNN on the wall; strongest vetting number in the program at
DESI resolution. Phase D predictors: human grade (primary comparator), CNN,
student-750 logprob, (C3 as documented reference). Phase D prerequisite:
serve ckpt-750 once more to score parity_bench arm1+arm2 (manifests+cutouts
need staging to Perlmutter, same label pipeline).

## Phase D — the parity comparison (DONE 2026-07-13)

Bench scored (Perlmutter job 55863819, ckpt-750, logprob): arm1 912/912,
arm2 2,605/2,605 -> outputs/preds_bench_arm{1,2}.parquet. Analysis:
parity/phase_d_analysis.py -> outputs/parity_phase_d.json.

**E2 PRIMARY (130 decided, 122/8; NI margin delta0=0.05, paired stratified
bootstrap + DeLong):**
- human 0.577 [0.400,0.755] | cnn 0.642 (n=105) | **student 0.685 [0.538,0.819]**
  | rep 0.660
- ΔAUC vs human: **student +0.108 [-0.028,+0.241] p=0.12 -> NON-INFERIOR**;
  **cnn +0.182 [-0.011,+0.425] p=0.13 -> NON-INFERIOR**; rep +0.084 -> inconclusive.
- Both pre-registered machine systems formally NON-INFERIOR to the expert grade
  on truth; point estimates favor machines; superiority NOT established.
- CNN-coverage note: decided rows are mostly Paper II candidates whose published
  probability lives in huang2021_published_catalog.csv (not the inchausti score
  CSVs); huang2020 publishes no probability (25 rows NaN -> n=105).

**E1 secondary:** student QWK 0.044 (n=162) — the machine does NOT reproduce the
letter grade (anchors 0.29/0.42/0.776). Score ranks; labels stay human.

**Arm2 scope limits:** student 0.509 [0.268,0.750] on the 44 SLDE-B-decided —
truth signal does NOT transfer to the Euclid-selected case mix; Euclid 0.1"
expert anchor 0.833 [0.600,1.000]; student score weakly monotone in Euclid
grade at scale (A 0.049 / B 0.035 / C 0.023 over 2,584 rows).

**Report:** papers/parity.tex (11 pp) — all 24 result macros filled, compiled,
registered as site page current/lensjudge-parity (strict build + visual QA
pass). Program conclusion: score-based vetting through calibrated operating
points is deployable at DESI resolution; grades stay human; a verdict at
delta0<=0.05 with real power awaits DESI STP outcomes / Euclid DR1 refutations.
