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

## Next
- Rebuild arm2 truth join when euclid_q1_{nisp,modeling}.csv land; then Phase C
  (three machine systems; gate on disjoint benches, never select on parity bench).
