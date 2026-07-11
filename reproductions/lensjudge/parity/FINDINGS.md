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

## Next (Phase B)
`eval/build_parity_bench.py`: Arm 1 = parity_truth_master (decided rows) at DESI
resolution; Arm 2 = LS grz cutouts at Euclid Q1 positions (~500-900 truth-anchored);
negative-augmentation strata (grade-D 2,345 / random 65k); SHA-pinned manifest;
paired-bootstrap ΔAUC power check (reuse residual_ab_full_metrics machinery).
