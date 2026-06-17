# DR11 Campaign — running findings

Living log of the full DR11 strong-lens search: ClaudeNet **v3blend8** finder (DECam-south +
BASS/MzLS-north retrained) → **LensJudge v3** cascade vetting (HSC PDR3 + Euclid + SuGOHI
escalation) → cross-validated candidate/confirmed catalog. Branch
`claudenet-v3-dr11-campaign`. Data under `data/`/`outputs/` is gitignored (regeneratable);
this file + the scripts + the report are the tracked record. Plan:
`~/.claude/plans/i-want-to-do-melodic-clover.md`.

Compute: Perlmutter `gdbenson`, charge **cosmo** (CPU) / **cosmo_g** (GPU, uncapped). LLM
vetting budget **~$250** (LensJudge). venvs `/home2/benson/.venvs/{claudenet,lensjudge}`.

## Key verified facts (campaign start, 2026-06-16)

- DR11 on CFS `/global/cfs/cdirs/cosmo/data/legacysurvey/dr11/`: **south** sweep 11.0 = 1,600
  files / 1.7 TB, native **griz**; **north** = 526 files / 171 GB, **grz only** (BASS g,r +
  MzLS z, no i). Confirmed north high-dec brick (1801p400) has g,r,z.
- v3blend8 = 8 members; ckpts present local + Perlmutter (`data/v2/ckpt_lean/*` ×5 +
  `data/v2/ckpt/*_b50.pt` ×3).
- Pipeline release-parametrized (`360/111/315/365 --release dr11 --footprint {south,north}`).
- North training positives ≈ 265 catalog (Storfer 80 + Huang 183 + Inchausti 2, dec>32.375) +
  297 BASS/MzLS-native curated already on disk; **33 catalog north lenses leak into v1 train**.
- **Scratch quota 20 TB, 7.1 TB used** (DR10 4.9 TB + DR9 ~2 TB prior campaigns) → ~12.9 TB
  free. DR11-south ~66 M gal ≈ 8 TB cutouts → **delete each part's cutouts after stage-1
  scoring** to bound peak; don't touch prior-campaign data unless quota forces it.

### Three load-bearing correctness forks
1. **DR9→DR11 threshold drift** — DR9 1e-4 thresholds undershoot on DR11; run 370-style
   DR11-native recalibration first, feed `operating_points_dr11.csv` into 162.
2. **121 never substitutes positives** (`121:53,61`) — north retrain needs north-instrument
   positives → patch 121 (`--base-table/--init-ckpt/--out-suffix`), fine-tune from `_b50`.
3. **LensJudge cutout provenance** (`fetch.py:138` defaults ls-dr10) — stage real DR11 grz
   cutouts into `CUTOUT_DIRS["dr11"]` so the grader sees the CNN-seen pixels.

## Phase 0 — setup + regression gates  🟢

- Branch `claudenet-v3-dr11-campaign` off main (has all v3 + LensJudge-v3 scripts).
- Env verified: claudenet/lensjudge venvs (torch 2.6 local / 2.8 PM), ANTHROPIC_API_KEY +
  HSC_USER + HSC_PASSWORD set, Perlmutter reachable (cosmo/cosmo_g), 27 G HF cache staged.
- Scripts synced to Perlmutter `$HOME/claudenet` (flat copy; 365 + dr11_extract.slurm added).
- **G0 PASS** — `365` reproduces D5 EDF-F/S exactly (n=129721; grade-A top100/500/1000 =
  5/8/10; precision@30 = 5 Euclid-A/B; shortlist 60/59/54). Select toolchain intact.
- **G0b PASS** — `165 --synthetic-check` 14/14 (FDR arithmetic).
- G0c (112 --self-test on A100) — caught a real sync bug (below), re-running after fix.

**GOTCHA (sync):** 6 claudenet `.py` are symlinks into `../inchausti-2025/` (`_scorelib`,
`_trainlib`, `02_efficientnet`, `01_lanusse_resnet`, `01b_shielded_resnet`, `03_meta_learner`).
Perlmutter's flat `$HOME/claudenet` has no sibling `inchausti-2025/`, so **always sync with
`rsync -L`** (dereference → real files). `rsync -a`/`-az` copies them as dangling symlinks and
silently breaks all scoring/training (`ModuleNotFoundError: _scorelib`). The G0c self-test is
the canary — run it after every Perlmutter script sync.

## Phase S — DR11-south full sweep  🟡 in progress

- **360 parent DONE** (JID 54619948, 42 min): **53,809,040 galaxies** (51.4 M native-i;
  TYPE SER 24.5M/REX 12.0M/DEV 11.0M/EXP 6.2M; mag_z median 19.36). 2.38 GB parquet at
  `$SCRATCH/claudenet/sweep_dr11/parent_dr11_south.parquet`. Larger than DR10's 43.7 M (deeper
  DR11). Full extraction ≈ 6.6 TB → peak ~13.7/20 TB; extract all 32 parts, delete each after
  stage-1 scoring.
- **160 manifest DONE**: 32 footprint-pure parts (~1.68M each).
- **111 extraction DONE** (array 54621016): 32/32, **100% ok-fraction** (part00 1,681,558/1,681,559),
  **6.0 TB** cutouts, ~10 min/part, zero errors.
- **161 stage-1 scoring DONE** (array 54621139, aftercorr-pipelined): 32/32, 5 v2-lean members,
  ~54 min/part, no errors. 32 score parquets (`scores/sweep_dr11_south/stage1_part*_lean_*.parquet`).
- **370 DR11 recalibration RUNNING** (JID 54626515): applies isotonic cals to all 53.8M, draws 8M
  NegEval → DR11-native 1e-4 thresholds → recalibrated union survivors (budget 150k) +
  `negeval_dr10_combined.parquet` (for Phase F 165) + recall-of-811. Sweep-dir
  `$SCRATCH/claudenet/sweep_dr11/south` (stage1+manifest symlinks). Watcher armed.
- **370 recalibration DONE** (6 min): DR11-native 1e-4 thresholds are **TIGHTER** than DR9
  (effnet_S2_hard 0.962→0.977, effnet_B3_hard 0.972→0.980, **resnet46_C_hard 0.810→1.000**
  = saturates on DR11 randoms → effectively a 4-member union; zoobot looser). **95,104 survivors**
  (1.77e-3 pass, under the 150k budget). 8M NegEval written (`negeval_dr10_combined.parquet`) for
  Phase F. **Recall-of-811 (G6): grade-A 41% (37/90), B 31%, all 30%** — vs DR10's 62% recal.
  *Finding:* genuine DR9→DR11 domain shift — deeper DR11 → fatter random high-score tails → tighter
  1e-4 thresholds → lower known-lens recall at the operating point (denominator includes north +
  parent-cut lenses, so a conservative floor). Principled high-purity pool; v3blend8 re-ranks +
  vetting work on these 95k. May revisit (loosen FPR within budget) if candidate yield is thin.
- **315 survivor scoring RUNNING** (JID 54628241): 8 v3blend8 members on the 95,104 survivors via
  `--row-ids` (reads existing part cutouts, no re-extract). Watcher armed.
- **315 survivor scoring DONE** (after fixing a `$SCRATCH`-in-glob quoting bug via dedicated
  `nersc/dr11_survscore.slurm`): 8 v3blend8 members on 95,104 survivors, 5 min.
- **365 select DONE → DR11-south candidate list**: top-300 by v3blend8 (0.812–0.939) = **102 known +
  198 NEW**; v3blend8 ranks 102 known lenses into the top-300 (consistency ✓). Broader pool: 1,071
  survivors >0.7, 356 >0.8. Artifacts `data/v3/cv3_dr11south_candidates.csv`. **Euclid Q1 overlap of
  the GLOBAL survivor set is sparse** (only 4 grade-A + 2 grade-C in all 95k, 0 in top-100) — the
  global 1e-4 cut is too strict to retain EDF-region Euclid lenses (D1 finding); the Euclid-confirmable
  DR11-south set is D5's region-local 41. **For the full-south list, HSC-SSP Wide (~1300 deg²) is the
  main confirmation lever (Phase V).**

## Phase F — certified-FDR NegEval  🟢 DONE (honest power-floor result)

165 group-conformal on the 8M DR11-south NegEval (4M conformal split) + 95,104 survivors, m=53.8M:
**full-m BH selects 0 at α=0.05/0.10/0.25.** Power-limited (floor 1/(n_cal+1)≈2.5e-7 is ~50× short of
certifying k=1 at m=53.8M) — NOT a null; the rigorous full-m FDR needs infeasibly large calibration,
so the candidate list is the v3blend8 ranking (science list), reported separately. Gotcha hit+fixed:
`--calibration`/`--cal-manifest` share `footprint` → collision; split NegEval into
`negeval_cal[row_id,v2lean_average]` + `negeval_manifest[row_id,footprint]`. Artifacts
`$SCRATCH/.../south/{conformal.parquet,conformal_summary.json}`.

## ⚠️ Perlmutter 7-day maintenance (2026-06-17 06:00 → 06-24 06:00)

Discovered mid-campaign (`maintenance_20260617`, full-system). The SOUTH sweep finished all its
Perlmutter work BEFORE the window (parent/extract/score/survivor-score/candidate-FITS all done);
**south vetting + report run locally (Anthropic API + HSC), maintenance-immune → completable now.**
The NORTH workstream (extract→score→retrain→sweep, multi-hour GPU+CPU) **cannot fit before 06:00**
(north jobs hit `ReqNodeNotAvail`; cancelled). **North is DEFERRED to post-maintenance** — all prep
staged for a clean resume (below).

**North resume recipe (post-2026-06-24):** parent `$SCRATCH/claudenet/sweep_dr11/parent_dr11_north.parquet`
(11.6M) + 8-part manifest `manifest_north/` + 523 positives (`cutouts/north_pos_dr11` + FITS via 120b)
all staged. Resume: `dr11_extract.slurm FOOT=north NPARTS=8 MANDIR=manifest_north` → `dr11_score.slurm
FOOT=north` (aftercorr) → mine north mimics/negatives from north scores → `19b` base tables + `314`
north blend → patched `121 --base-table --init-ckpt _b50 --out-suffix _north` (3 members) → validate
gates G1–G5 → north select. (Phase A model re-iteration also deferred — Perlmutter GPU.)

## Phase V — LensJudge v3 cascade vetting (south)  🟡 in progress

- 500 NEW candidate FITS staged (`dr11-campaign/data/cutouts_dr11`, FORK-3); cutout resolution +
  v3 unit tests (9/9) pass; HSC + Anthropic API reachable (legacysurvey NOT → on-disk staging).
- **Pilot (40, pass_frac 0.5, $2.74) ✅**: full path works — 20 escalated, **3 reached HSC tier-2,
  all 3 flipped to grade A** (s_336801_4965 0.80, s_351303_461 0.88, s_318982_1451 0.95) + a DESI-only
  A (s_292418_3907 0.80) + several B; mimics → D (p≈0.04). HSC `das_cutout` tier-2 confirmed working.
- **Full south cascade DONE** (500, pass_frac 0.7, $40.38): 350 escalated, **26 reached HSC tier-2
  → 24 confirmed grade-A/B (19 A + 5 B)**. Grade dist 272 D / 125 C / 76 B / 27 A; 103 DESI-grade A/B
  overall. DESI→HSC p_lens flips dramatic (e.g. 0.03→0.95, 0.04→0.92) — the resolution lever converts.
- **Confirmed catalog (`data/dr11s_confirmed_AB.csv`):** of the 24 HSC-confirmed A/B —
  **15 cross-validated with SuGOHI** (v3blend8 ∩ LensJudge-HSC-A/B ∩ SuGOHI committee; = the
  independent validation anchor: the HSC grader recovers 15 known HSC lenses), **0 in DESI catalogs**
  (all new to storfer/inch/huang), **9 genuinely NEW** (new to DESI lens-finders AND SuGOHI; 8 A + 1 B):
  s_340513_3688, s_327526_9059, s_351303_461, s_345385_1515, s_340971_2860, s_345385_1514,
  s_326089_1785, s_325163_2981 (A) + s_318043_1037 (B). These 9 are the discovery payoff — pending
  human-expert + spectroscopic follow-up (LensJudge HSC grading is automated; single-grader caveat).
- **SuGOHI independent recovery:** v3blend8 pulls **79 SuGOHI HSC lenses** into the 95k survivors
  (median v3blend8 2.7× the survivor median; 27 in top-500) — independent evidence v3 ranks real lenses
  high. **Campaign LLM spend $43.12 / $250.**

Next: active-learning export (local) → report. (Phase A model re-iteration + north = Perlmutter, deferred.)

## Phase F — certified-FDR NegEval  ⚪ pending
## Phase N — DR11-north retrain + sweep  🟡 prep (north positive pool locked)

North trainable positive pool (analyzed 2026-06-16, local):
- **297 north-native curated** positives already on disk (`cutouts_fits_curated_dr9`,
  viewer auto-selected BASS/MzLS for dec>32.375), split 211 train / 53 val / 33 test.
- **264 unique catalog north** lenses (Storfer 80 + Huang 183 + Inchausti 2, 3″ dedup); **33
  leaked** into v1 (26 train/5 val/2 test, = the 33 already in curated) → **231 non-leaked**
  catalog north to extract fresh from DR11-north coadds.
- **Union ≈ 528 north positives**; clean leakage-free held-out for the G4 recall gate = the
  **86 curated north val/test** (53+33). Recipe: fine-tune 3 swappable members from `_b50`
  (patched 121), north-native mimic-blend negatives, D4 8× positive augmentation.
## Phase V — LensJudge v3 cascade vetting  ⚪ pending
## Phase A — active-learning loop  ⚪ pending
## Phase R — DR11 campaign report  ⚪ pending
