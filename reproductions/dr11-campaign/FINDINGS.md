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
- **160 manifest** (JID 54620826, --chunk 32) RUNNING. Watcher armed.
- Next: 111 extract array (dr11_extract.slurm, FOOT=south NPARTS=32) → 161 stage-1 (5 lean) →
  370 recalibrate (DR11 ops + 8M NegEval) → 162 survivors → 111 survivors → 112+315 score 8 →
  363/365 select → 163/164 crossmatch+list.

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
