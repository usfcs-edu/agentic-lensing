---
name: project-huang-2020-reproduction
description: Huang+2020 reproduction status — Phase 3a + 3b + DR7 ablation done; brick-driven pipeline pivot was the key engineering finding
metadata:
  type: project
---

# Huang+2020 (arXiv:1906.00970) reproduction — Phases 3a + 3b complete

Lives at `/raid/benson/git/agentic-lensing/reproductions/huang-2020/` (scripts
01-16). Three completed runs:

- **Phase 3a (DR9-trained)**: PyTorch port of Lanusse-2018 CMU DeepLens
  ResNet-46 on 929 NeuraLens positives + 4,908 DR1-zcat negatives at
  DR9 cutouts. test_auc = 0.9991 (vs paper's 0.98) in 25 min on one L4.
- **Phase 3b (DR9-trained deployment)**: brick-driven full DECaLS DR7
  sweep (6,242,507 DEV/COMP galaxies at z-mag ≤ 20.0) using the DR9
  checkpoint. 90% Grade-A recovery at p ≥ 0.9.
- **DR7-trained ablation**: same architecture retrained on DR7-cutouts at
  the same positions (591 positives — many NeuraLens lenses fall outside
  DR7's Dec ≤ +32° footprint). val_auc = 0.989, test_auc = 0.9943.
  Headline 83.3% Grade-A recovery at p ≥ 0.9 — the paper-exact baseline.

Tech-report at `papers/main.pdf` (12 pages); operator's guide at
`reproductions/huang-2020/README.md`. See sibling phases
[[project-foundry-i-reproduction]] and [[project-hsu-2025-reproduction]]
for the surrounding context.

## Headline recovery (vs Huang+2020 published 342 = 60 A + 106 B + 176 C)

| Grade   | DR9-trained p≥0.9 | DR7-trained p≥0.9 |
| :------ | :---------------: | :---------------: |
| A       | 90.0%             | **83.3%**         |
| B       | 68.9%             | 64.2%             |
| C       | 43.8%             | 27.8%             |
| **ALL** | **59.6%**         | **48.8%**         |

The ~10pp DR9 advantage is test-set leakage via the L18 NeuraLens
catalog — paper-exact (DR7-trained) is the honest baseline.

## Non-obvious engineering facts

1. **Brick-driven >>> cutout-endpoint at scale.** The
   `legacysurvey.org/viewer/fits-cutout` endpoint caps at ~0.86 rows/sec
   server-side (CPU-bound cutout generation), regardless of worker count.
   Per-brick FITS downloads (~45 MB per brick × 113K bricks) + local
   WCS-based slicing reaches ~1.7 bricks/sec/shard = ~100 galaxies/sec/shard.
   This is a ~200× speedup. The two L4s ran in parallel; the full 6.24M
   sweep finished in 10h on both checkpoints. See script `11b_brick_inference_dr7.py`
   and [[reference-legacysurvey-bulk-download]].

2. **DR9-vs-DR7 calibration shift is small for known lenses.** Smoke test
   (`08_smoketest_dr7.py`) on 6 Grade-A's at DR7 with DR9-trained
   checkpoint: all scored 0.946-1.000, mean 0.984. So the DR9-trained
   model generalizes to DR7 imaging without retraining. The
   DR7-trained ablation isn't necessary for the pipeline to work — it's
   necessary for **methodological honesty** about test-set overlap.

3. **The "7 missing" published candidates aren't pipeline failures.** All 7
   have a DR7 source within 0.8″ of published position; they just fail
   the paper's own cuts. Diagnostic at `data/missing_seven_diagnostic.csv`:
   - 5/7 fail z-mag ≤ 20.0 by tight margin (z-mag 20.2-20.8)
   - 2/7 fail TYPE = DEV/COMP (Tractor classifies them EXP in DR7)
   - 1/7 fails NOBS ≥ 3 (NOBS_G = 2)
   The paper's stated "92% of known lenses pass z-mag ≤ 20.0" predicts
   ~8% near-miss at the cut; we measure 5/342 = 1.5%, well within that.

4. **AUC 0.99+ is reachable in 25 minutes on a single L4** with the Lanusse
   ResNet, 949 positives + 5000 negatives, batch=128, 120 epochs. The
   architecture is small (3.5M params); the deployment is the long pole,
   not training.

5. **The NeuraLens catalog has 1,312 rows split L18 (949) / shielded (363).**
   For Phase 3a training we use all 949 L18 rows as positives. The 342
   Huang+2020 candidates are partially derived from L18 → test-set leakage
   in Phase 3a. Phase 3b (DR7-trained) intentionally re-trains with
   DR7-coverage-only positives (591) to bound this leakage; recovery drops
   ~10pp as expected.

6. **DR7 sweep parent-sample sanity numbers.** Total 6,242,507 rows
   (88% DEV, 12% COMP, median z-mag 18.99) from 292 sweep catalogs
   (~511 GB). Within ~10% of the paper's stated 5.7M.

7. **DR7 brick FITS structure**: `legacysurvey-NNNN-image-{g,r,z}.fits.fz`
   at `portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr7/coadd/AAA/NNNN/`
   where AAA = first 3 chars of NNNN. Each brick is 3600×3600 px float32,
   ~15 MB compressed per band, WCS in CompImageHDU header (HDU 1).

8. **FITS columns are big-endian → torch is little-endian** — wrap every
   cutout-read with `arr.astype(arr.dtype.newbyteorder("="))` for numeric
   columns or pyarrow conversion fails. Strings need
   `np.char.strip(arr).astype(object)` because the FITS-fixed-width pad
   doesn't round-trip cleanly to pandas/parquet equality.

## Layout (current)

```
reproductions/huang-2020/
  01_lanusse_resnet.py            # PyTorch ResNet-46 port
  02_filter_catalog.py            # NeuraLens CSV → positives parquet
  03_download_decals_cutouts.py   # Phase 3a DR9-layer cutout puller
  03b_download_dr7_train_cutouts.py # DR7 variant of 03
  04_build_negatives.py           # DR1 zcat → negatives parquet
  05_train_resnet.py              # Phase 3a training (DR9, writes checkpoint_best.pt)
  05b_train_resnet_dr7.py         # DR7 variant (writes checkpoint_best_dr7.pt)
  06_write_reproduction_report.py # papers/REPRODUCTION.md (early markdown report)
  07_plot_training_curves.py      # training-curve + ROC + arch figures
  08_smoketest_dr7.py             # Phase 3b decision gate
  09_download_dr7_sweeps.py       # 292 DR7 sweep catalogs (~511 GB)
  10_select_parent_sample.py      # filter to 6.24M-row parent_dr7.parquet
  11_stream_inference_dr7.py      # endpoint-driven (rate-limited; smoke tests only)
  11b_brick_inference_dr7.py      # brick-driven 2-shard production sweep
  12_merge_shards.py              # concat per-shard parquets
  13_extract_huang2020_catalog.py # pypdf parse paper Tables 1-3
  14_crossmatch_recovery.py       # recovery analysis for one run
  14b_recovery_comparison.py      # side-by-side DR9 vs DR7 trained
  15_diagnose_missing_seven.py    # why 7 published candidates fail our cuts
  16_build_inspection_viewer.py   # paginated HTML viewer of top-N
  papers/main.tex / main.pdf      # 12-page tech-report
  papers/figures/                 # recovery + score-hist + inspection viewer
  data/*                          # mostly gitignored, slim derived CSVs committed
```

Venv: `/raid/benson/.venvs/lensfinder` (PyTorch 2.12 + CUDA 12.6,
sees all 10 GPUs; needs pypdf for catalog extraction).

## Remaining work (not started)

- **Phase 3c (BASS/MzLS transfer learning)** — Huang+2020 §2 mentions this
  as future work. Extends deployment to northern footprint.
- **Visual grading of top-N candidates** — viewer at
  `papers/figures/inspection/index.html` (top 2000 thumbnails) is built
  but no human pass has been done. Without it we can claim "recovery vs
  published" but not "new candidates discovered."
- **Huang+2021 reproduction** (shielded ResNet) — natural Phase 4.
