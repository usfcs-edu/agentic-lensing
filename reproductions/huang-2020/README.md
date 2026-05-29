# Huang 2020 reproduction (Phases 3a + 3b)

Internal reproduction of \[Huang et al. 2020, *Finding Strong Gravitational
Lenses in the DESI DECam Legacy Survey*, ApJ 894:78,
[arXiv:1906.00970](https://arxiv.org/abs/1906.00970)\]. The full tech-report
PDF lives at [`papers/main.pdf`](papers/main.pdf); this README is a short
operator's guide and a results summary.

## Headline result

We re-implemented the Lanusse-2018 ResNet-46 (no Huang code) and ran the
same DECaLS DR7 deployment. Recovery of the published 342 lens candidates
(60 Grade A + 106 Grade B + 176 Grade C) of the 6,242,507 DR7 DEV/COMP
galaxies that pass the paper's z-mag ≤ 20.0 cut:

| Grade | DR9-trained recall (p ≥ 0.9) | **DR7-trained recall (p ≥ 0.9)** |
| :--- | ---: | ---: |
| A | 90.0% | **83.3%** |
| B | 68.9% | **64.2%** |
| C | 43.8% | **27.8%** |
| **ALL** | **59.6%** | **48.8%** |

The DR7-trained column is the **paper-exact** baseline (same DR for training
and deployment). The DR9-trained column has a ~10 pp recall advantage that
we attribute to test-set leakage via the L18 NeuraLens catalog — when
training is restricted to DR7-footprint coverage, ~338 NeuraLens positives
drop out, many of which were also published Huang+2020 candidates. See
`papers/main.tex` §11 for the full ablation discussion.

Candidate-pool size at p ≥ 0.9: 74,011 (DR9-trained) vs **25,792
(DR7-trained)** vs ~50,000 stated in the paper.

## Pipeline (in script-number order)

| Script | Purpose | Phase | Wall-clock |
| :--- | :--- | :---: | :---: |
| `01_lanusse_resnet.py` | PyTorch port of CMU DeepLens ResNet-46 | 3a | — |
| `02_filter_catalog.py` | Restrict NeuraLens catalog to L18-model rows | 3a | < 1 min |
| `03_download_decals_cutouts.py` | Fetch 101×101 grz cutouts at DR9 layer (training) | 3a | ~2 h |
| `03b_download_dr7_train_cutouts.py` | DR7 variant of `03` (DR7 layer + new output dir) | 3a-DR7 | ~2 h |
| `04_build_negatives.py` | Sample DESI-DR1 spectra for random negatives | 3a | < 1 min |
| `05_train_resnet.py` | 120-epoch training on DR9 cutouts | 3a | 25 min |
| `05b_train_resnet_dr7.py` | DR7 variant of `05` (writes `_dr7` suffixed outputs) | 3a-DR7 | 28 min |
| `06_write_reproduction_report.py` | Generates `papers/REPRODUCTION.md` | 3a | — |
| `07_plot_training_curves.py` | Training-curve + ROC + architecture figures | 3a | < 1 min |
| `08_smoketest_dr7.py` | Phase 3b decision-gate: 6 Grade-A's scored on DR7 | 3b | < 1 min |
| `09_download_dr7_sweeps.py` | All 292 DR7 sweep catalog FITS (~511 GB) | 3b | 51 min |
| `10_select_parent_sample.py` | Apply z<20 + DEV/COMP + NOBS≥3 cuts → 6.24M rows | 3b | 18 min |
| `11_stream_inference_dr7.py` | Endpoint-driven sweep (rate-limited; only used for smoke tests) | 3b | — |
| `11b_brick_inference_dr7.py` | **Brick-driven 2-shard sweep** (~200× faster than `11`) | 3b | 10 h 08 m |
| `12_merge_shards.py` | Concatenate per-shard parquets | 3b | < 1 min |
| `13_extract_huang2020_catalog.py` | pypdf-parse 342 candidates from paper PDF | 3b | < 1 min |
| `14_crossmatch_recovery.py` | Recovery-by-grade analysis for one run | 3b | < 1 min |
| `14b_recovery_comparison.py` | Side-by-side DR9-trained vs DR7-trained recovery | 3b | < 1 min |
| `15_diagnose_missing_seven.py` | Why 7 published candidates miss our cuts (z-mag, TYPE, NOBS) | 3b | < 1 min |
| `16_build_inspection_viewer.py` | Paginated HTML viewer of top-N candidates with Lupton-stretched thumbnails | 3b | < 1 min |

`11b` was the key engineering pivot. The naïve `11` design pulls one
cutout at a time from `legacysurvey.org/viewer/fits-cutout`, which caps at
~0.86 rows/sec server-side. Switching to per-brick `legacysurvey-NNNN-image-{g,r,z}.fits.fz`
downloads + local WCS-based slicing reaches ~1.7 bricks/sec/shard (~100
galaxies/sec scored) — a ~200× speed-up that turns 84 days into 10 hours.

## Key data files

Large files are gitignored; only the slim derived artefacts are committed.

| File | Committed? | Contents |
| :--- | :---: | :--- |
| `data/dr7_sweep/*.fits` | ✗ (~511 GB) | Raw DR7 sweep catalogs |
| `data/parent_dr7.parquet` | ✗ | 6.24M-row filtered parent sample |
| `data/checkpoint_best.pt` | ✗ | DR9-trained ResNet weights |
| `data/checkpoint_best_dr7.pt` | ✗ | DR7-trained ResNet weights |
| `data/inference_scores_dr9trained.parquet` | ✗ (183 MB) | All 6.24M scores (DR9-trained) |
| `data/inference_scores_dr7trained.parquet` | ✗ (183 MB) | All 6.24M scores (DR7-trained) |
| `data/inference_scores_dr9trained_p_ge_0.9.parquet` | ✓ (2.7 MB) | Top 74,011 (DR9-trained) |
| `data/inference_scores_dr7trained_p_ge_0.9.parquet` | ✓ (957 KB) | Top 25,792 (DR7-trained) |
| `data/huang2020_published_catalog.csv` | ✓ | 342 published candidates (name, RA, DEC, grade) |
| `data/recovery_matched_dr9trained.csv` | ✓ | Per-candidate nearest match + DR9-trained score |
| `data/recovery_summary_dr9trained.csv` | ✓ | Grade × threshold recovery counts |
| `data/recovery_compare.csv` | ✓ | DR9 vs DR7 trained side-by-side table |

## Reproducing from scratch

```bash
# Phase 3a (DR9 training) — ~3 h
./02_filter_catalog.py
./03_download_decals_cutouts.py --tier positives
./03_download_decals_cutouts.py --tier negatives
./04_build_negatives.py
CUDA_VISIBLE_DEVICES=0 ./05_train_resnet.py

# Phase 3a-DR7 (paper-exact retrain) — ~3 h
./03b_download_dr7_train_cutouts.py --tier positives --formats fits
./03b_download_dr7_train_cutouts.py --tier negatives --formats fits
CUDA_VISIBLE_DEVICES=0 ./05b_train_resnet_dr7.py

# Phase 3b (DR7 deployment sweep) — ~12 h total
./08_smoketest_dr7.py                          # gate
./09_download_dr7_sweeps.py --workers 4        # ~51 min
./10_select_parent_sample.py                   # ~18 min
./13_extract_huang2020_catalog.py              # < 1 min

# Run both checkpoints on both L4 GPUs (~10 h each, in parallel)
CUDA_VISIBLE_DEVICES=0 ./11b_brick_inference_dr7.py --shard 0 --gpu 0 --brick-workers 3 \
    --ckpt data/checkpoint_best.pt &
CUDA_VISIBLE_DEVICES=1 ./11b_brick_inference_dr7.py --shard 1 --gpu 0 --brick-workers 3 \
    --ckpt data/checkpoint_best.pt &
wait
./12_merge_shards.py
# rename inference_scores.parquet -> inference_scores_dr9trained.parquet

# Repeat for DR7-trained checkpoint, then:
./14b_recovery_comparison.py

# Build PDF
make -C papers pdf
```

## Tech-report

```bash
make -C papers pdf      # builds main.pdf (12 pages)
make -C papers clean    # remove latex artefacts
```

Source: [`papers/main.tex`](papers/main.tex); shared preamble at
[`../tech-report.sty`](../tech-report.sty).

## Caveats (short form, see paper §7 + §11)

1. **No Huang code released.** Reproduction is from-scratch off the
   published methodology + L18 architecture.
2. **Test-set leakage in Phase 3a (DR9-trained).** Positives come from
   the L18 NeuraLens catalog, which mixes Huang's training inputs with
   later discoveries. The 0.9991 test AUC is biased upward.
3. **No hand-curated hard negatives.** Huang's 3,000 by-eye-selected
   negatives (spirals, galaxy groups, cosmic rays, artefacts) were not
   publicly released. We use uniformly-random DR1 negatives instead, which
   makes Grade-C recovery harder (43.8% DR9, 27.8% DR7).
4. **DR7 footprint coverage.** 7 of 342 published candidates fall outside
   the DR7 sweep parent sample, so the achievable ceiling is 335/342.
   `15_diagnose_missing_seven.py` confirms: 5/7 fail z-mag ≤ 20.0 by a
   tight margin (z-mag 20.22–20.78), 2/7 are Tractor TYPE=EXP, 1/7 has
   NOBS_G=2. All 7 are present in DR7 imaging within 0.8″ of the
   published position — these are honest cut-edge effects, not pipeline
   failures.

## Visual-inspection viewer

After Phase 3b produced the ranked candidate pool, `16_build_inspection_viewer.py`
generates a paginated HTML viewer with Lupton-stretched (Q=8, stretch=0.5)
grz thumbnails:

```bash
./16_build_inspection_viewer.py --top-n 2000 --per-page 50
open papers/figures/inspection/index.html
```

Top 2,000 DR7-trained candidates at p ≥ 0.998 → 40 pages of 50 tiles.
Published Huang+2020 candidates (within 5″) are flagged with ★. 84/2000
(4.2%) of the top scorers are already in the published catalog; the rest
are candidates for visual grading.
