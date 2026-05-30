---
name: project-huang-2021-reproduction
description: Huang+2021 (shielded ResNet, arXiv:2005.04730) reproduction — Phase 4a architecture done; DR8 two-model deployment in progress
metadata:
  type: project
---

# Huang+2021 (arXiv:2005.04730) reproduction — Phase 4

Lives at `/raid/benson/git/agentic-lensing/reproductions/huang-2021/`. The
successor to [[project-huang-2020-reproduction]] in the lens-finder lineage.
Training inputs (cutouts, positives, negatives, L18 checkpoints, NeuraLens
catalog) are **symlinked** from `../huang-2020/data/` — not re-downloaded.

## Phase 4a — shielded architecture (DONE, committed 3d3b9ef)

The headline novelty: 1×1 conv "shield" layers inserted between every three
L18 residual blocks (`01b_shielded_resnet.py`, `ShieldedDeepLens`). Verified
reconstruction: **59,905 params (58.6× reduction from L18's 3,508,833),
4 shields, 15 blocks, final stage = 32 channels** — matches the paper's
"~60K params / ~50× / 32 best". `SHIELD_CH=16` is the one free knob (paper
doesn't give intermediate channel counts).

Controlled comparison (same cutouts/seed/split as the L18 baseline; only the
architecture changes), via `05_train_shielded.py --dr {dr9,dr7}` +
`06_compare_architectures.py`:

| Run | params | val AUC | test AUC |
| :-- | --: | --: | --: |
| L18 / DR9 | 3,508,833 | 0.9983 | 0.9991 |
| L18 / DR7 | 3,508,833 | 0.9890 | 0.9943 |
| shielded / DR9 | 59,905 | **0.9989** | 0.9988 |
| shielded / DR7 | 59,905 | 0.9875 | **0.9955** |

Shielded **matches L18 within ±0.002 AUC at 59× fewer params** — reproduces
the paper's claim (cut ~98% of params without hurting AUC). We don't reproduce
the absolute 0.992→0.997 *increase* because our leakage-inflated AUCs already
sit at ~0.99 (no headroom). Shielded DR9 trains in ~15 min on one L4.

## Phase 4b/4c — DR8 two-model deployment (DONE 2026-05-30)

Full sweep: **17,290,814 DEV/COMP galaxies** (12.27M south + 5.02M north),
298,843 brick-units, 0 failures, ~16h on the 2 L4s with the *northaug*
checkpoints. Merged scores well-calibrated (L18 2.9% / shielded 3.8% ≥0.1).
Recovery of the 1,312 published (combined max), leaked vs honest:
- all 1,312: p≥0.9 76.1%
- leaked 949 (= training positives): p≥0.9 **86.0%** (circular)
- **leak-free 363 (shielded discoveries): p≥0.9 50.4%** (honest; Grade-A 57.9%)
The 36pp gap directly measures the leakage. Shielded edges L18 on the leak-free
set (47.7 vs 44.9% all-grade @0.9). Top-2000 extended xmatch: 86.2% unmatched
(cf DR7 89%). 8-page tech-report `papers/main.pdf` complete. Below = the
pipeline as run.

The first Huang-group search this repro extends into the **northern BASS/MzLS
footprint** (δ≳+32°) that the DR7 Phase 3b lacked. Pipeline:
- `09_download_dr8_sweeps.py`: 723 sweeps (437 south `dr8/south/sweep/8.0/` +
  286 north `dr8/north/sweep/8.0/`), **777 GB total, 0 failures**.
- `10_select_parent_sample_dr8.py`: DEV/COMP + NOBS≥3 + z<20, tags each row
  `footprint` (south=RELEASE 8000–8004, north=9010) by source directory.
  Paper expects ~15.4M DEV/COMP; REX is an optional secondary pass.
- `11b_brick_inference_dr8.py`: footprint-aware `brick_url` →
  `dr8/{south,north}/coadd/`; scores BOTH L18 + shielded per brick in one
  download pass (net is the bottleneck); shards by md5(footprint/brick); run
  4 shards across the 2 L4s (~20–35h). keep-thresh 0.1 (paper operating point).
- `13_extract_huang2021_catalog.py`: rebuilds the published 1,312 catalog from
  `neuralens_catalog.csv` **exactly** — 216 A + 199 B + 897 C, per-model split
  949 L18 / 363 shielded, 185 north MzLS. Grade = visual Score (A≥3.5, B=3.0,
  C≤2.5).
- `14_crossmatch_recovery_dr8.py`: recovery by grade×model×{0.1,0.5,0.9},
  splitting **leaked** (≈949 L18 rows = our training positives) from
  **leak-free** (363 shielded-model discoveries — the honest test).

## Non-obvious facts

1. **The NeuraLens 1,312-catalog IS the Huang+2021 output**, labeled by which
   of the two deployed models (L18/shielded) flagged each. The 949 "L18" rows
   are exactly our training positives (from the huang-2020 repro) → leakage;
   the 363 "shielded" rows were never trained on → leak-free recovery target.
2. **North-negative calibration is essential (verified 2026-05-29).** The
   Phase-4a south-trained models had ~160 north (MzLS) *positives* but ZERO
   north *negatives* (negatives were 99% southern DR1). Result: on DR8 *north*
   (BASS/MzLS) imaging the L18 model OVER-FIRES — 91% of non-lenses score ≥0.1
   (median 0.45) vs 4.8% in south. The shielded model resisted (10% — a nice
   confirmation of the shielding-regularization thesis). Fix (`18_` + `05c_`):
   add 154 north positives (re-grabbed at DR8) + 2883 north negatives, retrain
   both → north L18 ≥0.1 drops 91%→0.8%, shielded 10%→1.4%, south unchanged;
   in-domain AUC stays 0.9985 (L18) / 0.9996 (shielded). Deployment uses
   `checkpoint_best_{l18,shielded}_northaug.pt`. ALWAYS include north negatives
   when deploying a south-trained finder on BASS/MzLS.
3. **North DR8 coadds are only ~50% grz-complete** at the brick level (others
   are gr/g/r-only), but parent-sample galaxies require NOBS_z≥3 so live only
   in z-covered (hence grz-complete) bricks — partial bricks never enter the
   sweep. `download_brick` drops a brick if any band 404s.
4. CUDA device-ordering gotcha → see [[reference-host-hardware]]
   (`CUDA_DEVICE_ORDER=PCI_BUS_ID`; L4s are indices 8,9).

Tech-report `papers/main.pdf` (14 pp): Phase 4a complete; DR8 recovery section
populated on deployment completion. Venv `/raid/benson/.venvs/lensfinder`.
